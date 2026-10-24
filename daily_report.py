"""
daily_report.py
---------------
گزارش دوره‌ای خودکار (روزانه/ساعتی): وضعیت چند نماد + جمع‌بندی AI + Order Book.

همه‌چیز از همان پایپ‌لاین واقعی پروژه استفاده می‌شود (core_pipeline + ai_brain +
orderbook) — هیچ عدد ساختگی در کار نیست. خروجی: فایل Markdown در پوشه‌ی reports/
+ خلاصه در ترمینال + (اختیاری) ارسال به تلگرام.

کاربردها:
    python daily_report.py                          # یک بار اجرا (پیش‌فرض 4h)
    python daily_report.py --timeframe 1h           # تایم‌فریم دلخواه
    python daily_report.py --schedule 8             # هر ۸ ساعت در یک حلقه
    python daily_report.py --telegram               # ارسال خلاصه به تلگرام
    python daily_report.py --symbols BTCUSDT,ETHUSDT
"""

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

# غیرفعال کردن پراکسی برای ارتباطات محلی (مثل main.py — جلوگیری از ارور 503)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from config.settings import Config
from core_pipeline import analyze_symbol, AnalysisError
from strategies.ai_brain import AIBrain
from orderbook import load_latest_orderbook_features

logger = logging.getLogger("DailyReport")

REPORTS_DIR = "reports"

DIM_LABEL_FA = {
    "trend": "روند",
    "momentum": "مومنتوم",
    "volatility": "نوسان",
    "volume": "حجم/نقدینگی",
}
BIAS_FA = {
    "bullish": "صعودی",
    "bearish": "نزولی",
    "neutral": "خنثی",
    "mixed": "مختلط",
    "unknown": "نامشخص",
}


# ----------------------------------------------------------------------
# ابزارهای قالب‌بندی و پاک‌سازی
# ----------------------------------------------------------------------
def _fmt_num(x) -> str:
    """نمایش عدد با جداکننده‌ی هزارگان؛ np.float64 و رشته‌ی عددی را هم می‌پذیرد."""
    try:
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return str(x)


# کاراکترهای CJK (چینی/ژاپنی/کره‌ای) — نشانه‌ی پژواک قالب توسط مدل
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _clean_watch(item) -> str:
    """
    پاک‌سازی آیتم watch: حذف براکت‌های زاویه‌دار و پژواک قالب
    (مثل «یک چیز ... به تماشا» یا متن چینی). اگر بی‌ارزش شد None برمی‌گرداند.
    """
    if not item:
        return None
    item = str(item).strip().strip("<>").strip()
    if _CJK_RE.search(item):
        return None
    if "یک چیز" in item:
        return None
    return item


# ----------------------------------------------------------------------
# تلگرام (اختیاری)
# ----------------------------------------------------------------------
def _telegram_configured() -> bool:
    """فقط اگر توکن/چت‌آیدی واقعی (غیر placeholder) در .env ست شده باشد."""
    token = getattr(Config, "TELEGRAM_BOT_TOKEN", "") or ""
    chat_id = str(getattr(Config, "TELEGRAM_CHAT_ID", "") or "")
    if not token or "your_" in token.lower():
        return False
    if not chat_id or "your_" in chat_id.lower():
        return False
    return True


def send_telegram(text: str) -> bool:
    """ارسال متن به تلگرام. اگر کانفیگ نباشد بی‌صدا رد می‌شود."""
    if not _telegram_configured():
        logger.info(
            "Telegram پیکربندی نشده (توکن/چت‌آیدی placeholder است) — ارسال نادیده گرفته شد."
        )
        return False
    try:
        import requests

        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id": Config.TELEGRAM_CHAT_ID,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        resp.raise_for_status()
        logger.info("گزارش خلاصه به تلگرام ارسال شد.")
        return True
    except Exception as e:
        logger.error(f"خطا در ارسال به تلگرام: {e}")
        return False


# ----------------------------------------------------------------------
# تحلیل یک نماد
# ----------------------------------------------------------------------
def analyze_one(ai: AIBrain, symbol: str, timeframe: str) -> dict:
    """اجرای پایپ‌لاین کامل برای یک نماد و ساخت ردیف خلاصه‌ی گزارش."""
    result = analyze_symbol(symbol, timeframe, klines_limit=250)

    # آخرین وضعیت Order Book (فقط اگر تازه‌تر از ۵ دقیقه باشد)
    ob_context = load_latest_orderbook_features(symbol, max_age_seconds=300)

    opinions = ai.analyze_with_consensus(
        symbol,
        result["technical_summary"],
        result["market_profile"],
        active_patterns=result.get("active_candle_patterns", []),
        chart_patterns=result.get("recent_chart_patterns", []),
        orderbook_context=ob_context,
    )

    mp = result["market_profile"]
    consensus = opinions.get("consensus", {})
    bias = consensus.get("bias", "unknown")

    # حذف آیتم‌های watch بی‌ارزش (پژواک قالب / کاراکترهای نامربوط)
    watch_items = []
    for item in opinions.get("watch_items", []):
        cleaned = _clean_watch(item)
        if cleaned:
            watch_items.append(cleaned)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "close": result["technical_summary"]["close"],
        "regime": mp.get("regime"),
        "agreement": (mp.get("agreement") or {}).get("label"),
        "dimensions": mp.get("dimensions", {}),
        "supports": result.get("supports", []),
        "resistances": result.get("resistances", []),
        "candle_patterns": result.get("active_candle_patterns", []),
        "chart_patterns": result.get("recent_chart_patterns", []),
        "ob_context": ob_context,
        "model_1_name": opinions.get("model_1_name"),
        "model_1_opinion": opinions.get("model_1_opinion"),
        "model_2_name": opinions.get("model_2_name"),
        "model_2_opinion": opinions.get("model_2_opinion"),
        "consensus": consensus,
        "consensus_bias_fa": BIAS_FA.get(bias, bias),
        "watch_items": watch_items,
    }


# ----------------------------------------------------------------------
# قالب‌بندی خروجی
# ----------------------------------------------------------------------
def _dim_line(name: str, data: dict) -> str:
    label_fa = DIM_LABEL_FA.get(name, name)
    return f"{label_fa}: {data.get('label')} (امتیاز خام: {data.get('score')})"


def _conf_str(consensus: dict) -> str:
    conf = consensus.get("avg_confidence")
    return f"{conf:.2f}" if conf is not None else "—"


def format_markdown(rows: list, timeframe: str, now_local: datetime) -> str:
    lines = []
    lines.append("# 📊 گزارش خودکار بازار کریپتو (پایه‌ی AI)")
    lines.append("")
    lines.append(f"- **زمان محلی:** {now_local.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **UTC:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **تایم‌فریم:** {timeframe}")
    lines.append("")
    lines.append(
        "> ⚠️ این گزارش **توصیه‌ی معاملاتی نیست**؛ توصیف وضعیت فعلی بازار "
        "برای تصمیم‌گیری آگاهانه‌ی انسانی است."
    )
    lines.append("")

    lines.append("## 🧾 خلاصه‌ی کل")
    lines.append("")
    lines.append("| نماد | قیمت | رژیم | جمع‌بندی AI | اطمینان میانگین |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['symbol']} | {_fmt_num(r['close'])} | {r['regime']} | "
            f"{r['consensus_bias_fa']} | {_conf_str(r['consensus'])} |"
        )
    lines.append("")

    for r in rows:
        lines.append(f"## {r['symbol']} ({r['timeframe']})")
        lines.append("")
        lines.append(f"- قیمت آخر: {_fmt_num(r['close'])}")
        lines.append(f"- رژیم بازار: {r['regime']}")
        dims = " | ".join(_dim_line(n, d) for n, d in r["dimensions"].items())
        lines.append(f"- ابعاد: {dims}")
        lines.append(f"- هم‌جهتی ابعاد: {r['agreement']}")
        if r["candle_patterns"]:
            lines.append(f"- الگوهای کندلی (آخرین کندل): {', '.join(r['candle_patterns'])}")
        if r["chart_patterns"]:
            cp = ", ".join(f"{p['pattern']} @ {_fmt_num(p['price'])}" for p in r["chart_patterns"])
            lines.append(f"- الگوهای نموداری اخیر: {cp}")
        if r["supports"]:
            lines.append(f"- حمایت‌ها: {', '.join(_fmt_num(s) for s in r['supports'])}")
        if r["resistances"]:
            lines.append(f"- مقاومت‌ها: {', '.join(_fmt_num(s) for s in r['resistances'])}")
        ob = r["ob_context"]
        if ob:
            lines.append(
                f"- 📗 Order Book (زنده): imbalance_20={ob.get('imbalance_20'):.2f} | "
                f"depth_ratio_20={ob.get('depth_ratio_20'):.2f} | "
                f"spread_pct={ob.get('spread_pct'):.2e}"
            )
        else:
            lines.append("- 📗 Order Book: داده‌ی تازه در دسترس نیست")
        lines.append("")

        cons = r["consensus"]
        lines.append("### 🧩 Ensemble Consensus")
        lines.append("")
        lines.append(cons.get("summary", "—"))
        if r["watch_items"]:
            lines.append("")
            lines.append("👀 نکات قابل مشاهده:")
            for item in r["watch_items"]:
                lines.append(f"- {item}")
        lines.append("")

        if r["model_1_name"]:
            lines.append(f"### 🧠 {r['model_1_name']}")
            lines.append("")
            lines.append(r["model_1_opinion"])
        if r.get("model_2_name") and r["model_2_name"] != "None":
            lines.append("")
            lines.append(f"### 🧠 {r['model_2_name']}")
            lines.append("")
            lines.append(r["model_2_opinion"])
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def format_console(rows: list, timeframe: str) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"📋 گزارش دوره‌ای — {timeframe}")
    lines.append("=" * 70)
    for r in rows:
        lines.append(
            f"{r['symbol']}: {_fmt_num(r['close'])} | رژیم: {r['regime']} | "
            f"AI: {r['consensus_bias_fa']} ({_conf_str(r['consensus'])})"
        )
        if r["watch_items"]:
            lines.append(f"   👀 {r['watch_items'][0]}")
        lines.append("-" * 70)
    return "\n".join(lines)


def format_telegram(rows: list, timeframe: str, now_local: datetime) -> str:
    lines = [
        f"📊 گزارش بازار ({timeframe}) — {now_local.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for r in rows:
        lines.append(f"• {r['symbol']}: {_fmt_num(r['close'])}")
        lines.append(
            f"  رژیم: {r['regime']} | AI: {r['consensus_bias_fa']} "
            f"({_conf_str(r['consensus'])})"
        )
        if r["watch_items"]:
            lines.append(f"  👀 {r['watch_items'][0]}")
        lines.append("")
    lines.append("⚠️ توصیفی، نه توصیه‌ی معاملاتی.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# اجرای اصلی
# ----------------------------------------------------------------------
def run_report(symbols: list, timeframe: str, send_tg: bool) -> list:
    ai = AIBrain()
    rows = []
    for symbol in symbols:
        logger.info(f"تحلیل {symbol} ({timeframe}) ...")
        try:
            row = analyze_one(ai, symbol, timeframe)
            rows.append(row)
            print(f"  ✅ {symbol}: جمع‌بندی = {row['consensus_bias_fa']}")
        except AnalysisError as e:
            logger.error(f"❌ {symbol}: {e}")
        except Exception as e:
            logger.error(f"❌ خطای غیرمنتظره برای {symbol}: {e}")

    if not rows:
        logger.error("هیچ نمادی تحلیل نشد — گزارش ساخته نمی‌شود.")
        return rows

    now_local = datetime.now()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fname = f"daily_report_{now_local.strftime('%Y-%m-%d_%H-%M')}.md"
    fpath = os.path.join(REPORTS_DIR, fname)
    md = format_markdown(rows, timeframe, now_local)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"گزارش ذخیره شد: {fpath}")

    print("\n" + format_console(rows, timeframe))

    if send_tg:
        send_telegram(format_telegram(rows, timeframe, now_local))

    return rows


def parse_args():
    p = argparse.ArgumentParser(description="گزارش دوره‌ای خودکار بازار کریپتو")
    p.add_argument(
        "--symbols",
        default=",".join(Config.DEFAULT_SYMBOLS),
        help="لیست نمادها با کاما (پیش‌فرض: BTCUSDT,ETHUSDT,SOLUSDT)",
    )
    p.add_argument("--timeframe", default="4h", help="تایم‌فریم (پیش‌فرض 4h)")
    p.add_argument(
        "--schedule",
        type=float,
        default=0.0,
        help="اجرای دوره‌ای: فاصله‌ی بین دو گزارش به ساعت (0 = فقط یک بار)",
    )
    p.add_argument(
        "--telegram",
        action="store_true",
        help="ارسال خلاصه به تلگرام (اگر در .env ست شده باشد)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("لیست نمادها خالی است.")
        sys.exit(1)

    while True:
        run_report(symbols, args.timeframe, args.telegram)
        if args.schedule <= 0:
            break
        next_run = datetime.now() + timedelta(hours=args.schedule)
        logger.info(
            f"گزارش بعدی: {next_run.strftime('%Y-%m-%d %H:%M')} "
            f"(هر {args.schedule:g} ساعت)"
        )
        try:
            time.sleep(args.schedule * 3600)
        except KeyboardInterrupt:
            logger.info("توقف به درخواست کاربر.")
            break


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Config.LOG_FILE, encoding="utf-8"),
        ],
    )
    main()
