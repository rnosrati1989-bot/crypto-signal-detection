"""
telegram_bot.py
---------------
مانیتور دائمی سیگنال‌های مدل LGBM تولیدی (پلن معاملاتی) + ارسال به تلگرام.

رفتار:
  1. هر N دقیقه (پیش‌فرض ۱۵) آخرین کندلِ بسته‌شدهٔ هر نماد را با
     build_live_signal چک می‌کند (ورود/SL/TP1..TP3 — همان live_signal.py).
  2. اگر سیگنال جدید LONG/SHORT باشد → همان پیامِ کاملِ live_signal
     (ورود + SL + سه پله TP) فوراً به تلگرام ارسال می‌شود.
  3. اگر سیگنال جدیدی نباشد → هر M دقیقه (پیش‌فرض ۶۰) یک خلاصه‌ی وضعیت
     (احتمال‌ها، ADX، جهت هر نماد) ارسال می‌شود.
  4. وضعیت آخرین ارسال‌ها در telegram_state.json ذخیره می‌شود تا بعد از
     ری‌استارت، همان کندل دوباره ارسال نشود (جلوگیری از اسپم).

راه‌اندازی تلگرام (یک بار):
  1. در تلگرام: @BotFather → /newbot → نام → یوزرنیم (به bot ختم شود)
     → توکن را کپی کن.
  2. فایل .env پروژه: TELEGRAM_BOT_TOKEN=<توکن>   (فقط همین لازم است)
  3. در تلگرام یوزرنیم رباتت را جستجو کن و دکمهٔ Start را بزن.
  4. اجرا:  venv\\Scripts\\python.exe telegram_bot.py
  chat_id لازم نیست — ربات در اولین اجرا به صورت خودکار آن را از
  آخرین گفتگو با تو پیدا و در .env ذخیره می‌کند.

اجرا (از داخل پوشه‌ی پروژه):
    venv\\Scripts\\python.exe telegram_bot.py
    venv\\Scripts\\python.exe telegram_bot.py --check-every 5
    venv\\Scripts\\python.exe telegram_bot.py --check-every 5 --momentum   # هشدار مومنتوم هم بفرست

🚦 رفتار فعلی (بازچینی‌شده — جای فهرست بالا):
  • فقط «سیگنال معاملاتی» LONG/SHORT پیام می‌دهد. HOLD و هشدار مومنتوم پیام نمی‌دهند.
  • خلاصه‌ی دوره‌ای وضعیت (digest) و تحلیل LLM پیش‌فرض خاموش است
    (با --status-every N دقیقه روشن می‌شود).
  • روزی یک‌بار (ساعت --daily-hour، پیش‌فرض ۲۳) «کارنامه‌ی سیگنال‌ها» می‌آید:
    چند درصد حد سود خورده‌اند، چند درصد حد ضرر، چند درصد خروج زمانی، چند تا باز.
  • اگر نمادی سیگنال باز داشته باشد تا بسته‌شدنش تکرار نمی‌شود (ضد‌اسپم telegram_state.json).

⚠️ خروجی مدل در دو اجرای Walk-Forward (~۶۸۵ معامله بعد از کارمزد و
فاندینگ) edge اثبات‌شده نشان نداد. این ابزار «ورودی کمکی برای تصمیم
انسانی» است — نه توصیه‌ی معامله با پول واقعی.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

from config.settings import Config
from live_signal import build_live_signal, format_signal
from signal_logger import (append_signal, append_alert, open_signal,
                           format_feedback_report)

logger = logging.getLogger("TelegramBot")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "telegram_state.json")
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

DISCLAIMER = "⚠️ کمکی برای تصمیم انسانی — نه توصیه‌ی معاملاتی."

_RESOLVED_CHAT_ID = None
_LAST_DETECT_TS = 0.0
_DETECT_COOLDOWN = 120.0  # ثانیه: حداقل فاصله بین تلاش‌های تشخیص خودکار chat_id


# ----------------------------------------------------------------------
# تلگرام
# ----------------------------------------------------------------------
def _auto_detect_chat_id(token: str) -> str:
    """
    پیدا کردن خودکار chat_id از آخرین گفتگو با ربات.
    پیش‌شرط: کاربر یک بار روی ربات Start زده باشد (یا پیامی فرستاده باشد).
    """
    try:
        import requests

        url = f"https://api.telegram.org/bot{token}/getUpdates"
        resp = requests.get(url, timeout=20)
        data = resp.json()
        if not data.get("ok"):
            logger.error(f"getUpdates ناموفق بود: {data}")
            return ""
        for upd in reversed(data.get("result", [])):
            msg = upd.get("message") or upd.get("channel_post") or {}
            chat = msg.get("chat", {})
            cid = chat.get("id")
            if cid is not None:
                return str(cid)
    except Exception as e:
        logger.error(f"خطا در تشخیص خودکار chat_id: {e}")
    return ""


def _persist_chat_id(chat_id: str) -> None:
    """ذخیره‌ی chat_id تشخیص‌داده‌شده در .env تا دفعات بعد پایدار باشد."""
    try:
        from dotenv import set_key
        set_key(ENV_FILE, "TELEGRAM_CHAT_ID", chat_id)
        logger.info(f"chat_id در .env ذخیره شد: {chat_id}")
    except Exception as e:
        logger.warning(f"ذخیره‌ی chat_id در .env ناموفق (مشکلی نیست): {e}")


def get_effective_chat_id(retry_attempts: int = 5, retry_delay: float = 10.0) -> str:
    """
    chat_id مؤثر: اول مقدار .env؛ اگر placeholder بود، از getUpdates
    تشخیص خودکار انجام می‌دهد و در .env ذخیره می‌کند.
    """
    global _RESOLVED_CHAT_ID, _LAST_DETECT_TS
    if _RESOLVED_CHAT_ID:
        return _RESOLVED_CHAT_ID

    cid = str(Config.TELEGRAM_CHAT_ID or "").strip()
    if cid and "your_" not in cid.lower():
        _RESOLVED_CHAT_ID = cid
        return cid

    token = (Config.TELEGRAM_BOT_TOKEN or "").strip()
    if not token or "your_" in token.lower():
        return ""
        

    now = time.time()
    if now - _LAST_DETECT_TS < _DETECT_COOLDOWN:
        return ""

    _LAST_DETECT_TS = now
    logger.info("chat_id در .env ست نشده — تشخیص خودکار از getUpdates ...")
    for attempt in range(1, retry_attempts + 1):
        cid = _auto_detect_chat_id(token)
        if cid:
            _RESOLVED_CHAT_ID = cid
            _persist_chat_id(cid)
            return cid
        if attempt < retry_attempts:
            logger.info("  تلاش %d/%d ناموفق بود — %g ثانیه دیگر دوباره ...",
                        attempt, retry_attempts, retry_delay)
            time.sleep(retry_delay)
    logger.warning("تشخیص خودکار chat_id پس از چند تلاش ناموفق بود — در چک بعدی دوباره تلاش می‌شود.")
    return ""
    
    
        
        
    
        
    


def _telegram_configured() -> bool:
    """فقط اگر توکن واقعی ست شده و chat_id معلوم باشد."""
    token = (Config.TELEGRAM_BOT_TOKEN or "").strip()
    if not token or "your_" in token.lower():
        return False
    return bool(get_effective_chat_id())


def send_telegram(text: str) -> bool:
    if not _telegram_configured():
        logger.warning(
            "تلگرام آماده نیست: توکن در .env ست نیست یا هنوز روی ربات Start نزده‌اید."
        )
        return False
    try:
        import requests

        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id": get_effective_chat_id(),
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=25,
        )
        resp.raise_for_status()
        logger.info("پیام به تلگرام ارسال شد.")
        return True
    except Exception as e:
        logger.error(f"خطا در ارسال به تلگرام: {e}")
        return False


# ----------------------------------------------------------------------
# وضعیت (ضد-اسپم)
# ----------------------------------------------------------------------
def _send_and_log(text: str, kind: str, symbol: str = "", side: str = "",
                  bar_time: str = "", momentum: str = "",
                  detail: str = "") -> bool:
    """
    ارسال پیام + ثبت آن در logs/alerts.csv (kind: CONNECT | TRADE | MOMENTUM |
    DIGEST | DAILY). این لاگ «هر پیامی که کاربر می‌بیند» را ثبت می‌کند تا با
    لاگ معاملات (logs/signals.csv) قابل‌مقایسه باشد.
    """
    ok = send_telegram(text)
    if ok:
        try:
            append_alert(kind, symbol=symbol, side=side, bar_time=bar_time,
                         momentum=momentum, detail=detail)
        except Exception as e:
            logger.debug(f"ثبت پیام در logs/alerts.csv ناموفق بود: {e}")
    return ok


def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("وضعیت ارسال‌ها ذخیره شد.")
    except Exception as e:
        logger.error(f"ذخیره‌ی وضعیت ناموفق: {e}")


# ----------------------------------------------------------------------
# بررسی بازار
# ----------------------------------------------------------------------
def check_once(symbols: list, interval: str, bars: int,
               min_adx, tau_up, tau_dn, tau_scale=1.0,
               mom_atr_mult=1.0, mom_window=None):
    """
    بررسی همه‌ی نمادها.
    خروجی: (summaries, alerts)
      summaries: برای هر نماد یک dict (sig یا error) — برای خلاصه‌ی وضعیت
      alerts:    سیگنال‌های LONG/SHORT + هشدار روند ریز — برای هشدار فوری
    """
    summaries, alerts = [], []
    for symbol in symbols:
        try:
            sig = build_live_signal(symbol, interval=interval, bars=bars,
                                    min_adx=min_adx, tau_up=tau_up,
                                    tau_dn=tau_dn, tau_scale=tau_scale,
                                    mom_atr_mult=mom_atr_mult,
                                    mom_window=mom_window)
            summaries.append({"symbol": symbol, "sig": sig, "error": None})
            if sig["side"] in ("LONG", "SHORT") or sig.get("momentum"):
                alerts.append(sig)
        except Exception as e:
            logger.error(f"❌ {symbol}: {e}")
            summaries.append({"symbol": symbol, "sig": None, "error": str(e)[:120]})
    return summaries, alerts


# ----------------------------------------------------------------------
# قالب‌بندی
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# تحلیل دو مدل هوش مصنوعی (LM Studio: Llama 3 + Mistral 7)
# ----------------------------------------------------------------------
def _ask_llm_model(prompt: str, model: str) -> str:
    """یک پرسش ساده از یک مدل LLM محلی (LM Studio، سازگار با OpenAI API)."""
    base = getattr(Config, "LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    if not model or "your_" in model.lower():
        return ""
    try:
        import requests

        resp = requests.post(
            f"{base}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 350,
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        logger.error(f"LM Studio ({model}) پاسخ نداد: {e}")
        return ""


def build_llm_analysis(summaries: list) -> str:
    """
    تحلیل کوتاه فارسی از دو مدل LM Studio (MODEL_LLAMA + MODEL_MISTRAL)
    بر اساس وضعیت فعلی هر نماد. اگر LM Studio روشن نباشد، پیام میدهد.
    """
    llama = getattr(Config, "MODEL_LLAMA", "") or ""
    mistral = getattr(Config, "MODEL_MISTRAL", "") or ""
    if (not llama or "your_" in llama.lower()) and (not mistral or "your_" in mistral.lower()):
        return ""

    symbols_ok = [s["symbol"] for s in summaries if not s.get("error")]
    rows = []
    for s in summaries:
        if s.get("error"):
            rows.append(f"{s['symbol']}: خطا — {s['error']}")
            continue
        sig = s["sig"]
        if sig["side"] == "HOLD":
            rows.append(
                f"{sig['symbol']} (قیمت {sig['close']:.0f}): HOLD — "
                f"احتمال صعود {sig['up_prob']:.2f} (آستانه {sig['tau_up']:.2f})، "
                f"نزول {sig['dn_prob']:.2f} (آستانه {sig['tau_dn']:.2f})، ADX {sig['adx']:.1f}"
            )
        else:
            rows.append(
                f"{sig['symbol']} (قیمت {sig['close']:.0f}): {sig['side']} قدرت {sig['margin']:.2f} — "
                f"ورود {sig['entry']:.0f}، SL {sig['sl']:.0f}"
            )

    prompt = (
        f"تو یک تحلیلگر بازار کریپتو هستی. وضعیت لحظه‌ای {len(symbols_ok)} نماد فیوچرز "
        f"({', '.join(symbols_ok)}) را ببین "
        "و در حداکثر ۵ جمله (فارسی) خلاصه کن: روند کلی، کدام نماد جالب‌تر است، و چه چیزی را "
        "باید زیر نظر گرفت. فقط توصیف کن؛ توصیه‌ی معاملاتی نده.\n\n"
        + "\n".join(rows)
    )

    parts = []
    if llama and "your_" not in llama.lower():
        txt = _ask_llm_model(prompt, llama)
        name = llama.split("/")[-1]
        parts.append(f"🤖 {name}:\n{txt}" if txt else f"🤖 {name}: (پاسخی نگرفت — LM Studio روشن است؟)")
    if mistral and "your_" not in mistral.lower():
        txt = _ask_llm_model(prompt, mistral)
        name = mistral.split("/")[-1]
        parts.append(f"🤖 {name}:\n{txt}" if txt else f"🤖 {name}: (پاسخی نگرفت — LM Studio روشن است؟)")

    return "\n\n".join(parts)


def format_status_digest(summaries: list, now_local: datetime) -> str:
    lines = [
        f"📡 وضعیت مدل (پلن معاملاتی) — {now_local.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for s in summaries:
        if s["error"]:
            lines.append(f"• {s['symbol']}: ⚠️ خطا — {s['error']}")
            continue
        sig = s["sig"]
        icon = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "HOLD": "🤚 HOLD"}[sig["side"]]
        if sig["side"] == "HOLD":
            reason = sig.get("hold_reason") or "آستانه فعال نشد"
            lines.append(
                f"• {sig['symbol']}: {sig['close']:.2f} | {icon} | ADX {sig['adx']:.1f} ({reason})"
                
            )
        else:
            lines.append(
                f"• {sig['symbol']}: {sig['close']:.2f} | {icon} قدرت {sig['margin']:.2f}"
            )
            lines.append(
                f"    ورود {sig['entry']:.2f} | SL {sig['sl']:.2f} | "
                f"TP {sig['tps'][0]:.2f} / {sig['tps'][1]:.2f} / {sig['tps'][2]:.2f}"
            )
        lines.append(f"    📊 مدل آماری: صعود {sig['up_prob']:.3f} (آستانه {sig['tau_up']:.3f}) | نزول {sig['dn_prob']:.3f} (آستانه {sig['tau_dn']:.3f})")
        mom = sig.get("momentum")
        if mom:
            arrow = "📈" if mom["side"] == "LONG" else "📉"
            lines.append(f"    {arrow} روند {'صعودی' if mom['side'] == 'LONG' else 'نزولی'}: "
                         f"{mom['move_pct']:+.2f}٪ در {mom['window']} کندل ({mom['mom_r']:+.2f}R)")
        liq = sig.get("liquidation")
        if liq and liq.get("n_orders"):
            fa_press = ("شورت‌ها" if liq["imbalance"] > 0.05
                        else "لانگ‌ها" if liq["imbalance"] < -0.05 else "متوازن")
            lines.append(f"    💥 لیکوییدیشن {liq['window_hours']}ساعت: "
                         f"لانگ ${liq['long_notional'] / 1e6:.1f}M | "
                         f"شورت ${liq['short_notional'] / 1e6:.1f}M "
                         f"(فشار: {fa_press})")
        lines.append(f"    کندل سیگنال: {sig['bar_time']}")
    llm_text = build_llm_analysis(summaries)
    if llm_text:
        lines.append("")
        lines.append("🧠 تحلیل دو مدل هوش مصنوعی (LM Studio):")
        lines.append(llm_text)
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# اجرای اصلی
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="مانیتور تلگرام سیگنال‌های مدل LGBM (پلن معاملاتی)"
    )
    parser.add_argument("--symbols",
                        default=",".join(Config.DEFAULT_SYMBOLS),
                        help=f"لیست نمادها با کاما (پیش‌فرض: {','.join(Config.DEFAULT_SYMBOLS)})")
    parser.add_argument("--interval", default="15m",
                        help="باید با مدل تولیدی هماهنگ باشد (پیش‌فرض 15m)")
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--check-every", type=float, default=5.0,
                        help="فاصله‌ی چک بازار به دقیقه (پیش‌فرض ۵)")
    parser.add_argument("--status-every", type=float, default=0.0,
                        help="خلاصه‌ی دوره‌ای وضعیت به دقیقه (۰ = خاموش — پیش‌فرض)")
                        # (سطر help قدیمی حذف شد — پیش‌فرض اکنون خاموش است)
    parser.add_argument("--min-adx", type=float, default=None,
                        help="نادیده‌گرفتن حداقل ADX مدل (اختیاری)")
    parser.add_argument("--tau-up", type=float, default=None,
                        help="نادیده‌گرفتن آستانه‌ی صعود مدل (اختیاری)")
    parser.add_argument("--tau-dn", type=float, default=None,
                        help="نادیده‌گرفتن آستانه‌ی نزول مدل (اختیاری)")
    parser.add_argument("--tau-scale", type=float, default=1.0,
                        help="حساسیت‌زدن آستانه‌ها (۱=بدون تغییر؛ <۱ حساس‌تر)")
    parser.add_argument("--mom-atr-mult", type=float, default=1.0,
                        help="آستانه‌ی هشدار روند ریز بر حسب ATR")
    parser.add_argument("--mom-window", type=int, default=None,
                        help="پنجره‌ی مومنتوم به کندل (پیش‌فرض: horizon مدل)")
    parser.add_argument("--momentum", action="store_true",
                        help="ارسال هشدار مومنتوم در حالت HOLD (پیش‌فرض: خاموش — فقط سیگنال معاملاتی)")
    parser.add_argument("--daily-hour", type=int, default=23,
                        help="ساعت ارسال کارنامه‌ی روزانه‌ی سیگنال‌ها (۰=خاموش؛ پیش‌فرض ۲۳)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("لیست نمادها خالی است.")
        sys.exit(1)

    if args.status_every < 0:
        args.status_every = 0.0

    if not _telegram_configured():
        print("⚠️  تلگرام هنوز آماده نیست:")
        print("    ۱) در تلگرام @BotFather → /newbot → نام → یوزرنیم (به bot ختم شود)")
        print("       توکن را کپی و در فایل .env کنار TELEGRAM_BOT_TOKEN= بچسبان")
        print("    ۲) در تلگرام یوزرنیم رباتت را جستجو کن و دکمهٔ Start را بزن")
        print("    ۳) دوباره همین اجرا را انجام بده")

    state = load_state()
    last_status_sent = 0.0  # اولین سیکل: خلاصه فوراً فرستاده می‌شود (بازخورد سریع)
    last_daily_date = ""  # تاریخ آخرین خلاصه‌ی روزانه‌ی ارسال‌شده

    # --- بررسی اتصال به بایننس پیش از شروع ------------------------------
    # خطای 451 یعنی مسدودسازی جغرافیایی (نه باگ کد). بهتر است همان اول به کاربر
    # گفته شود تا هر ۵ دقیقه هزار خطای «داده دریافت نشد» نبیند.
    try:
        from data.binance_futures_client import BinanceFuturesDataClient
        _ok, _msg = BinanceFuturesDataClient().ping()
        if _ok:
            logger.info(f"✅ {_msg}")
        else:
            logger.error(f"❌ {_msg}")
            _send_and_log("⚠️ اتصال به بایننس برقرار نیست:\n" + _msg +
                          "\n\nتا حل‌شدن این مشکل سیگنال معاملاتی ساخته نمی‌شود "
                          "(تلگرام و جمع‌آور لیکوییدیشن مستقل کار می‌کنند).",
                          "CONNECT", detail="binance-unreachable")
    except Exception as _e:
        logger.warning(f"بررسی اتصال بایننس انجام نشد: {_e}")

    _send_and_log(
        "🤖 ربات سیگنال متصل شد.\n"
        f"نمادها: {', '.join(symbols)} | بازه: {args.interval}\n"
        f"چک هر {args.check_every:g} دقیقه"
        + (f" | خلاصه هر {args.status_every:g} دقیقه" if args.status_every > 0
           else " | خلاصه‌ی وضعیت غیرفعال")
        + "\n" + DISCLAIMER,
        "CONNECT", detail=",".join(symbols),
    )

    try:
        while True:
            start = time.time()
            now_local = datetime.now()
            summaries, alerts = check_once(symbols, args.interval, args.bars,
                                           args.min_adx, args.tau_up, args.tau_dn,
                                           tau_scale=args.tau_scale,
                                           mom_atr_mult=args.mom_atr_mult,
                                           mom_window=args.mom_window)
            # 🚦 فقط سیگنال معاملاتی پیام می‌دهد؛ HOLD و هشدار مومنتوم ساکت می‌مانند
            if not args.momentum:
                alerts = [s for s in alerts if s["side"] in ("LONG", "SHORT")]

            # به‌روزرسانی نتایج سیگنال‌های باز (SL/TP/خروج زمانی) با داده‌ی واقعی
            try:
                review_pending()
            except Exception as rev_e:
                logger.debug(f"review_pending ناموفق بود: {rev_e}")

            # هشدار فوری سیگنال‌های جدید (فقط اگر همین کندل+جهت قبلاً ارسال نشده)
            sent_new = False
            for sig in alerts:
                key = sig["symbol"]
                tradeable = sig["side"] in ("LONG", "SHORT")
                # اگر برای این نماد سیگنالی باز است، تا بسته‌شدنش تکرار نکن
                if tradeable and open_signal(key, args.interval):
                    continue
                prev = state.get(key, {})
                cur = {"bar_time": sig["bar_time"], "side": sig["side"],
                       "mom": (sig.get("momentum") or {}).get("side")}
                if prev != cur:
                    # مدل ورود نداد (HOLD) و فقط هشدار مومنتوم است — صریح بنویس تا
                    # با سیگنال معاملاتی اشتباه نشود (این‌ها در signals.csv ثبت نمی‌شوند).
                    if tradeable:
                        msg = format_signal(sig) + "\n\n" + DISCLAIMER
                    else:
                        msg = ("📈 هشدار مومنتوم (مدل ورود نداد — HOLD)\n"
                               "این پیام «سیگنال ورود» نیست؛ فقط حرکت سریع قیمت است.\n\n"
                               + format_signal(sig) + "\n\n" + DISCLAIMER)
                    mom_side = (sig.get("momentum") or {}).get("side") or ""
                    if _send_and_log(msg,
                                     "TRADE" if tradeable else "MOMENTUM",
                                     symbol=key, side=sig["side"],
                                     bar_time=sig["bar_time"],
                                     momentum=mom_side):
                        state[key] = cur
                        sent_new = True
                        if tradeable:
                            try:
                                append_signal(sig)
                            except Exception as log_e:
                                logger.error(f"ثبت سیگنال در log ناموفق بود: {log_e}")
                # else: همان سیگنال قبلاً ارسال شده — سکوت (ضد اسپم)

            # خلاصه‌ی دوره‌ای وضعیت (اگر همین سیکل هشدار جدید فرستاده شده، رد می‌شود)
            if (args.status_every > 0 and not sent_new
                    and (time.time() - last_status_sent) >= args.status_every * 60):
                digest = format_status_digest(summaries, now_local)
                if _send_and_log(digest, "DIGEST",
                                 detail=f"{len(summaries)} نماد"):
                    last_status_sent = time.time()

            # خلاصه‌ی روزانه‌ی سیگنال‌ها (یک‌بار در روز، در ساعت مشخص)
            if (args.daily_hour > 0 and now_local.hour == args.daily_hour
                    and last_daily_date != now_local.strftime("%Y-%m-%d")):
                # خلاصه‌ی روزانه: هم روز تقویمی، هم بازه‌ی غلتان ۲۴ ساعت (ربات شب‌ها
                # هم سیگنال می‌دهد؛ گزارش تقویمیِ تنها گمراه‌کننده است)
                # کارنامه‌ی روزانه: امروز + آخرین ۲۴ ساعت + کل لاگ (چند درصد TP / SL)
                daily_txt = format_feedback_report(limit=15)
                # (بازچینی: گزارش تک‌پیام شد — بخش «۲۴ ساعته» داخل خودِ کارنامه آمده)
                if _send_and_log(daily_txt, "DAILY"):
                    last_daily_date = now_local.strftime("%Y-%m-%d")

            save_state(state)

            elapsed = time.time() - start
            sleep_s = max(1.0, args.check_every * 60 - elapsed)
            logger.info(f"چک بعدی تا {timedelta(seconds=int(sleep_s))} دیگر...")
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        logger.info("توقف به درخواست کاربر.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
