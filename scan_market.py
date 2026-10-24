"""
scan_market.py
--------------
اسکن چندنمادیِ سیگنال‌های ریز (LGBM live) + دیپ‌دایو هوش مصنوعی روی کاندیدها.

اجرا (از داخل پوشه‌ی پروژه):
    venv\\Scripts\\python.exe scan_market.py
    venv\\Scripts\\python.exe scan_market.py --symbols BTCUSDT ETHUSDT SOLUSDT --signal-tf 15m --deep 2

طرز کار:
  1) سیگنال LGBM (همان live_signal) برای همه‌ی نمادها روی تایم‌فریم ریز
     (پیش‌فرض 15m) محاسبه و جدول رتبه‌بندی چاپ می‌شود — سریع و بدون AI.
  2) «دیپ‌دایو» (تحلیل توصیفی + نظر دو LLM از main.run_analysis_pipeline)
     فقط روی کاندیدهای منتخب اجرا می‌شود؛ چون هر دیپ‌دایو ~۱-۲ دقیقه طول
     می‌کشد (Qwen 14B محلی). پیش‌فرض (--deep بدون مقدار): فقط نمادهایی که
     سیگنال عملی (LONG/SHORT) دارند. --deep N → N کاندید برتر، --no-deep → خاموش.

⚠️ یادآوری: مدل‌های تولیدی روی تایم‌فریم 15m آموزش دیده‌اند و تنها BTC/ETH/SOL
را پوشش می‌دهند (models/prod_15m_meta.json). برای تایم‌فریم‌های ریزتر (5m/1m)
یا نمادهای بیشتر، اول زنجیره‌ی زیر را اجرا کنید:
    download_history.py ← build_dataset.py ← train_production_models.py
"""

import argparse
import logging
import os

# غیرفعال کردن پراکسی برای ارتباط محلی (سازگار با بقیه‌ی پروژه)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from config.settings import Config
from live_signal import build_live_signal

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ScanMarket")


def _scan(symbols: list, signal_tf: str) -> list:
    """سیگنال LGBM برای همه‌ی نمادها؛ خطای هر نماد جدا گزارش می‌شود."""
    rows = []
    for s in symbols:
        try:
            rows.append(build_live_signal(s, interval=signal_tf))
        except Exception as e:
            print(f"⚠️ {s}: سیگنال در دسترس نیست — {str(e)[:160]}")
    return rows


def _sort_key(sig: dict):
    """LONG/SHORT بالاتر از HOLD؛ داخل هر گروه بر اساس حاشیه (قدرت سیگنال)."""
    order = {"LONG": 2, "SHORT": 2, "HOLD": 1}.get(sig.get("side"), 0)
    return (order, sig.get("margin", 0.0))


def _print_scan_table(rows: list, symbols: list, signal_tf: str) -> None:
    print("\n" + "=" * 80)
    print(f"🔎 LIVE SCAN — {signal_tf} | نمادها: {', '.join(symbols)}")
    print("=" * 80)
    print(f"{'SYMBOL':<10}{'CLOSE':>12}{'UP':>7}{'DN':>7}{'ADX':>6}"
          f"  {'SIDE':<6}{'MARGIN':>7}  REASON")
    for sig in sorted(rows, key=_sort_key, reverse=True):
        reason = sig.get("hold_reason") or ""
        print(f"{sig['symbol']:<10}{sig['close']:>12.2f}"
              f"{sig['up_prob']:>7.3f}{sig['dn_prob']:>7.3f}"
              f"{sig['adx']:>6.1f}  {sig['side']:<6}"
              f"{sig.get('margin', 0.0):>7.3f}  {reason}")
    print("=" * 80 + "\n")


def _choose_candidates(rows: list, deep) -> list:
    """انتخاب نمادهای دیپ‌دایو: پیش‌فرض سیگنال‌های عملی؛ --deep N → N برتر."""
    if deep == 0:
        return []
    if deep is None:
        return [r for r in rows if r.get("side") in ("LONG", "SHORT")]
    ranked = sorted(rows, key=_sort_key, reverse=True)
    return ranked[:deep]


def main():
    parser = argparse.ArgumentParser(
        description="اسکن چندنمادی سیگنال ریز + دیپ‌دایو AI")
    parser.add_argument("--symbols", nargs="+", default=Config.DEFAULT_SYMBOLS)
    parser.add_argument("--signal-tf", default="15m",
                        help="تایم‌فریم سیگنال LGBM (باید با مدل آموزش‌دیده هماهنگ باشد)")
    parser.add_argument("--descriptive-tf", default="4h",
                        help="تایم‌فریم گزارش توصیفی/AI در دیپ‌دایو")
    parser.add_argument("--deep", type=int, default=None, nargs="?",
                        const=-1,
                        help="تعداد کاندیدهای دیپ‌دایو؛ بدون مقدار=فقط سیگنال‌های عملی، "
                             "0 = خاموش")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]
    rows = _scan(symbols, args.signal_tf)
    if not rows:
        print("🔴 هیچ سیگنالی ساخته نشد. اتصال/مدل‌ها را بررسی کنید.")
        return
    _print_scan_table(rows, symbols, args.signal_tf)

    deep = args.deep
    if args.deep == -1:  # --deep بدون مقدار
        deep = len([r for r in rows if r.get("side") in ("LONG", "SHORT")]) or 0

    candidates = _choose_candidates(rows, deep)
    if not candidates:
        print("ℹ️ دیپ‌دایو اجرا نشد (سیگنال عملی وجود ندارد). "
              "برای اجرای اجباری: --deep N")
        return

    print(f"🧠 دیپ‌دایو AI برای {len(candidates)} کاندید "
          f"({args.descriptive_tf}) — هر کدام ~۱-۲ دقیقه طول می‌کشد ...\n")

    # import داخل تابع تا اسکنِ سریع با کد سنگین main (AIBrain و...) کند نشود
    from main import run_analysis_pipeline
    for cand in candidates:
        print("\n" + "#" * 80)
        print(f"# DEEP-DIVE: {cand['symbol']} — {args.descriptive_tf}")
        print("#" * 80)
        try:
            run_analysis_pipeline(cand["symbol"], args.descriptive_tf)
        except Exception as e:
            logger.error(f"دیپ‌دایو برای {cand['symbol']} شکست خورد: {e}")


if __name__ == "__main__":
    main()