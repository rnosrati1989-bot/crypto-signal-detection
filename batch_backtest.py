"""
batch_backtest.py
-------------------
اجرای backtest.py روی چند نماد و چند تایم‌فریم به‌صورت خودکار، برای این‌که
بفهمیم آیا نتیجه‌ی یک بک‌تست تکی (مثلاً فقط BTCUSDT/1h) یک الگوی پایدار
هست یا صرفاً نویز آماری روی یک نمونه‌ی کوچیک.

خروجی: یک جدول مقایسه‌ای با ستون‌های نماد، تایم‌فریم، تعداد معاملات،
Win Rate، Expectancy (R) و Profit Factor، به‌همراه میانگین کلی.
"""

import argparse
import logging
import time
import pandas as pd

from backtest import prepare_dataframe, run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BatchBacktest")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]


def run_batch(symbols, timeframes, limit=1000, atr_mult=1.5, rr=2.0, warmup=210):
    results = []

    for symbol in symbols:
        for tf in timeframes:
            logger.info(f"Backtesting {symbol} / {tf} ...")
            try:
                df = prepare_dataframe(symbol, tf, limit=limit)
                stats = run_backtest(df, atr_sl_mult=atr_mult, risk_reward=rr, warmup=warmup)
            except Exception as e:
                logger.warning(f"Skipped {symbol}/{tf}: {e}")
                continue

            if stats.get("total_trades", 0) == 0:
                results.append({
                    "symbol": symbol, "timeframe": tf, "trades": 0,
                    "win_rate_pct": None, "avg_r": None,
                    "profit_factor": None, "max_dd_pct": None,
                })
            else:
                results.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "trades": stats["total_trades"],
                    "win_rate_pct": stats["win_rate_pct"],
                    "avg_r": stats["avg_r_multiple"],
                    "profit_factor": stats["profit_factor"],
                    "max_dd_pct": stats["max_drawdown_pct_at_1pct_risk"],
                })

            time.sleep(0.3)  # کمی وقفه برای رعایت rate-limit بایننس

    return pd.DataFrame(results)


def print_summary(df_results: pd.DataFrame, required_win_rate_pct: float):
    print("\n" + "=" * 100)
    print("📊 BATCH BACKTEST SUMMARY — همه‌ی نماد/تایم‌فریم‌ها")
    print("=" * 100)
    print(df_results.to_string(index=False))
    print("-" * 100)

    valid = df_results.dropna(subset=["avg_r"])
    if valid.empty:
        print("هیچ نتیجه‌ی معتبری برای جمع‌بندی وجود نداره (احتمالاً هیچ سیگنالی صادر نشده).")
        print("=" * 100 + "\n")
        return

    total_trades = valid["trades"].sum()
    weighted_avg_r = (valid["avg_r"] * valid["trades"]).sum() / total_trades if total_trades > 0 else 0
    overall_win_rate = (valid["win_rate_pct"] * valid["trades"]).sum() / total_trades if total_trades > 0 else 0
    positive_combos = (valid["avg_r"] > 0).sum()
    total_combos = len(valid)

    print(f"• مجموع معاملات روی همه‌ی ترکیب‌ها:        {total_trades}")
    print(f"• میانگین وزن‌دارِ R (کل):                  {round(weighted_avg_r, 3)}")
    print(f"• Win Rate وزن‌دارِ کل:                     {round(overall_win_rate, 2)}%")
    print(f"• Win Rate سر به سر لازم (با این RR):       {round(required_win_rate_pct, 2)}%")
    print(f"• تعداد ترکیب‌های نماد/تایم‌فریم با R مثبت:  {positive_combos} از {total_combos}")
    print("-" * 100)

    if weighted_avg_r > 0.05:
        verdict = "روی این نمونه، سیستم به‌طور نسبتاً پایدار برتری آماری داره. ارزش تنظیم و ادامه رو داره."
    elif weighted_avg_r > -0.05:
        verdict = "سیستم تقریباً سر به سره (نه سود پایدار، نه ضرر واضح) — یعنی منطق فعلی هنوز edge واقعی نداره."
    else:
        verdict = "سیستم به‌طور نسبتاً پایدار منفیه. قبل از اضافه‌کردن الگوی بیشتر، باید خودِ وزن‌دهی/آستانه‌های اسکورینگ بازبینی بشه."

    print(f"🔎 جمع‌بندی: {verdict}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch backtest روی چند نماد/تایم‌فریم")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--atr-mult", type=float, default=1.5)
    parser.add_argument("--rr", type=float, default=2.0)
    args = parser.parse_args()

    required_win_rate = 100 / (1 + args.rr)

    df_results = run_batch(
        args.symbols, args.timeframes,
        limit=args.limit, atr_mult=args.atr_mult, rr=args.rr
    )
    print_summary(df_results, required_win_rate)
