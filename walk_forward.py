"""
walk_forward.py
-----------------
اعتبارسنجی Walk-Forward: به‌جای یک تقسیم Train/Test تکی (که optimize_backtest.py
انجام میده)، داده‌ی تاریخی رو به چند پنجره‌ی متوالی تقسیم می‌کنیم. توی هر
پنجره: روی بخش Train بهترین پارامتر پیدا میشه، بعد بلافاصله روی بخش Test
(که زمانی درست بعد از Train میاد) امتحان میشه. بعد پنجره به جلو "قدم" برمی‌داره.

چرا بهتر از یه تقسیم تکیه؟
یک تقسیم Train/Test فقط یک رژیم بازار رو می‌سنجه (مثلاً ۴۲ روز خاص). اگه توی
اون بازه بازار روند‌دار یا رنج بوده، نتیجه به همون رژیم خاص وابسته‌ست. با
اجرای همین آزمایش روی ۵-۶ پنجره‌ی متوالی مختلف (ماه‌های مختلف)، می‌فهمیم که
آیا سیستم به‌طور *پایدار* توی رژیم‌های مختلف کار می‌کنه، یا فقط یه بار شانسی
تو یه بازه‌ی خاص خوب بوده.

معیار قضاوت نهایی: نسبت پنجره‌هایی که Test مثبت شده (Positive Window Ratio)
و میانگین وزن‌دارِ R روی کل پنجره‌های Test.
"""

import argparse
import logging
import pandas as pd

from backtest import prepare_dataframe, run_backtest
from optimize_backtest import grid_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WalkForward")


def run_walk_forward(
    symbol: str,
    timeframe: str,
    train_size: int = 700,
    test_size: int = 300,
    step: int = 300,
    n_windows: int = 6,
    min_trades: int = 15,
):
    total_needed = train_size + test_size + (n_windows - 1) * step
    logger.info(f"دریافت {total_needed} کندل تاریخی برای {symbol}/{timeframe} (برای {n_windows} پنجره)...")

    try:
        df = prepare_dataframe(symbol, timeframe, limit=total_needed)
    except RuntimeError as e:
        print(f"\n❌ عدم امکان دریافت داده: {e}")
        return

    if len(df) < train_size + test_size:
        print(f"\n⚠️ داده‌ی کافی برای حتی یک پنجره وجود نداره. دریافت شد: {len(df)}, لازم: {train_size + test_size}")
        return

    window_results = []
    start = 0
    window_num = 0

    while start + train_size + test_size <= len(df):
        window_num += 1
        train_df = df.iloc[start: start + train_size]
        test_df = df.iloc[start + train_size: start + train_size + test_size]

        window_start_date = train_df.index[0]
        window_test_start_date = test_df.index[0]
        window_test_end_date = test_df.index[-1]

        logger.info(
            f"پنجره {window_num}: Train از {window_start_date.date()} | "
            f"Test از {window_test_start_date.date()} تا {window_test_end_date.date()}"
        )

        train_warmup = min(210, train_size // 3)
        grid_results = grid_search(train_df, warmup=train_warmup, min_trades=min_trades)

        if grid_results.empty:
            logger.warning(f"پنجره {window_num}: هیچ ترکیب پارامتری با حداقل معامله‌ی لازم پیدا نشد. رد شد.")
            start += step
            continue

        best = grid_results.iloc[0]  # بهترین ترکیب بر اساس avg_r روی Train همین پنجره

        test_warmup = min(50, test_size // 3)
        try:
            test_stats = run_backtest(
                test_df,
                atr_sl_mult=best["atr_mult"],
                risk_reward=best["rr"],
                warmup=test_warmup,
                buy_threshold=best["buy_threshold"],
                sell_threshold=best["sell_threshold"],
                regime_adx_min=best["regime_adx"],
            )
        except RuntimeError:
            start += step
            continue

        window_results.append({
            "window": window_num,
            "test_period": f"{window_test_start_date.date()} → {window_test_end_date.date()}",
            "best_atr_mult": best["atr_mult"],
            "best_rr": best["rr"],
            "best_buy_th": best["buy_threshold"],
            "best_sell_th": best["sell_threshold"],
            "best_regime_adx": best["regime_adx"],
            "train_avg_r": round(best["avg_r"], 3),
            "test_avg_r": test_stats.get("avg_r_multiple"),
            "test_trades": test_stats.get("total_trades", 0),
            "test_win_rate": test_stats.get("win_rate_pct"),
        })

        start += step

    print_walk_forward_report(symbol, timeframe, window_results)
    return window_results


def print_walk_forward_report(symbol: str, timeframe: str, window_results: list):
    print("\n" + "=" * 110)
    print(f"🚶 WALK-FORWARD VALIDATION REPORT — {symbol} ({timeframe})")
    print("=" * 110)

    if not window_results:
        print("هیچ پنجره‌ای با داده‌ی کافی/معاملات کافی برای گزارش وجود نداشت.")
        print("=" * 110 + "\n")
        return

    df_results = pd.DataFrame(window_results)
    print(df_results.to_string(index=False))
    print("-" * 110)

    valid = df_results.dropna(subset=["test_avg_r"])
    total_windows = len(valid)
    positive_windows = (valid["test_avg_r"] > 0).sum()
    total_test_trades = valid["test_trades"].sum()

    if total_test_trades > 0:
        weighted_test_r = (valid["test_avg_r"] * valid["test_trades"]).sum() / total_test_trades
    else:
        weighted_test_r = 0

    positive_ratio = (positive_windows / total_windows * 100) if total_windows > 0 else 0

    print(f"• تعداد کل پنجره‌های معتبر:                 {total_windows}")
    print(f"• پنجره‌های با Test مثبت:                    {positive_windows} از {total_windows} ({round(positive_ratio,1)}%)")
    print(f"• میانگین وزن‌دارِ R روی همه‌ی پنجره‌های Test:  {round(weighted_test_r, 3)}")
    print("-" * 110)

    if positive_ratio >= 70 and weighted_test_r > 0.05:
        verdict = (
            "سیستم توی اکثر پنجره‌های زمانی مختلف edge مثبت نشون داده. این قوی‌ترین "
            "شواهدیه که تا الان جمع کردیم — هنوز به‌معنی تضمین سود نیست، ولی ارزش ادامه دادن و تست بیشتر (حتی paper trading) رو داره."
        )
    elif positive_ratio >= 40:
        verdict = (
            "نتیجه ترکیبیه — بعضی رژیم‌های بازار مثبت، بعضی منفی. این معمولاً یعنی سیستم "
            "فقط توی شرایط خاصی از بازار (مثلاً روند قوی) کار می‌کنه؛ باید یه فیلتر رژیم بازار اضافه بشه."
        )
    else:
        verdict = (
            "سیستم توی اکثر پنجره‌های زمانی مختلف edge نداشته — این یافته با نتیجه‌ی optimize_backtest.py "
            "هم‌خونیه. یعنی مشکل از خودِ انتخاب اندیکاتورها/فرمول اسکورینگه، نه از شانس یه بازه‌ی خاص."
        )

    print(f"🔎 جمع‌بندی: {verdict}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="اعتبارسنجی Walk-Forward روی چند پنجره‌ی زمانی متوالی")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--train-size", type=int, default=700)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--step", type=int, default=300)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--min-trades", type=int, default=15)
    args = parser.parse_args()

    run_walk_forward(
        args.symbol, args.timeframe,
        train_size=args.train_size, test_size=args.test_size,
        step=args.step, n_windows=args.n_windows, min_trades=args.min_trades,
    )
