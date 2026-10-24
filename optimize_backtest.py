"""
optimize_backtest.py
----------------------
پیدا کردن بهترین پارامترهای اسکورینگ (آستانه‌ی BUY/SELL، ضریب ATR، RR) با
یک روش درست: تقسیم داده به دو بخش زمانی جدا (Train / Test).

چرا این کار لازمه؟
اگه فقط روی کل داده پارامتر رو بهینه کنیم و همون‌جا هم نتیجه رو گزارش بدیم،
عملاً داریم پارامترها رو "روی امتحان" تنظیم می‌کنیم (overfitting) — نتیجه
همیشه خوب به‌نظر میاد ولی هیچ ربطی به آینده نداره.

روش درست:
1. اولین ٪۷۰ داده‌ی تاریخی = Train: روی این بخش، تمام ترکیب‌های پارامتر رو
   امتحان می‌کنیم و بهترین‌ها رو بر اساس Expectancy (میانگین R) انتخاب می‌کنیم
   (با حداقل تعداد معامله، تا یه پارامتر با ۲ معامله‌ی شانسی انتخاب نشه).
2. آخرین ٪۳۰ داده = Test (Out-of-Sample): همون پارامتر برگزیده رو، بدون هیچ
   تغییری، روی این بخش که مدل موقع انتخاب پارامتر ندیده، امتحان می‌کنیم.
3. اگه نتیجه‌ی Test هم مثبت/نزدیک به Train بود => احتمال بیشتری داره که edge
   واقعی باشه. اگه Test بد بود (حتی وقتی Train عالی بود) => یعنی صرفاً
   overfitting بوده و اون پارامتر ارزش استفاده‌ی واقعی نداره.
"""

import argparse
import logging
import itertools
import pandas as pd

from backtest import prepare_dataframe, run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Optimize")


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.7):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def grid_search(
    train_df: pd.DataFrame,
    atr_mults=(1.0, 1.5, 2.0),
    rrs=(1.0, 1.5, 2.0, 2.5),
    buy_thresholds=(60, 70, 80),
    sell_thresholds=(20, 30, 40),
    regime_adx_filters=(0, 20, 25),
    warmup: int = 210,
    min_trades: int = 15,
):
    """
    جست‌وجوی شبکه‌ای روی Train؛ فقط ترکیب‌هایی با حداقل min_trades معامله در نظر گرفته میشن.
    regime_adx_filters: ۰ یعنی بدون فیلتر رژیم بازار؛ ۲۰/۲۵ یعنی فقط وقتی ADX بالای اون
    مقدار باشه (بازار روند‌دار تشخیص داده بشه) اجازه‌ی ورود صادر میشه — برای تست فرضیه‌ی
    «سیستم فقط توی بازار رونددار کار می‌کنه».
    """
    results = []

    for atr_mult, rr, buy_th, sell_th, regime_adx in itertools.product(
        atr_mults, rrs, buy_thresholds, sell_thresholds, regime_adx_filters
    ):
        try:
            stats = run_backtest(
                train_df,
                atr_sl_mult=atr_mult,
                risk_reward=rr,
                warmup=warmup,
                buy_threshold=buy_th,
                sell_threshold=sell_th,
                regime_adx_min=regime_adx,
            )
        except RuntimeError:
            continue

        if stats.get("total_trades", 0) < min_trades:
            continue

        results.append({
            "atr_mult": atr_mult,
            "rr": rr,
            "buy_threshold": buy_th,
            "sell_threshold": sell_th,
            "regime_adx": regime_adx,
            "trades": stats["total_trades"],
            "avg_r": stats["avg_r_multiple"],
            "win_rate_pct": stats["win_rate_pct"],
            "profit_factor": stats["profit_factor"],
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("avg_r", ascending=False).reset_index(drop=True)


def run_optimization(symbol: str, timeframe: str, limit: int, min_trades: int, top_n: int):
    logger.info(f"Fetching data for {symbol}/{timeframe}...")
    try:
        df = prepare_dataframe(symbol, timeframe, limit=limit)
    except RuntimeError as e:
        print(f"\n❌ عدم امکان دریافت داده: {e}")
        print("   (اگر از VPN/پروکسی استفاده می‌کنید، از پایدار بودن اتصال مطمئن شوید و دوباره امتحان کنید)")
        return

    train_df, test_df = split_train_test(df, train_ratio=0.7)
    logger.info(f"Train: {len(train_df)} کندل | Test (Out-of-Sample): {len(test_df)} کندل")

    logger.info("در حال جست‌وجوی شبکه‌ای پارامترها روی داده‌ی Train...")
    grid_results = grid_search(train_df, min_trades=min_trades)

    if grid_results.empty:
        print("\n⚠️ هیچ ترکیب پارامتری با حداقل تعداد معامله‌ی لازم پیدا نشد. یا داده کمه، یا min_trades رو کم کن.")
        return

    print("\n" + "=" * 100)
    print(f"🏆 بهترین {top_n} ترکیب پارامتر روی داده‌ی TRAIN (in-sample)")
    print("=" * 100)
    print(grid_results.head(top_n).to_string(index=False))

    # حالا بهترین ترکیب رو روی TEST (که مدل ندیده) امتحان می‌کنیم
    print("\n" + "=" * 100)
    print("🔬 اعتبارسنجی بهترین ترکیب‌ها روی TEST (Out-of-Sample — داده‌ای که مدل موقع انتخاب پارامتر ندیده)")
    print("=" * 100)

    validation_rows = []
    for _, row in grid_results.head(top_n).iterrows():
        try:
            test_stats = run_backtest(
                test_df,
                atr_sl_mult=row["atr_mult"],
                risk_reward=row["rr"],
                warmup=min(50, len(test_df) // 3),  # warmup کوچک‌تر چون test_df کوچیک‌تره
                buy_threshold=row["buy_threshold"],
                sell_threshold=row["sell_threshold"],
                regime_adx_min=row["regime_adx"],
            )
        except RuntimeError as e:
            validation_rows.append({**row.to_dict(), "test_trades": 0, "test_avg_r": None, "test_win_rate": None})
            continue

        validation_rows.append({
            "atr_mult": row["atr_mult"],
            "rr": row["rr"],
            "buy_threshold": row["buy_threshold"],
            "sell_threshold": row["sell_threshold"],
            "regime_adx": row["regime_adx"],
            "train_avg_r": row["avg_r"],
            "train_trades": row["trades"],
            "test_avg_r": test_stats.get("avg_r_multiple"),
            "test_trades": test_stats.get("total_trades", 0),
            "test_win_rate": test_stats.get("win_rate_pct"),
        })

    validation_df = pd.DataFrame(validation_rows)
    print(validation_df.to_string(index=False))
    print("-" * 100)

    consistent = validation_df[
        (validation_df["test_avg_r"].notna()) & (validation_df["test_avg_r"] > 0) & (validation_df["train_avg_r"] > 0)
    ]

    if not consistent.empty:
        print(f"✅ {len(consistent)} ترکیب پارامتر هم روی Train و هم روی Test مثبت بودن — این‌ها کاندیدای واقعی edge هستن.")
        print("   (هنوز هم قبل از استفاده‌ی واقعی، حتماً روی داده‌ی جدیدتر و نمادهای دیگه هم چک کن.)")
    else:
        print("❌ هیچ ترکیبی که روی Train خوب بود، روی Test (داده‌ی ندیده) مثبت نموند.")
        print("   این یعنی نتایج خوبِ Train صرفاً overfitting بودن، نه edge واقعی.")
        print("   نتیجه: مشکل از خودِ فرمول اسکورینگ/اندیکاتورهای انتخابیه، نه فقط تنظیم پارامتر.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="بهینه‌سازی پارامترهای اسکورینگ با اعتبارسنجی Train/Test")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--min-trades", type=int, default=15, help="حداقل تعداد معامله برای معتبر بودن یک ترکیب پارامتر")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    run_optimization(args.symbol, args.timeframe, args.limit, args.min_trades, args.top_n)
