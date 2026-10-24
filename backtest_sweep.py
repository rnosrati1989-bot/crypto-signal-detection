"""
backtest_sweep.py
-------------------
تست جامع‌تر از backtest.py: منطق اسکورینگ فعلی رو روی چند نماد، چند
تایم‌فریم و چند تنظیم ریسک/ریوارد اجرا می‌کنه و یک جدول مقایسه‌ای چاپ می‌کنه.

هدف: قبل از اضافه‌کردن الگو/استراتژی جدید، بفهمیم آیا SignalScorer فعلی
اصلاً *جایی* (روی هر نماد/تایم‌فریمی) یک edge آماری معنادار داره یا نه.
اگه همه‌جا حول صفر یا منفیه، مشکل از پایه (وزن‌دهی اسکورینگ) هست، نه از
کمبود تعداد الگوها.

اجرا:
    python backtest_sweep.py
    python backtest_sweep.py --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --rrs 1.5 2.0 3.0
"""

import argparse
import logging
import pandas as pd

from backtest import prepare_dataframe, run_backtest

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BacktestSweep")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]
DEFAULT_RRS = [1.5, 2.0, 3.0]


def breakeven_winrate(rr: float) -> float:
    """درصد بردِ لازم برای سربه‌سر شدن با نسبت ریسک/ریوارد مشخص."""
    return 100.0 / (1 + rr)


def run_sweep(symbols, timeframes, rrs, atr_mult=1.5, limit=1000):
    rows = []

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                df = prepare_dataframe(symbol, timeframe, limit=limit)
            except Exception as e:
                rows.append({
                    "symbol": symbol, "timeframe": timeframe, "rr": "-",
                    "trades": 0, "win_rate": None, "avg_r": None,
                    "profit_factor": None, "max_dd_pct": None,
                    "note": f"خطا در دریافت داده: {str(e)[:60]}"
                })
                continue

            for rr in rrs:
                try:
                    stats = run_backtest(df, atr_sl_mult=atr_mult, risk_reward=rr)
                except Exception as e:
                    rows.append({
                        "symbol": symbol, "timeframe": timeframe, "rr": rr,
                        "trades": 0, "win_rate": None, "avg_r": None,
                        "profit_factor": None, "max_dd_pct": None,
                        "note": f"خطا: {str(e)[:60]}"
                    })
                    continue

                if stats.get("total_trades", 0) == 0:
                    rows.append({
                        "symbol": symbol, "timeframe": timeframe, "rr": rr,
                        "trades": 0, "win_rate": None, "avg_r": None,
                        "profit_factor": None, "max_dd_pct": None,
                        "note": "بدون معامله"
                    })
                    continue

                be = breakeven_winrate(rr)
                edge = stats["win_rate_pct"] - be  # فاصله از نقطه‌ی سربه‌سر (مثبت = edge واقعی)

                rows.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "rr": rr,
                    "trades": stats["total_trades"],
                    "win_rate": stats["win_rate_pct"],
                    "breakeven_wr": round(be, 2),
                    "edge_vs_breakeven": round(edge, 2),
                    "avg_r": stats["avg_r_multiple"],
                    "profit_factor": stats["profit_factor"],
                    "max_dd_pct": stats["max_drawdown_pct_at_1pct_risk"],
                    "note": ""
                })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("📊 SWEEP REPORT — مقایسه‌ی SignalScorer روی نمادها/تایم‌فریم‌ها/RRهای مختلف")
    print("=" * 100)
    if df.empty:
        print("هیچ نتیجه‌ای تولید نشد.")
        return

    display_cols = ["symbol", "timeframe", "rr", "trades", "win_rate",
                     "breakeven_wr", "edge_vs_breakeven", "avg_r", "profit_factor", "max_dd_pct", "note"]
    print(df[display_cols].to_string(index=False))

    print("-" * 100)
    valid = df[df["trades"] > 0].copy()
    if not valid.empty:
        positive_edge = valid[valid["edge_vs_breakeven"] > 0]
        print(f"• تعداد سناریوهایی با edge مثبت (بالاتر از سربه‌سر): {len(positive_edge)} از {len(valid)}")
        if not positive_edge.empty:
            print("  سناریوهای با edge مثبت:")
            print(positive_edge[["symbol", "timeframe", "rr", "win_rate", "edge_vs_breakeven", "trades"]].to_string(index=False))
        else:
            print("  ⚠️ هیچ ترکیبی edge مثبت نداشت — یعنی مشکل از منطق اسکورینگ (وزن‌دهی)ه، نه تعداد الگوها.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="تست جامع SignalScorer روی چند نماد/تایم‌فریم/RR")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    parser.add_argument("--rrs", nargs="+", type=float, default=DEFAULT_RRS)
    parser.add_argument("--atr-mult", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    result_df = run_sweep(args.symbols, args.timeframes, args.rrs, atr_mult=args.atr_mult, limit=args.limit)
    print_summary(result_df)

    # ذخیره خروجی کامل برای بررسی بیشتر
    result_df.to_csv("logs/sweep_results.csv", index=False)
    print("نتایج کامل در logs/sweep_results.csv ذخیره شد.")
