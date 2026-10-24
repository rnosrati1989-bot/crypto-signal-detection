"""
sweep_experiments.py
--------------------
آزمایش خودکار ترکیب‌های مختلف پیکربندی استراتژی روی Walk-Forward + شبیه‌سازی
واقعی (کارمزد + فاندینگ). هدف: پیدا کردن پیکربندی که بعد از هزینه‌ها edge دارد.

ابعاد آزمایش:
    - افق نگه‌داری (horizon): 4، 16، 96 کندل
    - آستانه‌ی احتمال (threshold): 0.25، 0.30، 0.35
    - فیلتر رژیم (min_adx): 0 (خاموش)، 25 (روند قوی)

نحوه‌ی اجرا:
    python sweep_experiments.py
"""

import argparse
import logging
import itertools

import numpy as np
import pandas as pd

from backtest_model import load_combined_dataset, simulate_all, compute_stats
from features import get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Sweep")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_INTERVAL = "15m"
DEFAULT_FEE = 0.0005


def main():
    parser = argparse.ArgumentParser(description="آزمایش ترکیب‌های پیکربندی")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--horizons", nargs="+", type=int, default=[4, 16, 96])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.30, 0.35])
    parser.add_argument("--min-adxs", nargs="+", type=float, default=[0, 25])
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("\n❌ lightgbm نصب نیست: pip install lightgbm scikit-learn\n")
        return

    df = load_combined_dataset([s.upper() for s in args.symbols], args.interval)

    feature_cols = get_feature_columns(df)
    if "symbol" not in feature_cols:
        feature_cols = feature_cols + ["symbol"]

    X = df[feature_cols].copy()
    X["symbol"] = X["symbol"].astype("category")

    n = len(X)
    n_windows = 0
    start = 0
    while start + args.train_size + args.test_size <= n:
        n_windows += 1
        start += args.step

    print(f"\n📡 آزمایش {len(args.horizons)}×{len(args.thresholds)}×{len(args.min_adxs)} "
          f"= {len(args.horizons)*len(args.thresholds)*len(args.min_adxs)} ترکیب "
          f"| {n_windows} پنجره | {n} ردیف\n")

    rows = []
    for horizon, threshold, min_adx in itertools.product(
            args.horizons, args.thresholds, args.min_adxs):

        target_col = f"target_up_{horizon}"
        if target_col not in df.columns:
            continue
        y = df[target_col].astype(int)

        total_trades = 0
        exp_sum = 0.0
        pos_windows = 0
        windows_used = 0

        start = 0
        while start + args.train_size + args.test_size <= n:
            tr = slice(start, start + args.train_size)
            te = slice(start + args.train_size, start + args.train_size + args.test_size)

            X_train, X_test = X.iloc[tr], X.iloc[te]
            y_train = y.iloc[tr]
            test_df = df.iloc[te].copy()

            model = lgb.LGBMClassifier(
                n_estimators=args.n_estimators, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
            )
            model.fit(X_train, y_train,
                      eval_X=X_test, eval_y=y.iloc[te],
                      callbacks=[lgb.early_stopping(20, verbose=False)])

            probs = model.predict_proba(X_test)[:, 1]
            trades = simulate_all(test_df, probs, horizon, threshold,
                                  DEFAULT_FEE, min_adx, use_funding=True)
            st = compute_stats(trades)

            windows_used += 1
            total_trades += st["n"]
            exp_sum += st["expectancy_pct"] * st["n"]
            if st["n"] > 0 and st["expectancy_pct"] > 0:
                pos_windows += 1

            start += args.step

        if total_trades == 0:
            continue

        weighted_exp = exp_sum / total_trades
        rows.append({
            "horizon": horizon,
            "threshold": threshold,
            "min_adx": min_adx,
            "windows": windows_used,
            "pos_windows": pos_windows,
            "trades": total_trades,
            "exp_pct": weighted_exp,
        })

    if not rows:
        print("هیچ ترکیبی معامله‌ای تولید نکرد.")
        return

    res = pd.DataFrame(rows).sort_values("exp_pct", ascending=False)

    print("=" * 90)
    print("🏆 نتایج آزمایش ترکیب‌ها — مرتب شده بر اساس Expectancy (بعد از هزینه‌ها)")
    print("=" * 90)
    print(f"    {'افق':<5} {'آستانه':<8} {'ADX≥':<6} {'پنجره‌ها':<8} {'سودده':<7} "
          f"{'معامله':<7} {'Expectancy':<12} وضعیت")
    print("    " + "-" * 82)
    for _, r in res.iterrows():
        status = "✅" if r["exp_pct"] > 0 else "❌"
        print(f"    {r['horizon']:<5} {r['threshold']:<8.2f} {r['min_adx']:<6.0f} "
              f"{r['windows']:<8} {r['pos_windows']:<7} {r['trades']:<7} "
              f"{r['exp_pct']:>+9.4f}٪  {status}")
    print("    " + "-" * 82)

    best = res.iloc[0]
    print(f"\n🏆 بهترین پیکربندی: افق {best['horizon']} | آستانه {best['threshold']} "
          f"| ADX≥{best['min_adx']} → Expectancy {best['exp_pct']:+.4f}٪ "
          f"({best['trades']} معامله)")
    if best["exp_pct"] > 0:
        print("✅ این پیکربندی بعد از هزینه‌ها سودده بوده — کاندید ادامه‌ی مسیر.")
    else:
        print("🔴 هیچ ترکیبی بعد از هزینه‌ها سودده نبود — باید به سراغ ویژگی/هدف جدید برویم.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
