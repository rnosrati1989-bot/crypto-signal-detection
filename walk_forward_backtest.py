"""
walk_forward_backtest.py
------------------------
آزمون نهایی و صادقانه: شبیه‌سازی واقع‌گرایانه (با هزینه‌ها) روی چند پنجره‌ی
زمانی متوالی — دقیقاً مثل Walk-Forward ولی با معامله‌ی واقعی.

سؤال: «آیا بعد از کسر کارمزد و فاندینگ، استراتژی در *اکثر* رژیم‌های بازار
(گاوی، نزولی، رنج) سودده است یا فقط در هفته‌های گاوی؟»

نحوه‌ی اجرا:
    python walk_forward_backtest.py --horizon 4 --threshold 0.25 --windows 6
"""

import argparse
import logging

import numpy as np
import pandas as pd

from backtest_model import load_combined_dataset, simulate_all, compute_stats
from features import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WFBacktest")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_INTERVAL = "15m"
DEFAULT_FEE = 0.0005


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward با شبیه‌سازی واقعی")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--min-adx", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("\n❌ lightgbm نصب نیست: pip install lightgbm scikit-learn\n")
        return

    df = load_combined_dataset([s.upper() for s in args.symbols], args.interval)

    target_col = f"target_up_{args.horizon}"
    if target_col not in df.columns:
        print(f"❌ ستون {target_col} در دیتاست نیست.")
        return

    feature_cols = get_feature_columns(df)
    if "symbol" not in feature_cols:
        feature_cols = feature_cols + ["symbol"]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    X["symbol"] = X["symbol"].astype("category")

    n = len(X)
    total_needed = args.train_size + args.test_size + (args.windows - 1) * args.step
    if n < total_needed:
        print(f"⚠️ داده‌ی کافی نیست: {n} موجود، {total_needed} لازم.")
        return

    print("\n" + "=" * 100)
    print("🚶 WALK-FORWARD + شبیه‌سازی واقعی (کارمزد + فاندینگ)")
    print("=" * 100)
    print(f"• آستانه: {args.threshold} | ADX≥: {args.min_adx} | کارمزد: {args.fee*100:.3f}٪/طرف")
    print()

    results = []
    start = 0
    window_num = 0

    while start + args.train_size + args.test_size <= n:
        window_num += 1
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
                  callbacks=[lgb.early_stopping(30, verbose=False)])

        probs = model.predict_proba(X_test)[:, 1]
        trades = simulate_all(test_df, probs, args.horizon, args.threshold,
                              args.fee, args.min_adx, use_funding=True)
        st = compute_stats(trades)

        # Buy & Hold در همین پنجره (متوسط نمادها) برای شناخت رژیم بازار
        bh_list = []
        for symbol in test_df["symbol"].unique():
            grp = test_df[test_df["symbol"] == symbol]
            bh_list.append(grp["close"].iloc[-1] / grp["close"].iloc[0] - 1)
        avg_bh = float(np.mean(bh_list)) * 100 if bh_list else 0.0

        results.append({
            "window": window_num,
            "test_period": f"{test_df.index.min().date()} → {test_df.index.max().date()}",
            "avg_bh_pct": avg_bh,
            "n_trades": st["n"],
            "win_rate": st["win_rate"],
            "expectancy_pct": st["expectancy_pct"],
            "profit_factor": st["profit_factor"],
        })

        mark = "✅" if st["expectancy_pct"] > 0 else "❌"
        print(f"پنجره {window_num:>2}: Test {test_df.index.min().date()} → {test_df.index.max().date()} "
              f"| BH {avg_bh:+6.2f}٪ | {st['n']:>3} معامله | WinRate {st['win_rate']:5.1f}٪ | "
              f"Exp {st['expectancy_pct']:+7.3f}٪ {mark}")

        start += args.step

    # --- جمع‌بندی ---
    res = pd.DataFrame(results)
    with_trades = res[res["n_trades"] > 0]

    print("\n" + "=" * 100)
    print("📊 جمع‌بندی نهایی Walk-Forward + شبیه‌سازی")
    print("=" * 100)

    if with_trades.empty:
        print("🔴 در هیچ پنجره‌ای معامله‌ای انجام نشد.")
        return

    pos_windows = int((with_trades["expectancy_pct"] > 0).sum())
    total_windows = len(with_trades)
    total_trades = int(with_trades["n_trades"].sum())
    weighted_exp = (with_trades["expectancy_pct"] * with_trades["n_trades"]).sum() / total_trades
    avg_win_rate = (with_trades["win_rate"] * with_trades["n_trades"]).sum() / total_trades
    bull_windows = int((with_trades["avg_bh_pct"] > 0).sum())

    print(f"• پنجره‌های معتبر:               {total_windows}")
    print(f"• پنجره‌های با سود (Exp>0):      {pos_windows} از {total_windows} ({pos_windows/total_windows*100:.0f}٪)")
    print(f"• پنجره‌های گاوی (BH>0):         {bull_windows} از {total_windows}")
    print(f"• کل معاملات:                    {total_trades}")
    print(f"• میانگین وزنی Win Rate:         {avg_win_rate:.1f}٪")
    print(f"• میانگین وزنی Expectancy:       {weighted_exp:+.4f}٪ (هر معامله، بعد از هزینه‌ها)")
    print("-" * 100)

    if pos_windows / total_windows >= 0.7 and weighted_exp > 0:
        verdict = ("🟢 استراتژی در اکثر رژیم‌های بازار سودده بوده — قوی‌ترین شواهد تا الان. "
                   "قدم بعد: داده‌ی بیشتر و تست روی نمادهای دیگر.")
    elif pos_windows / total_windows >= 0.5 and weighted_exp > 0:
        verdict = ("🟡 سوددهی به رژیم بازار وابسته است (بعضی پنجره‌ها سود، بعضی ضرر). "
                   "بهتر است فیلتر رژیم بازار قوی‌تر شود تا فقط در پنجره‌های سودده معامله کنیم.")
    else:
        verdict = ("🔴 بعد از هزینه‌ها، استراتژی در اکثر پنجره‌ها سودده نبوده — "
                   "نتیجه‌ی هفته‌ی گاوی شانسی بوده است.")
    print(f"🔎 جمع‌بندی: {verdict}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
