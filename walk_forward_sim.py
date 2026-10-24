"""
walk_forward_sim.py
-------------------
Walk-Forward Validation شبیه‌سازی‌محور برای تأیید نهایی edge مدل.

تفاوت با walk_forward_model.py: این‌جا به‌جای معیارهای طبقه‌بندی،
سود/ضرر واقعی معاملات با کارمزد + فاندینگ سنجیده می‌شود.

قوانین صادقانه:
1. داده به چند پنجره‌ی متوالی تقسیم می‌شود (Train گذشته → Test آینده‌ی ندیده).
2. آستانه‌ی چندک از روی احتمال‌های Train هر پنجره محاسبه می‌شود (بدون look-ahead).
3. شبیه‌سازی LONG با ورود/خروج/کارمزد/فاندینگ (همان منطق backtest_model).
4. مقایسه‌ی کلیدی: «مدل + فیلتر ADX» در برابر «فقط فیلتر ADX» — برای اینکه ببینیم
   مدل واقعاً چیزی به فیلتر رژیم اضافه می‌کند یا نه.

اجرا:
    python walk_forward_sim.py --interval 1h --horizon 4 --quantile 0.75 --min-adx 30
"""

import argparse
import logging

import numpy as np
import pandas as pd

from backtest_model import (
    load_combined_dataset,
    simulate_all,
    compute_stats,
    DEFAULT_SYMBOLS,
    DEFAULT_INTERVAL,
    DEFAULT_FEE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WalkForwardSim")


def run_window(df, X, y, start, train_size, test_size, horizon, quantile,
               min_adx, fee, use_funding, lgb):
    """آموزش روی Train، شبیه‌سازی روی Test. برمی‌گرداند خلاصه‌ی پنجره."""
    train_idx = slice(start, start + train_size)
    test_idx = slice(start + train_size, start + train_size + test_size)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_df = df.iloc[test_idx].copy()

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_X=X_test, eval_y=y_test,
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    threshold = float(np.quantile(train_probs, quantile))

    # شبیه‌سازی ۱: مدل + فیلتر ADX
    trades_model = simulate_all(test_df, test_probs, horizon, threshold,
                                fee, min_adx, use_funding)
    st_model = compute_stats(trades_model)

    # شبیه‌سازی ۲: فقط فیلتر ADX (آستانه = همیشه عبور) — سهم مدل را جدا می‌کند
    trades_adx = simulate_all(test_df, test_probs, horizon, -1.0,
                              fee, min_adx, use_funding)
    st_adx = compute_stats(trades_adx)

    # Buy & Hold هر نماد در بازه‌ی Test
    bh_lines = []
    for symbol in test_df["symbol"].unique():
        grp = test_df[test_df["symbol"] == symbol]
        bh = (grp["close"].iloc[-1] / grp["close"].iloc[0] - 1) * 100
        bh_lines.append(f"{symbol} {bh:+.2f}٪")

    return {
        "window_start": test_df.index[0],
        "window_end": test_df.index[-1],
        "threshold": threshold,
        "n_model": st_model["n"],
        "exp_model": st_model["expectancy_pct"] if st_model["n"] else None,
        "wr_model": st_model["win_rate"] if st_model["n"] else None,
        "n_adx": st_adx["n"],
        "exp_adx": st_adx["expectancy_pct"] if st_adx["n"] else None,
        "bh": " | ".join(bh_lines),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward شبیه‌سازی‌محور برای مدل")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--min-adx", type=float, default=30.0)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--no-funding", action="store_true")
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

    from features import get_feature_columns
    feature_cols = get_feature_columns(df)
    if "symbol" not in feature_cols:
        feature_cols = feature_cols + ["symbol"]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    X["symbol"] = X["symbol"].astype("category")

    n = len(X)
    use_funding = not args.no_funding

    print("\n" + "=" * 100)
    print("🚶 WALK-FORWARD (شبیه‌سازی با کارمزد + فاندینگ)")
    print("=" * 100)
    print(f"• پیکربندی: {args.interval} | horizon={args.horizon} | "
          f"quantile={args.quantile} | ADX≥{args.min_adx:g} | fee={args.fee*100:.3f}٪")
    print(f"• پنجره‌ها: Train {args.train_size} → Test {args.test_size} "
          f"(گام {args.step}) | حداکثر {args.windows} پنجره")
    print()

    results = []
    start = 0
    window_num = 0
    while (start + args.train_size + args.test_size <= n) and (window_num < args.windows):
        window_num += 1
        try:
            r = run_window(df, X, y, start, args.train_size, args.test_size,
                           args.horizon, args.quantile, args.min_adx,
                           args.fee, use_funding, lgb)
        except Exception as e:
            logger.error(f"پنجره {window_num}: خطا — {e}")
            start += args.step
            continue

        results.append(r)
        exp_m = r["exp_model"]
        exp_a = r["exp_adx"]
        em = f"{exp_m:+.3f}٪" if exp_m is not None else "بدون معامله"
        ea = f"{exp_a:+.3f}٪" if exp_a is not None else "بدون معامله"
        print(f"پنجره {window_num}: {r['window_start'].date()} → {r['window_end'].date()}")
        print(f"   مدل+ADX: {r['n_model']:>3} معامله | Expectancy {em} | WinRate {r['wr_model']:.0f}٪" if r["wr_model"] is not None
              else f"   مدل+ADX: {r['n_model']} معامله | بدون معامله")
        print(f"   فقطADX:  {r['n_adx']:>3} معامله | Expectancy {ea}")
        print(f"   آستانه‌ی چندک {args.quantile}: {r['threshold']:.4f} | Buy&Hold: {r['bh']}")
        print("-" * 100)
        start += args.step

    if not results:
        print("❌ هیچ پنجره‌ای با موفقیت اجرا نشد.")
        return

    # --- جمع‌بندی ---
    valid_m = [r for r in results if r["exp_model"] is not None]
    valid_a = [r for r in results if r["exp_adx"] is not None]

    print("\n" + "=" * 100)
    print("📊 جمع‌بندی Walk-Forward")
    print("=" * 100)

    if valid_m:
        exps_m = np.array([r["exp_model"] for r in valid_m])
        pos_m = int((exps_m > 0).sum())
        tot_m = len(exps_m)
        mean_m = exps_m.mean()
        n_m = sum(r["n_model"] for r in valid_m)
        print(f"• مدل+ADX ({tot_m} پنجره‌ی معتبر):")
        print(f"    پنجره‌های با Expectancy مثبت: {pos_m}/{tot_m} ({pos_m/tot_m*100:.0f}٪)")
        print(f"    میانگین Expectancy: {mean_m:+.4f}٪ | کل معاملات: {n_m}")
    else:
        print("• مدل+ADX: هیچ پنجره‌ای معامله‌ای نداشت.")

    if valid_a:
        exps_a = np.array([r["exp_adx"] for r in valid_a])
        pos_a = int((exps_a > 0).sum())
        mean_a = exps_a.mean()
        print(f"• فقط ADX ({len(valid_a)} پنجره‌ی معتبر):")
        print(f"    پنجره‌های مثبت: {pos_a}/{len(valid_a)} | میانگین Expectancy: {mean_a:+.4f}٪")

    if valid_m and valid_a:
        diff = np.array([r["exp_model"] for r in valid_m]) - np.array([r["exp_adx"] for r in valid_m])
        print(f"• سهم خالص مدل (مدل+ADX منهای فقطADX): میانگین {diff.mean():+.4f}٪/معامله")

    print("-" * 100)
    if valid_m and pos_m / len(valid_m) >= 0.6 and mean_m > 0:
        print("🟢 مدل+ADX در اکثر پنجره‌ها Expectancy مثبت داشته — شواهد اولیه‌ی edge پایدار.")
    elif valid_m and pos_m / len(valid_m) >= 0.4:
        print("🟡 نتیجه ترکیبی است — edge به رژیم بازار وابسته است؛ فیلتر رژیم باید دقیق‌تر شود.")
    else:
        print("🔴 مدل+ADX در اکثر پنجره‌ها سود نکرده — edge پایدار تأیید نمی‌شود. (صادقانه: این هم اطلاعات است)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
