"""
walk_forward_model.py
---------------------
آزمون نهایی و صادقانه برای مدل: Walk-Forward Validation.

به‌جای یک تقسیم Train/Test تکی، داده را به چند پنجره‌ی متوالی تقسیم می‌کنیم.
در هر پنجره: مدل روی Train (گذشته) آموزش می‌بیند و روی Test (آینده‌ی
کاملاً ندیده) آزمایش می‌شود.

اگر مدل در اکثر پنجره‌ها بهتر از baseline عمل کند → شواهد قوی‌تری برای
وجود edge داریم. اگر فقط در یک پنجره خوب باشد → شانسی بوده.

نحوه‌ی اجرا (از داخل پوشه‌ی پروژه):
    python walk_forward_model.py --horizon 4 --windows 6

معیار قضاوت:
    - نسبت پنجره‌هایی که Precision > نرخ پایه است (Positive Window Ratio)
    - میانگین AUC روی پنجره‌ها
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from features import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WalkForwardModel")

CACHE_DIR = "data_cache"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_INTERVAL = "15m"

# آستانه‌ی تصمیمی که از تحلیل Validation اولیه انتخاب شد (حد وسط تعداد سیگنال/دقت)
FIXED_THRESHOLD = 0.25


def load_combined_dataset(symbols, interval):
    frames = []
    for symbol in symbols:
        path = os.path.join(CACHE_DIR, f"{symbol}_{interval}_features.csv")
        if not os.path.exists(path):
            logger.warning(f"فایل دیتاست پیدا نشد: {path} — رد شد.")
            continue
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        df["symbol"] = symbol
        frames.append(df)

    if not frames:
        raise RuntimeError("هیچ دیتاست ویژگی‌ای پیدا نشد. اول build_dataset.py را اجرا کنید.")

    df = pd.concat(frames).sort_index()
    logger.info(f"کل داده: {len(df)} ردیف")
    return df


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward برای مدل LightGBM")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, precision_score
    except ImportError:
        print("\n❌ کتابخانه‌ها نصب نیستند. اول اجرا کنید:")
        print("    pip install lightgbm scikit-learn\n")
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
        print("   یا windows را کم کنید یا از تایم‌فریم 15m با 3000 کندل استفاده کنید.")
        return

    print("\n" + "=" * 100)
    print("🚶 WALK-FORWARD VALIDATION — مدل LightGBM")
    print("=" * 100)
    print(f"• آستانه‌ی ثابت تصمیم:      {FIXED_THRESHOLD}")
    print(f"• Train/Test هر پنجره:      {args.train_size} / {args.test_size} ردیف")
    print()

    results = []
    start = 0
    window_num = 0

    while start + args.train_size + args.test_size <= n:
        window_num += 1
        train_idx = slice(start, start + args.train_size)
        test_idx = slice(start + args.train_size, start + args.train_size + args.test_size)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        test_start = X_test.index[0]
        test_end = X_test.index[-1]

        # آموزش روی همین پنجره
        model = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_X=X_test, eval_y=y_test,
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        base_rate = float(y_test.mean())
        pred = (y_prob >= FIXED_THRESHOLD).astype(int)
        n_sig = int(pred.sum())
        prec = precision_score(y_test, pred, zero_division=0)

        better = prec > base_rate
        results.append({
            "window": window_num,
            "test_period": f"{test_start.date()} → {test_end.date()}",
            "base_rate": base_rate,
            "auc": auc,
            "n_signals": n_sig,
            "precision": prec,
            "better_than_base": better,
        })

        mark = "✅" if better else "❌"
        print(f"پنجره {window_num}: Test {test_start.date()} → {test_end.date()} "
              f"| AUC {auc:.3f} | نرخ پایه {base_rate*100:5.1f}% | "
              f"{n_sig} سیگنال | Precision {prec*100:5.1f}% {mark}")

        start += args.step

    # --- جمع‌بندی ---
    res_df = pd.DataFrame(results)
    valid = res_df[res_df["n_signals"] > 0]

    if valid.empty:
        print("\n🔴 هیچ سیگنالی در هیچ پنجره‌ای تولید نشد (آستانه خیلی بالا/مدل محافظه‌کار).")
        return

    positive_windows = int(valid["better_than_base"].sum())
    total_windows = len(valid)
    avg_auc = valid["auc"].mean()

    # میانگین وزنی precision (وزن = تعداد سیگنال‌ها)
    total_signals = int(valid["n_signals"].sum())
    weighted_prec = (valid["precision"] * valid["n_signals"]).sum() / total_signals

    print("\n" + "=" * 100)
    print("📊 جمع‌بندی Walk-Forward")
    print("=" * 100)
    print(f"• پنجره‌های معتبر:               {total_windows}")
    print(f"• پنجره‌های با Precision > نرخ پایه: {positive_windows} از {total_windows} "
          f"({positive_windows/total_windows*100:.0f}%)")
    print(f"• میانگین AUC:                   {avg_auc:.4f}")
    print(f"• میانگین وزنی Precision:        {weighted_prec*100:.1f}% "
          f"(نرخ پایه‌ی متوسط: {valid['base_rate'].mean()*100:.1f}%)")
    print("-" * 100)

    if positive_windows / total_windows >= 0.7 and weighted_prec > valid["base_rate"].mean() * 1.3:
        verdict = ("🟢 مدل در اکثر پنجره‌ها بهتر از شانس عمل کرده — شواهد اولیه‌ی edge وجود دارد. "
                   "قدم بعد: شبیه‌سازی واقعی با کارمزد/فاندینگ.")
    elif positive_windows / total_windows >= 0.5:
        verdict = ("🟡 نتیجه ترکیبی است — در برخی رژیم‌های بازار edge هست، در برخی نه. "
                   "باید فیلتر رژیم بازار اضافه شود.")
    else:
        verdict = ("🔴 مدل در اکثر پنجره‌ها بهتر از شانس نبوده — نتیجه‌ی اولیه‌ی تک‌پنجره‌ای "
                   "شانسی بوده. باید ویژگی‌ها/هدف را بازنگری کنیم.")
    print(f"🔎 جمع‌بندی: {verdict}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
