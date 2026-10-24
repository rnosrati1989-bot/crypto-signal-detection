"""
compare_smart_money.py
----------------------
آزمون صادقانه: ویژگی‌های Smart Money ارزش پیش‌بینی‌کنندگی اضافه می‌کنند یا نه؟

حالت‌های اجرا (--mode):
  both    : مقایسه‌ی دو مدل روی یک تقسیم‌بندی زمانیِ یکسان
             ۱) baseline: همه‌ی ویژگی‌ها به‌جز Smart Money
             ۲) کامل    : همه‌ی ویژگی‌ها (شامل Smart Money)
            اگر AUC مدل دوم به‌طور معنادار (≥ 0.01) بالاتر باشد → SM edge اضافه کرده.

  sm-only : فقط ۲۲ ویژگی Smart Money به مدل داده می‌شود.
            این «آزمون تشخیص» است:
            - AUC > 0.55  → اطلاعات در این ویژگی‌ها هست ولی با بقیه تکراری است
                            (راه‌حل: هرس/بهبود، نه حذف کامل)
            - AUC ≈ 0.5   → فرم فعلی ویژگی‌ها سیگنال واقعی نمی‌گیرد
                            (راه‌حل: بازطراحی مفهوم‌ها، نه افزودن همین‌ها)

نحوه‌ی اجرا (از داخل پوشه‌ی پروژه):
    python compare_smart_money.py --interval 15m --horizon 4
    python compare_smart_money.py --interval 15m --horizon 4 --mode sm-only

خروجی:
    - AUC مدل‌ها (روی Validation — داده‌ی ندیده)
    - تفاوت AUC و قضاوت صادقانه
    - مهم‌ترین ویژگی‌های Smart Money از نظر Gain
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from features import get_feature_columns
from smart_money import get_smart_money_columns, get_core_sm_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CompareSM")

CACHE_DIR = "data_cache"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def load_combined_dataset(symbols, interval):
    """خواندن همه‌ی دیتاست‌های ویژگی و ترکیب آن‌ها (مثل train_model)."""
    frames = []
    for symbol in symbols:
        path = os.path.join(CACHE_DIR, f"{symbol}_{interval}_features.csv")
        if not os.path.exists(path):
            logger.warning(f"فایل دیتاست پیدا نشد: {path} — رد شد.")
            continue
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        sym = pd.DataFrame({"symbol": symbol}, index=df.index)
        frames.append(pd.concat([df, sym], axis=1))
        logger.info(f"{symbol}: {len(df)} ردیف")

    if not frames:
        raise RuntimeError("هیچ دیتاست ویژگی‌ای پیدا نشد. اول build_dataset.py را اجرا کنید.")

    df = pd.concat(frames).sort_index()
    logger.info(f"کل داده: {len(df)} ردیف")
    return df


def train_eval(X_train, y_train, X_val, y_val, n_estimators):
    """آموزش LightGBM با پارامترهای یکسان و برگرداندن AUC + اهمیت ویژگی‌ها."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_X=X_val, eval_y=y_val,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    return model, auc


def print_sm_importance(model, feature_cols, sm_cols):
    """رتبه‌بندی ویژگی‌های Smart Money از نظر Gain در مدل کامل."""
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    sm_imp = importance[sm_cols].sort_values(ascending=False)
    total_imp = importance.sum()
    print(f"\n📌 رتبه‌ی ویژگی‌های Smart Money (از {len(feature_cols)} ویژگی):")
    print(f"    {'ویژگی':<24} {'رتبه':<6} {'Gain':<10} {'سهم از کل':<10}")
    print("    " + "-" * 54)
    for feat, imp in sm_imp.items():
        rank = int((importance > imp).sum()) + 1
        share = imp / total_imp * 100 if total_imp else 0
        print(f"    {feat:<24} {rank:<6} {imp:<10.0f} {share:<10.2f}%")
    print("    " + "-" * 54)
    if sm_imp.empty:
        print("    (هیچ ستون Smart Money در دیتاست نیست — build_dataset را دوباره اجرا کنید)")


def run_sm_only(df, feature_cols, sm_cols, y, split_idx, n_estimators):
    """آزمون تشخیص: مدل فقط با ویژگی‌های Smart Money."""
    print("\n" + "=" * 76)
    print("🧪 آزمون تشخیص: فقط ویژگی‌های Smart Money")
    print("=" * 76)
    X = df[sm_cols].copy()
    X["symbol"] = df["symbol"].astype("category")
    _, auc = train_eval(
        X.iloc[:split_idx], y.iloc[:split_idx],
        X.iloc[split_idx:], y.iloc[split_idx:],
        n_estimators
    )
    base_rate = float(y.iloc[split_idx:].mean())
    print(f"    AUC فقط با Smart Money : {auc:.4f}")
    print(f"    نرخ پایه‌ی صعود        : {base_rate*100:.1f}%")
    print("-" * 76)
    if auc >= 0.58:
        print("✅ اطلاعات در این ویژگی‌ها هست — مشکل «تکراری بودن» با بقیه است.")
        print("   قدم بعدی: هرس (حذف ویژگی‌های بی‌استفاده) و دوباره تست.")
    elif auc >= 0.55:
        print("🟡 سیگنال ضعیف ولی موجود — نیاز به بهبود/هرس دارد.")
    else:
        print("🔴 فرم فعلی ویژگی‌ها سیگنال واقعی نمی‌گیرد.")
        print("   قدم بعدی: بازطراحی مفهوم‌ها (واگرایی، Wyckoff، پروفایل) — نه افزودن همین‌ها.")
    print("=" * 76)
    return auc


def run_core(df, base_cols, sm_cols, y, split_idx, n_estimators):
    """هرس: baseline در برابر baseline + فقط ۶ ویژگی «هسته‌ی قوی» Smart Money."""
    core_cols = get_core_sm_columns(df)
    if not core_cols:
        print("\n❌ ستون‌های هسته‌ی Smart Money در دیتاست نیست.")
        return
    print("\n" + "=" * 76)
    print("🧪 هرس: فقط ۶ ویژگی «هسته‌ی قوی» Smart Money")
    print("=" * 76)
    print(f"    هسته: {', '.join(core_cols)}")

    Xb = df[base_cols].copy()
    Xb["symbol"] = df["symbol"].astype("category")
    _, auc_base = train_eval(
        Xb.iloc[:split_idx], y.iloc[:split_idx],
        Xb.iloc[split_idx:], y.iloc[split_idx:], n_estimators)

    add_cols = [c for c in core_cols if c not in base_cols]
    Xc = df[base_cols + add_cols].copy()
    Xc["symbol"] = df["symbol"].astype("category")
    _, auc_core = train_eval(
        Xc.iloc[:split_idx], y.iloc[:split_idx],
        Xc.iloc[split_idx:], y.iloc[split_idx:], n_estimators)

    diff = auc_core - auc_base
    print(f"\n    AUC بدون SM (baseline) : {auc_base:.4f}")
    print(f"    AUC با هسته‌ی SM       : {auc_core:.4f}")
    print(f"    تفاوت                  : {diff:+.4f}")
    print("-" * 76)
    if diff >= 0.01:
        print("✅ هسته‌ی Smart Money edge اضافه کرد — هرس جواب داد!")
    elif diff >= 0.005:
        print("🟡 بهبود جزئی — باید با Walk-Forward تأیید شود.")
    elif diff <= -0.01:
        print("🔴 حتی هسته هم ضرر دارد — این ویژگی‌ها با بقیه تکراری/مضرند.")
    else:
        print("⚪ تفاوت ناچیز — هسته نه کمک می‌کند نه ضرر.")
    print("=" * 76)


def run_both(df, feature_cols, sm_cols, base_cols, y, split_idx, n_estimators, args):
    """مقایسه‌ی اصلی: baseline (بدون SM) در برابر کامل (با SM)."""
    Xb = df[base_cols].copy()
    Xb["symbol"] = df["symbol"].astype("category")
    _, auc_base = train_eval(
        Xb.iloc[:split_idx], y.iloc[:split_idx],
        Xb.iloc[split_idx:], y.iloc[split_idx:],
        n_estimators
    )

    Xf = df[feature_cols].copy()
    Xf["symbol"] = df["symbol"].astype("category")
    model_full, auc_full = train_eval(
        Xf.iloc[:split_idx], y.iloc[:split_idx],
        Xf.iloc[split_idx:], y.iloc[split_idx:],
        n_estimators
    )

    diff = auc_full - auc_base
    print(f"\n📊 نتیجه روی Validation (داده‌ی ندیده):")
    print(f"    AUC بدون Smart Money : {auc_base:.4f}")
    print(f"    AUC با Smart Money   : {auc_full:.4f}")
    print(f"    تفاوت                : {diff:+.4f}")
    print("-" * 76)

    if diff >= 0.01:
        print("✅ Smart Money ارزش پیش‌بینی‌کنندگی معناداری اضافه کرده.")
    elif diff >= 0.005:
        print("🟡 Smart Money بهبود جزئی داده — باید با Walk-Forward تأیید شود.")
    elif diff <= -0.01:
        print("🔴 Smart Money عملکرد را بدتر کرده (احتمالاً overfit روی Train).")
    else:
        print("⚪ تفاوت ناچیز است — Smart Money روی این داده edge اضافه نکرده.")
    print("=" * 76)

    print_sm_importance(model_full, feature_cols, sm_cols)

    # ۱۵ ویژگی برتر کل (برای دیدن اینکه چه چیزهایی غالب هستند)
    importance = pd.Series(model_full.feature_importances_, index=feature_cols)
    top = importance.sort_values(ascending=False).head(15)
    print(f"\n📊 ۱۵ ویژگی برتر کل مدل:")
    for i, (feat, imp) in enumerate(top.items(), 1):
        sm_tag = " 🧠SM" if feat in sm_cols else ""
        print(f"    {i:>2}. {feat:<26} Gain={imp}{sm_tag}")


def main():
    parser = argparse.ArgumentParser(description="مقایسه‌ی مدل با/بدون ویژگی‌های Smart Money")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--mode", choices=["both", "sm-only", "core"], default="both",
                        help="both=مقایسه‌ی کامل | sm-only=فقط ویژگی‌های Smart Money")
    args = parser.parse_args()

    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        print("\n❌ کتابخانه‌ی lightgbm نصب نیست. اول اجرا کنید:")
        print("    pip install lightgbm scikit-learn\n")
        return

    df = load_combined_dataset([s.upper() for s in args.symbols], args.interval)

    target_col = f"target_up_{args.horizon}"
    if target_col not in df.columns:
        print(f"❌ ستون {target_col} در دیتاست نیست. با horizon یکسان build_dataset اجرا کنید.")
        return

    feature_cols = get_feature_columns(df)
    if "symbol" not in feature_cols:
        feature_cols = feature_cols + ["symbol"]
    sm_cols = get_smart_money_columns(df)
    base_cols = [c for c in feature_cols if c not in sm_cols]

    y = df[target_col].astype(int)
    split_idx = int(len(df) * (1 - args.val_fraction))

    print("\n" + "=" * 76)
    print("🧪 مقایسه‌ی صادقانه: ویژگی‌های Smart Money edge اضافه می‌کنند؟")
    print("=" * 76)
    print(f"• نمادها: {args.symbols} | بازه: {args.interval} | افق: {args.horizon} | حالت: {args.mode}")
    print(f"• کل ویژگی‌ها: {len(feature_cols)} | Smart Money: {len(sm_cols)} | بقیه: {len(base_cols)}")
    print(f"• Train: {split_idx} ردیف | Validation: {len(df) - split_idx} ردیف")
    print(f"• بازه‌ی Validation: {df.index[split_idx]} تا {df.index[-1]}")
    print("-" * 76)

    if args.mode == "sm-only":
        run_sm_only(df, feature_cols, sm_cols, y, split_idx, args.n_estimators)
    elif args.mode == "core":
        run_core(df, base_cols, sm_cols, y, split_idx, args.n_estimators)
    else:
        run_both(df, feature_cols, sm_cols, base_cols, y, split_idx,
                 args.n_estimators, args)


if __name__ == "__main__":
    main()
