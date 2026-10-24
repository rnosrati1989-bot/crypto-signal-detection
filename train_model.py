"""
train_model.py
--------------
آموزش اولین مدل یادگیری ماشین (LightGBM) روی دیتاست ویژگی‌ها.

هدف: پیش‌بینی «آیا قیمت horizon کندل بعد حداقل X% بالا می‌رود؟»
این یک «آزمون صداقت» است: قبل از هر ادعایی، می‌بینیم مدل روی داده‌ای
که ندیده (Validation) بهتر از حدس ساده (baseline) عمل می‌کند یا نه.

نحوه‌ی اجرا (از داخل پوشه‌ی پروژه):
    python train_model.py --horizon 4

خروجی:
    - معیارهای ارزیابی روی Validation (داده‌ی ندیده)
    - جدول آستانه‌های تصمیم + بهترین آستانه (Youden)
    - مقایسه‌ی صادقانه با baseline
    - اهمیت ویژگی‌ها و ذخیره‌ی مدل
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
logger = logging.getLogger("TrainModel")

CACHE_DIR = "data_cache"
MODELS_DIR = "models"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_INTERVAL = "15m"


def load_combined_dataset(symbols, interval, horizon):
    """خواندن همه‌ی دیتاست‌های ویژگی و ترکیب آن‌ها در یک DataFrame."""
    frames = []
    for symbol in symbols:
        path = os.path.join(CACHE_DIR, f"{symbol}_{interval}_features.csv")
        if not os.path.exists(path):
            logger.warning(f"فایل دیتاست پیدا نشد: {path} — رد شد.")
            continue
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        df["symbol"] = symbol
        frames.append(df)
        logger.info(f"{symbol}: {len(df)} ردیف")

    if not frames:
        raise RuntimeError("هیچ دیتاست ویژگی‌ای پیدا نشد. اول build_dataset.py را اجرا کنید.")

    df = pd.concat(frames)
    df = df.sort_index()  # مرتب‌سازی زمانی — برای تقسیم زمانی درست
    logger.info(f"کل داده: {len(df)} ردیف از {len(symbols)} نماد")
    return df


def main():
    parser = argparse.ArgumentParser(description="آموزش اولین مدل LightGBM")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="سهم داده‌ی اعتبارسنجی از انتهای بازه (پیش‌فرض 0.2)")
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("\n❌ کتابخانه‌ی lightgbm نصب نیست. اول اجرا کنید:")
        print("    pip install lightgbm scikit-learn\n")
        return

    # --- ۱. بارگذاری داده ---
    df = load_combined_dataset([s.upper() for s in args.symbols],
                               args.interval, args.horizon)

    target_col = f"target_up_{args.horizon}"
    if target_col not in df.columns:
        print(f"❌ ستون {target_col} در دیتاست نیست. با horizon یکسان build_dataset اجرا کنید.")
        return

    # --- ۲. ستون‌های ویژگی + برچسب ---
    feature_cols = get_feature_columns(df)
    if "symbol" not in feature_cols:
        feature_cols = feature_cols + ["symbol"]
    X = df[feature_cols]
    y = df[target_col].astype(int)

    # symbol را به‌عنوان ویژگی دسته‌ای (categorical) به LightGBM معرفی می‌کنیم
    X["symbol"] = X["symbol"].astype("category")

    # --- ۳. تقسیم زمانی (بدون shuffle — ترتیب زمان حفظ می‌شود) ---
    split_idx = int(len(X) * (1 - args.val_fraction))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"Train: {len(X_train)} ردیف | Validation: {len(X_val)} ردیف "
                f"(بازه: {X_val.index[0]} تا {X_val.index[-1]})")

    # --- ۴. آموزش LightGBM ---
    model = lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
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

    # --- ۵. ارزیابی ---
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, roc_curve)

    y_prob = model.predict_proba(X_val)[:, 1]
    base_rate = float(y_val.mean())
    auc = roc_auc_score(y_val, y_prob)

    # بهترین آستانه با معیار Youden (بیشترین TPR - FPR)
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    best_thr = float(thresholds[best_idx])
    best_pred = (y_prob >= best_thr).astype(int)
    best_prec = precision_score(y_val, best_pred, zero_division=0)
    best_rec = recall_score(y_val, best_pred, zero_division=0)

    print("\n" + "=" * 76)
    print("🤖 گزارش آموزش اولین مدل — نتیجه‌ی صادقانه")
    print("=" * 76)
    print(f"• نمونه‌های Train:              {len(X_train)}")
    print(f"• نمونه‌های Validation:         {len(X_val)}")
    print(f"• تعداد ویژگی:                 {len(feature_cols)}")
    print(f"• نرخ پایه‌ی کلاس «صعود»:       {base_rate*100:.1f}%")
    print(f"• AUC:                          {auc:.4f}")
    print("-" * 76)

    if auc > 0.55:
        print("✅ AUC بالاتر از 0.55 — مدل چیزی یاد گرفته (نیاز به تأیید با Walk-Forward)")
    elif auc > 0.52:
        print("🟡 AUC نزدیک به 0.52 — سیگنال خیلی ضعیف/نامطمئن")
    else:
        print("🔴 AUC نزدیک 0.5 — مدل هنوز edge قابل‌اتکایی پیدا نکرده")
    print("=" * 76)

    # --- ۶. جدول آستانه‌های تصمیم ---
    print("\n📊 تأثیر آستانه‌ی تصمیم (روی Validation — داده‌ی ندیده):")
    print(f"    نرخ پایه‌ی صعود = {base_rate*100:.1f}%  |  اگر Precision از این بالاتر باشد، سیگنال‌ها اطلاعات واقعی دارند")
    print("    " + "-" * 62)
    print("    آستانه | سیگنال‌ها | Precision | Recall | دقت")
    print("    " + "-" * 62)
    for thr in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]:
        pred = (y_prob >= thr).astype(int)
        n_sig = int(pred.sum())
        prec = precision_score(y_val, pred, zero_division=0)
        rec = recall_score(y_val, pred, zero_division=0)
        acc = accuracy_score(y_val, pred)
        marker = " <-- بهترین" if abs(thr - best_thr) < 0.01 else ""
        print(f"    {thr:.2f}   | {n_sig:5d}     | {prec*100:6.2f}%  | {rec*100:5.2f}%  | {acc*100:5.2f}%{marker}")
    print("    " + "-" * 62)
    print(f"    بهترین آستانه (Youden) = {best_thr:.3f} → "
          f"Precision {best_prec*100:.1f}% | Recall {best_rec*100:.1f}% "
          f"| {int(best_pred.sum())} سیگنال")

    # --- ۷. اهمیت ویژگی‌ها ---
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print("\n📌 ۱۵ ویژگی مهم (بر اساس Gain):")
    for feat, imp in importance.head(15).items():
        print(f"    {feat}: {imp}")

    # --- ۸. ذخیره‌ی مدل ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"lightgbm_up_{args.horizon}.txt")
    model.booster_.save_model(model_path)
    print(f"\n💾 مدل ذخیره شد: {model_path}")
    print("⚠️ نکته: این مدل فقط برای آزمایش است؛ قبل از هر استفاده‌ی واقعی،")
    print("   باید با Walk-Forward و هزینه‌های واقعی (کارمزد/فاندینگ) ارزیابی شود.")


if __name__ == "__main__":
    main()
