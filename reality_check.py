"""
reality_check.py
----------------
آزمون White's Reality Check (نسخه‌ی عملی) برای سنجش صادقانه‌ی edge مدل.

مشکل (سوگیری داده‌کاوی — آرونسون، فصل ۶): وقتی بین چند پیکربندی جستجو
می‌کنیم (مثلاً آستانه‌های مختلف ADX) و «بهترین» را انتخاب می‌کنیم، آن بهترین
به‌صورت سیستماتیک بیش‌ازحد خوب است. حتی با edge واقعیِ صفر، شانس می‌تواند
یکی از ترکیب‌ها را مثبت نشان دهد. هرچه بیشتر جستجو کنیم، این شانس بیشتر می‌شود.

راه‌حل: توزیع «بهترین عملکردِ صرفاً شانسی» را می‌سازیم:
  1. مدل یک‌بار روی Train آموزش می‌بیند (دقیقاً مثل backtest_model).
  2. «بهترین Expectancy واقعی» = حداکثر روی همه‌ی پیکربندی‌های جستجو.
  3. N بار جایگشت: احتمال‌های مدل را در طول زمان جابه‌جا می‌کنیم (شانس خالص —
     رابطه‌ی سیگنال با بازده‌های آینده کاملاً از بین می‌رود)، با همین
     پیکربندی‌ها شبیه‌سازی می‌کنیم و «بهترین شانسی» را ثبت می‌کنیم.
  4. p-value = سهم جایگشت‌هایی که بهترین‌شان >= بهترین واقعی بود.

تفسیر:
  - p-value کوچک (< 0.05): نتیجه‌ی واقعی به‌سختی می‌تواند شانسی باشد.
  - p-value بزرگ: نمی‌توانیم فرض «بدون edge» را رد کنیم — نتیجه صادقانه است
    (این «شکست» هم اطلاعات است، نه بهانه).

اجرا:
    python reality_check.py --horizon 4 --threshold 0.30 --n-perms 200
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
logger = logging.getLogger("RealityCheck")

# پیکربندی‌های جستجو شده (همان داده‌کاوی‌ای که در backtest_model --sweep-adx انجام می‌شد)
ADX_GRID = [0, 15, 20, 25, 30]


def train_and_predict(df: pd.DataFrame, horizon: int, val_fraction: float = 0.2):
    """آموزش LightGBM (دقیقاً مثل backtest_model) و برگرداندن Validation + احتمال‌ها."""
    from features import get_feature_columns
    import lightgbm as lgb

    target_col = f"target_up_{horizon}"
    if target_col not in df.columns:
        raise RuntimeError(f"ستون {target_col} در دیتاست نیست.")

    feature_cols = get_feature_columns(df)
    if "symbol" not in feature_cols:
        feature_cols = feature_cols + ["symbol"]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    X["symbol"] = X["symbol"].astype("category")

    split_idx = int(len(X) * (1 - val_fraction))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    val_df = df.iloc[split_idx:].copy()

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_X=X_val, eval_y=y_val,
              callbacks=[lgb.early_stopping(40, verbose=False)])
    probs = model.predict_proba(X_val)[:, 1]
    return val_df, probs


def best_expectancy(val_df, probs, horizon, threshold, fee, use_funding,
                    adx_grid):
    """
    بهترین Expectancy (حداکثر روی پیکربندی‌های ADX) برای یک بردار احتمال.
    برمی‌گرداند: (بهترین Expectancy، ADX برنده، تعداد معاملات برنده)
    """
    best = -float("inf")
    best_adx = None
    best_n = 0
    for madx in adx_grid:
        trades = simulate_all(val_df, probs, horizon, threshold, fee,
                              float(madx), use_funding)
        st = compute_stats(trades)
        exp = st["expectancy_pct"] if st["n"] > 0 else -float("inf")
        if exp > best:
            best = exp
            best_adx = madx
            best_n = st["n"]
    return best, best_adx, best_n


def main():
    parser = argparse.ArgumentParser(description="آزمون White's Reality Check برای edge مدل")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None,
                    help="آستانه‌ی ثابت احتمال (پیش‌فرض: از روی چندک probs محاسبه می‌شود)")
    parser.add_argument("--quantile", type=float, default=0.75,
                        help="چندک احتمال برای آستانه (0.75 = فقط 25٪ بالای احتمال‌ها وارد شوند)")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--n-perms", type=int, default=200,
                        help="تعداد جایگشت‌های شانسی (بیشتر = دقیق‌تر ولی کندتر)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-funding", action="store_true",
                        help="فاندینگ را در محاسبه لحاظ نکن")
    args = parser.parse_args()

    df = load_combined_dataset([s.upper() for s in args.symbols], args.interval)
    val_df, probs = train_and_predict(df, args.horizon)
    use_funding = not args.no_funding

    # آستانه: اگر ثابت داده نشده، از چندک توزیع احتمال‌های مدل محاسبه می‌شود
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = float(np.quantile(probs, args.quantile))
    print(f"[تشخیص] آستانه‌ی نهایی: {threshold:.4f} (چندک {args.quantile}) — "
          f"{int((probs >= threshold).sum())} از {len(probs)} عبور می‌کنند")

    # --- تشخیص: آیا احتمال‌های مدل معنی‌دارند یا ثابت؟ ---
    n_unique = len(np.unique(np.round(probs, 6)))
    n_pass = int((probs >= threshold).sum())
    print("\n[تشخیص] احتمال‌های مدل روی Validation:")
    print(f"   min={probs.min():.4f} | max={probs.max():.4f} | "
          f"mean={probs.mean():.4f} | std={probs.std():.4f} | "
          f"مقادیر یکتا(6 رقم): {n_unique}")
    print(f"   تعداد عبور از آستانه {threshold}: {n_pass} از {len(probs)}")
    if n_unique <= 2:
        print("   ⚠️ احتمال‌ها تقریباً ثابت‌اند -> مدل چیزی یاد نگرفته (خروجی ≈ احتمال پایه).")
        print("      آزمون جایگشت در این حالت بی‌معنی است؛ مشکل از مدل/داده است، نه از آزمون.")
    test_shuffled = np.random.default_rng(args.seed).permutation(probs)
    same_mask = (probs >= threshold) == (test_shuffled >= threshold)
    print(f"   [تشخیص] بعد از یک جایگشت: محل عبورها {int(same_mask.sum())} از {len(probs)} "
          f"یکسان ماند (اگر = همه -> احتمال‌ها ثابت‌اند)")
    print()

    # ۱) بهترین واقعی (با همان جستجوی داده‌کاوی)
    actual_best, actual_adx, actual_n = best_expectancy(
        val_df, probs, args.horizon, threshold, args.fee,
        use_funding, ADX_GRID)
    if actual_best == -float("inf"):
        print("❌ هیچ معامله‌ای در هیچ پیکربندی انجام نشد — آستانه/فیلتر خیلی سخت‌گیرانه است.")
        return

    # ۲) توزیع صفر (جایگشت احتمال‌ها در زمان)
    logger.info(f"اجرای {args.n_perms} جایگشت شانسی (هر کدام شامل {len(ADX_GRID)} پیکربندی)...")
    rng = np.random.default_rng(args.seed)
    null_best = np.empty(args.n_perms)
    null_adx = []
    for k in range(args.n_perms):
        shuffled = rng.permutation(probs)  # رابطه‌ی سیگنال با زمان/بازده نابود می‌شود
        b, madx, _ = best_expectancy(val_df, shuffled, args.horizon,
                                     threshold, args.fee, use_funding,
                                     ADX_GRID)
        null_best[k] = b
        null_adx.append(madx)
        if (k + 1) % 50 == 0:
            logger.info(f"  جایگشت {k + 1}/{args.n_perms} ...")

    # ۳) p-value مونت‌کارلو
    p_value = (1 + int((null_best >= actual_best).sum())) / (1 + args.n_perms)

    # ۴) گزارش
    mean_null = null_best.mean()
    p90 = np.percentile(null_best, 90)
    p95 = np.percentile(null_best, 95)
    p99 = np.percentile(null_best, 99)
    max_null = null_best.max()

    print("\n" + "=" * 82)
    print("🎲 WHITE'S REALITY CHECK — آیا بهترین نتیجه‌ی ما شانسی است؟")
    print("=" * 82)
    print(f"• بازه‌ی Validation:     {val_df.index.min()} تا {val_df.index.max()}")
    print(f"• پیکربندی:             horizon={args.horizon} | threshold={threshold} "
          f"| fee={args.fee*100:.3f}٪ | funding={'بله' if use_funding else 'خیر'}")
    print(f"• تعداد پیکربندی‌های جستجو: {len(ADX_GRID)} (ADX در {ADX_GRID})")
    print("-" * 82)
    print(f"✅ بهترین Expectancy واقعی: {actual_best:+.4f}٪/معامله "
          f"(ADX≥{actual_adx}, {actual_n} معامله)")
    print(f"🎲 توزیع بهترین شانسی ({args.n_perms} جایگشت):")
    print(f"   میانگین: {mean_null:+.4f}٪ | p90: {p90:+.4f} | "
          f"p95: {p95:+.4f} | p99: {p99:+.4f} | حداکثر: {max_null:+.4f}")
    print(f"   ← جستجو به‌تنهایی به‌طور میانگین {mean_null:+.4f}٪ شانس می‌سازد")
    print("-" * 82)
    print(f"🔑 p-value = {p_value:.4f}   "
          f"(احتمال اینکه شانس به خوبیِ نتیجه‌ی واقعی برسد)")
    print("-" * 82)

    if p_value < 0.05:
        print("🟢 نتیجه‌ی صادقانه: با این سطح، نتیجه به‌سختی می‌تواند شانسی باشد —")
        print("   احتمال وجود edge واقعی وجود دارد (هنوز نیاز به داده‌ی بیشتر و Walk-Forward دارد).")
    elif p_value < 0.20:
        print("🟡 نتیجه‌ی صادقانه: شواهد ضعیف‌اند — نتیجه‌ی واقعی ممکن است تا حدی")
        print("   شانسی باشد. برای نتیجه‌گیری مطمئن‌تر، داده/جایگشت بیشتری لازم است.")
    else:
        print("🔴 نتیجه‌ی صادقانه: نمی‌توانیم فرض «بدون edge» را رد کنیم — بهترین نتیجه‌ی")
        print("   ما با این تعداد جستجو می‌تواند صرفاً شانسِ داده‌کاوی باشد. این «شکست»")
        print("   هم اطلاعات است: قبل از بهبود مدل، باید پایه‌ی سیگنال را قوی‌تر کنیم.")
    print("=" * 82 + "\n")

    # ذخیره‌ی نتایج برای مقایسه‌های بعدی
    results = pd.DataFrame({
        "metric": ["actual_best_expectancy_pct", "actual_best_adx", "actual_trades",
                   "n_permutations", "p_value", "null_mean_pct", "null_p90_pct",
                   "null_p95_pct", "null_p99_pct", "null_max_pct"],
        "value": [actual_best, actual_adx, actual_n, args.n_perms, p_value,
                  mean_null, p90, p95, p99, max_null],
    })
    results.to_csv("logs/reality_check_results.csv", index=False)
    print("نتایج کامل در logs/reality_check_results.csv ذخیره شد.")


if __name__ == "__main__":
    main()
