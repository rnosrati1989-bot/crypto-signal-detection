"""
analyze_sm_features.py
----------------------
تشخیص «جهت» هر ویژگی Smart Money نسبت به برچسب (فقط روی Train).

چرا؟ در آزمون sm-only، AUC روی 15m = 0.446 شد — یعنی مدل «برعکس» پیش‌بینی
می‌کرد. این یعنی بعضی ویژگی‌ها اطلاعات دارند ولی جهت/علامتشان نسبت به هدف
وارونه است. این اسکریپت برای هر ویژگی یک AUC تک‌متغیره و همبستگی با هدف
محاسبه می‌کند تا ببینیم کدام ویژگی‌ها وارونه‌اند.

قانون تفسیر (AUC تک‌متغیره — مقدار خود ویژگی به‌عنوان امتیاز):
    AUC < 0.48  → رابطه‌ی وارونه (جهت تعریف ویژگی باید اصلاح شود)
    AUC > 0.52  → رابطه‌ی مستقیم
    وسط        → بی‌اطلاع یا غیرخطی

نکته‌ی صداقت: محاسبه فقط روی ۸۰٪ اول (Train) انجام می‌شود تا Validation
برای ارزیابی نهایی دست‌نخورده بماند.

نحوه‌ی اجرا:
    python analyze_sm_features.py --interval 15m --horizon 4
    python analyze_sm_features.py --interval 1h --horizon 4
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from features import get_feature_columns
from smart_money import get_smart_money_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AnalyzeSM")

CACHE_DIR = "data_cache"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def load_combined_dataset(symbols, interval):
    frames = []
    for symbol in symbols:
        path = os.path.join(CACHE_DIR, f"{symbol}_{interval}_features.csv")
        if not os.path.exists(path):
            logger.warning(f"فایل دیتاست پیدا نشد: {path} — رد شد.")
            continue
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        sym = pd.DataFrame({"symbol": symbol}, index=df.index)
        frames.append(pd.concat([df, sym], axis=1))
    if not frames:
        raise RuntimeError("هیچ دیتاست ویژگی‌ای پیدا نشد. اول build_dataset.py را اجرا کنید.")
    return pd.concat(frames).sort_index()


def main():
    parser = argparse.ArgumentParser(description="تشخیص جهت ویژگی‌های Smart Money")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    df = load_combined_dataset([s.upper() for s in args.symbols], args.interval)
    target_col = f"target_up_{args.horizon}"
    if target_col not in df.columns:
        print(f"❌ ستون {target_col} در دیتاست نیست.")
        return

    sm_cols = get_smart_money_columns(df)
    y = df[target_col].astype(int)
    split_idx = int(len(df) * (1 - args.val_fraction))

    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 80)
    print("🧭 تشخیص جهت ویژگی‌های Smart Money (روی Train فقط)")
    print("=" * 80)
    print(f"• بازه: {args.interval} | افق: {args.horizon} | Train: {split_idx} ردیف")
    print(f"• نرخ پایه‌ی صعود در Train: {y.iloc[:split_idx].mean()*100:.1f}%")
    print("-" * 80)
    print(f"    {'ویژگی':<24} {'AUC':>7} {'جهت':>12} {'همبستگی':>9} {'n':>7}")
    print("    " + "-" * 62)

    rows = []
    for col in sm_cols:
        x = df[col].iloc[:split_idx]
        yt = y.iloc[:split_idx]
        mask = x.notna() & yt.notna()
        if int(mask.sum()) < 200:
            continue
        xv = x[mask].to_numpy(dtype=float)
        yv = yt[mask].to_numpy(dtype=int)
        if len(np.unique(yv)) < 2:
            continue
        try:
            auc = roc_auc_score(yv, xv)
        except ValueError:
            continue
        corr = (float(np.corrcoef(xv, yv)[0, 1])
                if np.std(xv) > 0 else np.nan)
        rows.append({"feature": col, "auc": auc, "corr": corr,
                     "n": int(mask.sum())})

    res = pd.DataFrame(rows).sort_values("auc")
    for _, r in res.iterrows():
        if r["auc"] < 0.48:
            direction = "وارونه 🔄"
        elif r["auc"] > 0.52:
            direction = "مستقیم ✅"
        else:
            direction = "خنثی"
        print(f"    {r['feature']:<24} {r['auc']:>7.4f} {direction:>12} "
              f"{r['corr']:>+9.4f} {r['n']:>7d}")

    print("    " + "-" * 62)

    inverted = res[res["auc"] < 0.48]
    direct = res[res["auc"] > 0.52]
    print(f"\n📌 جمع‌بندی ({args.interval}):")
    print(f"   وارونه (AUC<0.48): {len(inverted)} ویژگی")
    for _, r in inverted.iterrows():
        print(f"      🔄 {r['feature']}  (AUC={r['auc']:.3f})")
    print(f"   مستقیم (AUC>0.52): {len(direct)} ویژگی")
    for _, r in direct.iterrows():
        print(f"      ✅ {r['feature']}  (AUC={r['auc']:.3f})")
    print("=" * 80)

    if len(inverted) > 0:
        print("\n💡 پیشنهاد: برای ویژگی‌های «وارونه»، علامت تعریف آن‌ها را برگردانیم")
        print("   (مثلاً واگرایی مثبت→منفی) و دوباره sm-only را تست کنیم.")
    else:
        print("\n💡 هیچ ویژگی وارونه‌ای نیست؛ AUC پایین sm-only احتمالاً از")
        print("   ترکیب/نویز است نه جهت — باید مفهوم‌ها را بازطراحی کنیم.")


if __name__ == "__main__":
    main()
