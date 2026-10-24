"""
build_dataset.py
----------------
ساخت دیتاست نهایی ویژگی‌ها + برچسب‌ها برای آموزش مدل.

از فایل‌های CSV خام در data_cache می‌خواند، ویژگی‌ها و برچسب می‌سازد
و خروجی آماده‌ی آموزش را ذخیره می‌کند.

نحوه‌ی اجرا (از داخل پوشه‌ی پروژه):
    python build_dataset.py --symbols BTCUSDT ETHUSDT SOLUSDT --interval 15m --horizons 4 16 96

خروجی:
    - یک فایل CSV برای هر نماد در data_cache (مثل BTCUSDT_15m_features.csv)
    - خلاصه‌ی آماری: تعداد نمونه، تعداد ویژگی، توازن کلاس‌ها
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from features import (
    build_features, add_targets, get_feature_columns,
    attach_btc_context, load_btc_context, load_liquidation_candles,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BuildDataset")

CACHE_DIR = "data_cache"

# مخزن لیکوییدیشن (خروجی liquidations_collector.py) — از API قابل بازسازی نیست
LIQ_DIR = os.path.join("data_cache", "liquidations")

# کشِ context بیت‌کوین — برای هر تایم‌فریم یک‌بار ساخته و بین نمادها بازاستفاده می‌شود
_BTC_CONTEXT_CACHE = {}

# ستون‌هایی که برای داشتن یک نمونه‌ی معتبر «ضروری» هستند (همه فقط NaN اولیه دارند)
ESSENTIAL_FEATURES = [
    "ret_1", "RSI", "ema9_dist", "ATR", "range_pct",
    "funding_rate", "taker_buy_ratio",
]


def process_symbol(symbol: str, interval: str, horizons: list,
                   up_threshold: float, down_threshold: float,
                   include_smart_money: bool = False,
                   include_chart_patterns: bool = False,
                   include_scores: bool = False,
                   include_absolute_levels: bool = False,
                   include_liquidations: bool = False,
                   include_btc_context: bool = True,
                   tail_days: int = 0,
                   max_nan_share: float = 0.95,
                   include_elliott: bool = False):
    raw_path = os.path.join(CACHE_DIR, f"{symbol}_{interval}.csv")
    if not os.path.exists(raw_path):
        logger.error(f"فایل خام پیدا نشد: {raw_path} — اول fetch_futures_data.py را اجرا کنید.")
        return None

    df = pd.read_csv(raw_path, index_col="timestamp", parse_dates=True)
    logger.info(f"{symbol}: {len(df)} ردیف خام خوانده شد.")

    # برش پنجره‌ی اخیر (اختیاری) — برای وقتی که فقط روی بازه‌ای کار می‌کنیم که
    # داده‌ی Open Interest/لیکوییدیشن در آن موجود است (وگرنه این ستون‌ها
    # تقریباً ۱۰۰٪ NaN می‌شوند و خودکار حذف می‌شوند).
    if tail_days and int(tail_days) > 0:
        cutoff = df.index[-1] - pd.Timedelta(days=int(tail_days))
        before_rows = len(df)
        df = df[df.index >= cutoff]
        logger.info(f"{symbol}: برش پنجره‌ی اخیر ({int(tail_days)} روز): "
                    f"{before_rows} → {len(df)} ردیف")

    # لیکوییدیشن: چسباندن ستون‌های دلاریِ هر کندل از مخزن محلی (اگر خواسته شده)
    # ⚠️ این داده از API قابل بازسازی نیست و فقط از لحظه‌ی اجرای
    #    liquidations_collector.py وجود دارد → همراه --tail-days استفاده شود.
    if include_liquidations:
        liq = load_liquidation_candles(symbol, df.index, interval, LIQ_DIR)
        if liq.empty:
            logger.warning(f"{symbol}: مخزن لیکوییدیشن خالی/نبود — بلوک لیکوییدیشن "
                           "ساخته نمی‌شود (liquidations_collector.py را اجرا کنید).")
        else:
            covered = int((liq.sum(axis=1) > 0).sum())
            logger.info(f"{symbol}: لیکوییدیشن روی {covered} از {len(df)} کندل پیوست شد.")
            df = df.join(liq)

    # ساخت ویژگی‌ها
    df = build_features(df,
                        include_smart_money=include_smart_money,
                        include_chart_patterns=include_chart_patterns,
                        include_scores=include_scores,
                        include_absolute_levels=include_absolute_levels,
                        include_liquidations=include_liquidations,
                        include_elliott=include_elliott)

    # context بیت‌کوین (ویژگی‌های متقاطع) — روی همه‌ی نمادها از جمله خود بیت‌کوین
    # تا شکل دیتاست همه‌ی نمادها یکسان بماند (ضد train/serve skew).
    if include_btc_context:
        if interval not in _BTC_CONTEXT_CACHE:
            _BTC_CONTEXT_CACHE[interval] = load_btc_context(interval, CACHE_DIR)
        df = attach_btc_context(df, _BTC_CONTEXT_CACHE[interval])

    # ساخت برچسب‌ها برای همه‌ی افق‌های درخواستی (در یک دیتاست)
    for h in horizons:
        df = add_targets(df, horizon=h,
                         up_threshold=up_threshold, down_threshold=down_threshold)

    feature_cols = get_feature_columns(df)

    # --- ۱. گزارش ستون‌های دارای NaN (برای عیب‌یابی) ---
    nan_counts = df[feature_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if not nan_counts.empty:
        print(f"\n⚠️ ستون‌های دارای NaN ({symbol}) — {len(nan_counts)} ستون:")
        for col, cnt in nan_counts.head(10).items():
            pct = cnt / len(df) * 100
            print(f"    {col}: {cnt} ردیف ({pct:.1f}%)")

    # --- ۲. حذف خودکار ستون‌های تقریباً خالی (بیش از max_nan_share نان) ---
    bad_cols = list(nan_counts[nan_counts / len(df) >= float(max_nan_share)].index)
    if bad_cols:
        print(f"🗑️ حذف ستون‌های تقریباً خالی ({symbol}): {bad_cols}")
        df = df.drop(columns=bad_cols)
        feature_cols = get_feature_columns(df)

    # --- ۳. حذف ردیف‌های دارای NaN در ستون‌های ضروری یا هر برچسب ---
    before = len(df)
    essential_present = [c for c in ESSENTIAL_FEATURES if c in df.columns]
    label_cols = [f"fwd_ret_{h}" for h in horizons]
    df_clean = df.dropna(subset=essential_present + label_cols)
    dropped = before - len(df_clean)
    logger.info(f"{symbol}: {before} ردیف -> {len(df_clean)} ردیف "
                f"(حذف {dropped} ردیف دارای NaN)")

    if df_clean.empty:
        logger.warning(f"{symbol}: دیتاست خالی شد.")
        return None

    out_path = os.path.join(CACHE_DIR, f"{symbol}_{interval}_features.csv")
    df_clean.to_csv(out_path, encoding="utf-8")
    logger.info(f"ذخیره شد: {out_path}")

    # --- متادیتای دیتاست (ضد train/serve skew) ---
    # فلگ‌های ساخت ویژگی را کنار فایل ذخیره می‌کنیم تا آموزش و اجرای زنده
    # دقیقاً همان مجموعه ستون را بسازند (باگ قبلی: 15m بدون فلگ، 1h با فلگ).
    sidecar = {
        "symbol": symbol,
        "interval": interval,
        "horizons": list(horizons),
        "up_threshold": float(up_threshold),
        "down_threshold": float(down_threshold),
        "feature_flags": {
            "include_smart_money": bool(include_smart_money),
            "include_chart_patterns": bool(include_chart_patterns),
            "include_scores": bool(include_scores),
            "include_absolute_levels": bool(include_absolute_levels),
            "include_elliott": bool(include_elliott),
            "include_liquidations": bool(include_liquidations),
            "include_btc_context": bool(include_btc_context),
        },
        "n_rows": int(len(df_clean)),
        "n_features": int(len(feature_cols)),
        "start": str(df_clean.index[0]),
        "end": str(df_clean.index[-1]),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = os.path.join(CACHE_DIR, f"{symbol}_{interval}_features_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    logger.info(f"متادیتای دیتاست ذخیره شد: {meta_path}")

    # --- گزارش خلاصه (بر اساس اولین افق) ---
    n = len(df_clean)
    h0 = horizons[0]
    up_col = f"target_up_{h0}"
    down_col = f"target_down_{h0}"
    up_count = int(df_clean[up_col].sum())
    down_count = int(df_clean[down_col].sum())
    hold_count = n - up_count - down_count

    print("\n" + "=" * 70)
    print(f"🧠 دیتاست ویژگی‌ها — {symbol} ({interval})")
    print("=" * 70)
    print(f"• نمونه‌های آموزشی:      {n}")
    print(f"• تعداد ویژگی:          {len(feature_cols)}")
    print(f"• افق‌های برچسب:        {horizons}")
    print(f"• بازه‌ی زمانی:          {df_clean.index[0]} تا {df_clean.index[-1]}")
    print(f"• برچسب صعود (افق {h0}): {up_count} ({up_count/n*100:.1f}%)")
    print(f"• برچسب نزول (افق {h0}): {down_count} ({down_count/n*100:.1f}%)")
    print(f"• برچسب خنثی (افق {h0}): {hold_count} ({hold_count/n*100:.1f}%)")
    print("=" * 70 + "\n")

    return df_clean


def main():
    parser = argparse.ArgumentParser(description="ساخت دیتاست ویژگی‌ها برای آموزش مدل")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--horizons", nargs="+", type=int, default=[4],
                        help="افق‌های پیش‌بینی به تعداد کندل (مثلاً 4 16 96)")
    parser.add_argument("--up-threshold", type=float, default=0.002,
                        help="آستانه‌ی صعود (0.002 = 0.2٪)")
    parser.add_argument("--include-smart-money", action="store_true",
                        help="افزودن ویژگی‌های Smart Money (پیش‌فرض: خاموش — طبق نتیجه‌ی آزمایش edge اضافه نمی‌کنند)")
    parser.add_argument("--include-chart-patterns", action="store_true",
                        help="افزودن الگوهای نموداری (Double Top/Bottom، H&S، مثلث، Flag/Pennant، Gartley)")
    parser.add_argument("--include-scores", action="store_true",
                        help="افزودن لایه‌ی رژیم/اسکورینگ (MarketRegime + RegimeFilter + SignalScorer)")
    parser.add_argument("--include-absolute-levels", action="store_true",
                        help="نگه‌داشتن سطوح مطلق قیمت (EMA/BB/KC/DC/VWAP/OBV/ایچیموکو/PSAR/SuperTrend)")
    parser.add_argument("--down-threshold", type=float, default=0.002,
                        help="آستانه‌ی نزول (0.002 = 0.2٪)")
    parser.add_argument("--include-liquidations", action="store_true",
                        help="افزودن ویژگی‌های لیکوییدیشن (نیازمند ستون‌های خام liq_* در داده‌ی خام)")
    parser.add_argument("--btc-context", dest="btc_context", action="store_true",
                        default=True,
                        help="افزودن context بیت‌کوین به همه‌ی نمادها (پیش‌فرض: روشن)")
    parser.add_argument("--no-btc-context", dest="btc_context", action="store_false",
                        help="خاموش‌کردن context بیت‌کوین")
    parser.add_argument("--tail-days", type=int, default=0,
                        help="فقط N روز آخر داده را بساز (0 = کل تاریخچه) — برای وقتی که "
                             "می‌خواهیم داده‌ی Open Interest/لیکوییدیشن موجود باشد")
    parser.add_argument("--max-nan-share", type=float, default=0.95,
                        help="آستانه‌ی حذف خودکار ستون‌های خالی (پیش‌فرض 0.95 = ۹۵٪)")
    parser.add_argument("--include-elliott", action="store_true",
                        help="فاز ۲: افزودن بلوک ساختار موج الیوت (پیوت تأییدشده + "
                             "عمق فیبوناچی + سه قاعده‌ی سخت) — برای مقایسه‌ی A/B")
    args = parser.parse_args()

    for symbol in args.symbols:
        process_symbol(symbol.upper(), args.interval, args.horizons,
                       args.up_threshold, args.down_threshold,
                       args.include_smart_money,
                       args.include_chart_patterns,
                       args.include_scores,
                       args.include_absolute_levels,
                       args.include_liquidations,
                       args.btc_context,
                       args.tail_days,
                       args.max_nan_share,
                       include_elliott=args.include_elliott)

    print("✅ ساخت دیتاست تمام شد.")


if __name__ == "__main__":
    main()
