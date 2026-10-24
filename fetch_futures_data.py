"""
fetch_futures_data.py
---------------------
دانلود داده‌ی فیوچرز (کندل + فاندینگ + Open Interest) و ذخیره‌ی محلی.

نحوه‌ی اجرا (از داخل پوشه‌ی پروژه):
    python fetch_futures_data.py --symbol BTCUSDT --interval 1h --limit 3000

خروجی:
    - یک فایل CSV در پوشه‌ی data_cache (مثلاً data_cache/BTCUSDT_1h.csv)
    - یک خلاصه‌ی کوتاه در ترمینال تا ببینید دقیقاً چه داده‌ای دارید

نکته: تاریخچه‌ی Open Interest در API عمومی بایننس فقط ~۳۰ روز است (محدودیت
مستند خود بایننس). بنابراین برای بازه‌های طولانی‌تر، ستون‌های open_interest
در ابتدای فایل NaN خواهند بود — این طبیعی است و در گزارش «مقادیر ازدست‌رفته»
به تفکیک ستون نشان داده می‌شود.
"""

import argparse
import logging
import os
import sys

# غیرفعال کردن پراکسی برای ارتباطات محلی (سازگار با بقیه‌ی پروژه)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import pandas as pd
from data.binance_futures_client import BinanceFuturesDataClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FetchFutures")


def main():
    parser = argparse.ArgumentParser(description="دانلود داده‌ی فیوچرز بایننس")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", help="مثل 15m, 1h, 4h, 1d")
    parser.add_argument("--limit", type=int, default=1000, help="تعداد کندل")
    args = parser.parse_args()

    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)

    logger.info(f"دریافت داده‌ی فیوچرز {args.symbol} ({args.interval}) — {args.limit} کندل ...")
    client = BinanceFuturesDataClient()
    df = client.get_futures_dataset(args.symbol, args.interval, limit=args.limit)

    if df.empty:
        logger.error("داده‌ای دریافت نشد. اتصال اینترنت/پروکسی را بررسی کنید.")
        sys.exit(1)

    out_path = os.path.join(cache_dir, f"{args.symbol}_{args.interval}.csv")
    df.to_csv(out_path, encoding="utf-8")
    logger.info(f"ذخیره شد: {out_path}")

    print("\n" + "=" * 70)
    print(f"📊 داده‌ی فیوچرز {args.symbol} ({args.interval})")
    print("=" * 70)
    print(f"• تعداد کندل:            {len(df)}")
    print(f"• بازه‌ی زمانی:          {df.index[0]} تا {df.index[-1]}")
    print(f"• ستون‌ها:               {list(df.columns)}")

    last = df.iloc[-1]
    print("-" * 70)
    print(f"• آخرین قیمت بسته‌شدن:   {last['close']:.2f}")
    if pd.notna(last["funding_rate"]):
        print(f"• آخرین فاندینگ:         {last['funding_rate_pct']:.4f}%")
    if pd.notna(last["open_interest"]):
        base_asset = args.symbol.upper().replace("USDT", "")
        print(f"• Open Interest:         {last['open_interest']:.0f} {base_asset} "
              f"(~${last['open_interest_value'] / 1e6:.1f}M)")
        # مقایسه با ~۲۴ کندل قبل (برای تایم‌فریم 1h یعنی ۲۴ ساعت قبل)
        lookback = min(25, len(df))
        oi_ago = df["open_interest"].iloc[-lookback]
        change = (last["open_interest"] / oi_ago - 1) * 100
        print(f"• تغییر OI نسبت به {lookback} کندل قبل: {change:+.2f}%")

    # گزارش دقیق مقادیر ازدست‌رفته به تفکیک ستون
    missing = int(df.isna().sum().sum())
    print(f"• تعداد مقادیر ازدست‌رفته: {missing}")
    miss_cols = df.isna().sum()
    miss_cols = miss_cols[miss_cols > 0]
    if not miss_cols.empty:
        print("  (به تفکیک ستون):")
        for col, cnt in miss_cols.items():
            pct = cnt / len(df) * 100
            print(f"    - {col}: {cnt} ردیف ({pct:.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
