"""
download_history.py
-------------------
ساخت/به‌روزرسانی مخزن تاریخی (Binance Vision) + بازنویسی فایل‌های خام
data_cache برای آموزش مدل‌ها روی چند سال داده.

اجرا (از داخل پوشه‌ی پروژه):
    venv\\Scripts\\python.exe download_history.py --symbols BTCUSDT ETHUSDT SOLUSDT --intervals 15m 1h

این اسکریپت:
  1) فایل‌های ماهانه‌ی klines + fundingRate از آرشیو رسمی Binance Vision را
     (از --from تا آخرین ماهِ کامل) دانلود و در data_cache/historical ذخیره
     می‌کند. اجرای دوباره فقط ماه/روزهای جاافتاده یا جدید را می‌گیرد
     (idempotent) — ماه‌هایی که فاندینگ‌شان جا مانده، خودکار بک‌فیل می‌شوند.
  2) Open Interest روزهای اخیر (metrics روزانه) را هم اضافه می‌کند.
  3) با --write-cache (پیش‌فرض)، data_cache/{SYMBOL}_{INTERVAL}.csv را از
     تاریخچه‌ی چندساله + دُمِ زنده‌ی API می‌سازد — دقیقاً همان قالب
     fetch_futures_data.py — تا build_dataset.py و train_production_models.py
     بلافاصله روی داده‌ی عمیق کار کنند.

نکته‌ی صادقانه: Open Interest کامل فقط برای روزهای اخیر در دسترس است
(API رسمی ~۳۰ روز و آرشیو metrics روزانه). برای ماه‌های قدیمی‌تر، ستون‌های
open_interest خالی (NaN) می‌مانند؛ LightGBM با NaN کار می‌کند و ستون‌های OI
در آن ردیف‌ها فقط بی‌اثر می‌شوند.

--clear-metrics: اگر قبلاً اجرا کرده‌اید و metrics دانلود نشده (روزها
نادرست «انجام‌شده» ثبت شده‌اند)، این فلگ روزها را پاک می‌کند تا دوباره
دانلود شوند. یک‌بار بعد از این فیکس کافی است.

--clear-funding: ماه‌های فاندینگِ ثبت‌شده را پاک می‌کند تا دوباره با ستون
صحیح (last_funding_rate) دانلود شوند — لازم بعد از این فیکس.
"""

import argparse
import logging
import os
import sys

# غیرفعال کردن پراکسی برای ارتباط محلی (سازگار با بقیه‌ی پروژه)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import pandas as pd

from config.settings import Config
from data.binance_futures_client import BinanceFuturesDataClient
from data.historical_store import (
    update_symbol, load_combined, coverage,
    clear_metrics_days, clear_funding_months,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DownloadHistory")

CACHE_DIR = "data_cache"


def write_cache(symbol: str, interval: str, live_tail: int) -> None:
    """تاریخچه (مخزن) + دُمِ زنده‌ی API → data_cache/{SYMBOL}_{INTERVAL}.csv"""
    history = load_combined(symbol, interval)
    if history.empty:
        logger.warning(f"مخزن تاریخی {symbol}/{interval} خالی است؛ فقط دُمِ زنده نوشته می‌شود.")

    tail = pd.DataFrame()
    try:
        client = BinanceFuturesDataClient()
        tail = client.get_futures_dataset(symbol, interval, limit=live_tail)
    except Exception as e:
        logger.error(f"دریافت دُمِ زنده برای {symbol}/{interval} ناموفق بود: {e}")

    parts = [p for p in (history, tail) if p is not None and not p.empty]
    if not parts:
        logger.error(f"هیچ داده‌ای برای {symbol}/{interval} موجود نیست.")
        return
    frame = pd.concat(parts)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, f"{symbol}_{interval}.csv")
    frame.to_csv(out_path, encoding="utf-8")
    logger.info(f"ذخیره شد: {out_path} ({len(frame)} ردیف، "
                f"{frame.index[0]} ← {frame.index[-1]})")


def main():
    parser = argparse.ArgumentParser(
        description="دانلود تاریخچه‌ی چندساله‌ی فیوچرز از Binance Vision + ساخت cache")
    parser.add_argument("--symbols", nargs="+", default=Config.DEFAULT_SYMBOLS)
    parser.add_argument("--intervals", nargs="+", default=["15m", "1h"])
    parser.add_argument("--from", dest="from_ym", default="2020-01",
                        help="اولین ماه برای دانلود (پیش‌فرض: 2020-01)")
    parser.add_argument("--metrics-days", type=int, default=120,
                        help="روزهای اخیرِ Open Interest (metrics) — 0 = خاموش")
    parser.add_argument("--clear-metrics", action="store_true",
                        help="پاک‌کردن روزهای ثبت‌شده‌ی metrics و دانلود دوباره "
                             "(بعد از فیکس قالب ستون‌ها یک‌بار لازم است)")
    parser.add_argument("--clear-funding", action="store_true",
                        help="پاک‌کردن ماه‌های ثبت‌شده‌ی فاندینگ و دانلود دوباره "
                             "(بعد از فیکس ستون last_funding_rate)")
    parser.add_argument("--live-tail", type=int, default=1500,
                        help="تعداد کندلِ دُمِ زنده‌ی API که روی تاریخچه سوار می‌شود")
    parser.add_argument("--skip-cache", action="store_true",
                        help="فقط مخزن را به‌روز کند و data_cache را بازنویسی نکند")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]
    total = {"klines": 0, "funding": 0, "metrics": 0}
    for symbol in symbols:
        for interval in args.intervals:
            print("\n" + "=" * 78)
            print(f"📦 به‌روزرسانی مخزن تاریخی: {symbol} ({interval})")
            print("=" * 78)
            if args.clear_metrics:
                clear_metrics_days(symbol, interval)
                print("   • روزهای metrics پاک شد — دوباره دانلود می‌شوند.")
            if args.clear_funding:
                clear_funding_months(symbol, interval)
                print("   • ماه‌های فاندینگ پاک شد — دوباره با ستون صحیح دانلود می‌شوند.")
            try:
                got = update_symbol(symbol, interval, from_ym=args.from_ym,
                                    metrics_days=args.metrics_days)
            except Exception as e:
                logger.error(f"به‌روزرسانی {symbol}/{interval} ناموفق بود: {e}")
                continue
            for k in total:
                total[k] += got.get(k, 0)

            if not args.skip_cache:
                write_cache(symbol, interval, args.live_tail)

            cov = coverage(symbol, interval)
            if cov["rows"]:
                print(f"   • پوشش: {cov['rows']} ردیف | "
                      f"{cov['first']} ← {cov['last']}")
                print(f"   • فاندینگ خالی: {cov['funding_na_pct']}٪ | "
                      f"OI خالی: {cov['oi_na_pct']}٪")
            else:
                print("   • مخزن خالی است (نماد/تایم‌فریم در آرشیو نیست؟)")

    print("\n" + "=" * 78)
    print(f"✅ جمع‌بندی دانلود این اجرا: کندل +{total['klines']} | "
          f"فاندینگ +{total['funding']} | OI +{total['metrics']}")
    print("=" * 78 + "\n")

    print("قدم بعدی (اختیاری، برای آموزش مدل روی تایم‌فریم/نماد جدید):")
    print("   1) venv\\Scripts\\python.exe build_dataset.py "
          f"--symbols {' '.join(symbols)} --interval <TF> --horizons 4 16 96")
    print("   2) venv\\Scripts\\python.exe train_production_models.py "
          f"--symbols {' '.join(symbols)} --interval <TF> --horizon 4")


if __name__ == "__main__":
    sys.exit(main())