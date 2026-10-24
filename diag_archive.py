"""
diag_archive.py
---------------
ابزار ریشه‌یابی خطاهای دانلود آرشیو Binance Vision.

مشکل‌هایی که این ابزار بررسی می‌کند:
  1) «کندل‌های ماه X قابل‌پردازش نبود» — مثلاً BTCUSDT/5m برای 2020-01..2021-12
     و SOLUSDT/15m برای 2022-03 به بعد.
  2) قالب/مقیاس ستون funding در فایل‌های آرشیو در برابر API رسمی.

اجرا (از داخل پوشه‌ی پروژه):
    venv\\Scripts\\python.exe diag_archive.py BTCUSDT 5m 2021-01
    venv\\Scripts\\python.exe diag_archive.py SOLUSDT 15m 2022-03
    venv\\Scripts\\python.exe diag_archive.py BTCUSDT 15m 2022-01 --funding --api-compare
"""

import argparse
import io
import sys
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

VISION_BASE = "https://data.binance.vision/data/futures/um"


def fetch(url: str):
    r = requests.get(url, timeout=120)
    print(f"  GET {url}")
    print(f"  -> HTTP {r.status_code} | {len(r.content):,} bytes | "
          f"content-type: {r.headers.get('content-type', '?')}")
    return r


def inspect_zip(label: str, raw: bytes):
    print(f"\n🔎 {label}")
    if len(raw) < 4 or raw[:2] != b"PK":
        print("  ⚠️ فایل ZIP نیست — ۴ بایت اول:", raw[:40])
        return
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for name in zf.namelist():
                csv_raw = zf.read(name)
                print(f"  داخل zip: {name} ({len(csv_raw):,} bytes)")
                head = csv_raw[:600].decode("utf-8", errors="replace")
                print("  --- ۳ خط اول خام ---")
                print("\n".join(head.splitlines()[:3]))
                df = pd.read_csv(io.BytesIO(csv_raw))
                print(f"  --- pandas: {len(df)} ردیف × {len(df.columns)} ستون ---")
                print("  ستون‌ها:", list(df.columns))
                if len(df):
                    print("  اولین ردیف:", df.iloc[0].to_dict())
    except Exception as e:
        print(f"  ❌ خطا در باز کردن zip: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("interval")
    ap.add_argument("ym")
    ap.add_argument("--funding", action="store_true",
                    help="فایل fundingRate همان ماه را هم بررسی کند")
    ap.add_argument("--api-compare", action="store_true",
                    help="مقایسه‌ی funding آرشیو با API رسمی (همان بازه)")
    args = ap.parse_args()
    s, iv, ym = args.symbol.upper(), args.interval, args.ym

    print("=" * 70)
    print(f"دیاگ آرشیو: {s} {iv} — {ym}")
    print("=" * 70)

    kurl = f"{VISION_BASE}/monthly/klines/{s}/{iv}/{s}-{iv}-{ym}.zip"
    r = fetch(kurl)
    if r.status_code == 200:
        inspect_zip("کندل‌ها", r.content)

    if args.funding:
        furl = (f"{VISION_BASE}/monthly/fundingRate/{s}/"
                f"{s}-fundingRate-{ym}.zip")
        r2 = fetch(furl)
        if r2.status_code == 200:
            inspect_zip("فاندینگ", r2.content)

    if args.api_compare:
        start = int(datetime(int(ym[:4]), int(ym[5:7]), 1,
                             tzinfo=timezone.utc).timestamp() * 1000)
        aurl = ("https://fapi.binance.com/fapi/v1/fundingRate"
                f"?symbol={s}&startTime={start}&limit=3")
        try:
            arr = requests.get(aurl, timeout=30).json()
            print("\n🔎 funding از API رسمی (۳ مقدار اول همان ماه):")
            for item in arr[:3]:
                print(" ", item)
        except Exception as e:
            print(f"  ⚠️ API ناموفق: {e}")

    print("\n✅ تمام شد.")


if __name__ == "__main__":
    sys.exit(main())