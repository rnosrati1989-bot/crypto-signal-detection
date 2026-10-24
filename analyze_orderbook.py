"""
analyze_orderbook.py
--------------------
آزمون صادقانه: آیا ویژگی‌های Order Book جهت کندلِ بعدی را پیش‌بینی می‌کنند؟

داده‌ی Order Book (از orderbook_stream.py) با کندل‌های همزمان بایننس join
می‌شود. برای هر کندل، «آخرین» نمونه‌ی دفتر سفارش داخلِ آن کندل گرفته
می‌شود — یعنی وضعیت کتاب در لحظه‌ی بسته‌شدن کندل (بدون look-ahead) — و
سؤال می‌کنیم: آیا این وضعیت، بازده کندلِ بعدی را پیش‌بینی می‌کند؟

معیار: AUC تک‌متغیره برای هر ویژگی (مقدار خود ویژگی به‌عنوان امتیاز).
  AUC > 0.52 → سیگنال مستقیم | AUC < 0.48 → وارونه | بین → بی‌اطلاع/غیرخطی

نحوه‌ی اجرا:
    python analyze_orderbook.py                     # پیش‌فرض: کندل 5m
    python analyze_orderbook.py --kline-interval 15m
    python analyze_orderbook.py --min-bars 80

⚠️ به حداقل تعداد کندل نیاز دارد (پیش‌فرض ۴۰). اگر کمتر باشد فقط گزارش
   پیشرفت می‌دهد — داده هر ساعت ~۱۲ کندل 5m اضافه می‌شود.
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AnalyzeOB")

OB_DIR = "data_cache/orderbook"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
KLINES_COLS = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "trades", "taker_buy_base",
               "taker_buy_quote", "ignore"]

FEATURES = ["imbalance_1", "imbalance_5", "imbalance_10", "imbalance_20",
            "depth_ratio_20", "spread_pct"]


def _interval_rule(interval: str) -> str:
    """'5m' -> '5min' | '15m' -> '15min' | '1h' -> '1h'"""
    if interval.endswith("m"):
        return f"{int(interval[:-1])}min"
    if interval.endswith("h"):
        return f"{int(interval[:-1])}h"
    raise ValueError(f"تایم‌فریم نامعتبر: {interval}")


def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int):
    """دریافت کندل‌های فیوچرز در بازه (عمومی — بدون کلید)."""
    rows = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": cur, "endTime": end_ms, "limit": 1000}
        resp = requests.get(KLINES_URL, params=params, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 1
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=KLINES_COLS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    # بایننس قیمت‌ها را رشته برمی‌گرداند — تبدیل به عدد
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df.drop_duplicates(subset="timestamp").sort_values("timestamp")


def _load_orderbook(symbol: str):
    path = os.path.join(OB_DIR, f"{symbol}_orderbook.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["timestamp"])


def _resample_to_bars(ob: pd.DataFrame, rule: str) -> pd.DataFrame:
    """آخرین نمونه در هر بازه‌ی کندل = وضعیت کتاب در لحظه‌ی بسته‌شدن کندل."""
    ob = ob.set_index("timestamp").sort_index()
    last = ob.resample(rule, label="left", closed="left").last()
    return last.dropna(subset=["mid_price"]).reset_index()


def _univariate_auc(x: pd.Series, y: pd.Series):
    from sklearn.metrics import roc_auc_score
    mask = x.notna() & y.notna()
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=int)
    if len(xv) < 30 or len(np.unique(yv)) < 2 or np.std(xv) == 0:
        return None, int(mask.sum())
    return roc_auc_score(yv, xv), int(mask.sum())


def _analyze_symbol(symbol, interval, rule, min_bars, up_threshold):
    ob = _load_orderbook(symbol)
    if ob is None or ob.empty:
        logger.warning(f"{symbol}: فایل Order Book پیدا نشد.")
        return None

    t0 = ob["timestamp"].min()
    t1 = ob["timestamp"].max()
    if t1 <= t0:
        return None

    end_ms = int(time.time() * 1000)
    start_ms = int((t0 - pd.Timedelta(hours=2)).timestamp() * 1000)  # کمی حاشیه

    try:
        klines = _fetch_klines(symbol, interval, start_ms, end_ms)
    except Exception as e:
        logger.error(f"{symbol}: خطا در دریافت کندل — {e}")
        return None

    bars = _resample_to_bars(ob, rule)
    merged = klines.merge(bars, on="timestamp", how="inner", suffixes=("", "_ob"))
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    if len(merged) < 5:
        return {"symbol": symbol, "bars": len(merged), "rows": None}

    merged["fwd_ret"] = merged["close"].shift(-1) / merged["close"] - 1.0
    merged["target"] = (merged["fwd_ret"] > up_threshold).astype(int)
    merged = merged.dropna(subset=["fwd_ret"])
    return {"symbol": symbol, "bars": len(merged), "rows": merged}


def main():
    parser = argparse.ArgumentParser(description="آزمون صادقانه‌ی سیگنال Order Book")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--kline-interval", default="5m",
                        choices=["1m", "5m", "15m", "1h"])
    parser.add_argument("--min-bars", type=int, default=40,
                        help="حداقل کندل برای محاسبه‌ی AUC")
    parser.add_argument("--up-threshold", type=float, default=0.0005,
                        help="آستانه‌ی صعود (0.0005 = 0.05٪)")
    args = parser.parse_args()

    rule = _interval_rule(args.kline_interval)
    print("\n" + "=" * 80)
    print("📖 آزمون صادقانه: آیا Order Book جهت کندل بعدی را پیش‌بینی می‌کند؟")
    print("=" * 80)
    print(f"• کندل: {args.kline_interval} | آستانه‌ی صعود: {args.up_threshold*100:.2f}% "
          f"| حداقل کندل: {args.min_bars}")
    print("-" * 80)

    enough_total = 0
    for symbol in [s.upper() for s in args.symbols]:
        res = _analyze_symbol(symbol, args.kline_interval, rule,
                              args.min_bars, args.up_threshold)
        if res is None:
            continue
        n = res["bars"]
        print(f"\n📌 {symbol}: {n} کندلِ هم‌پوشان")
        if res["rows"] is None or n < args.min_bars:
            print(f"   ⏳ داده کافی نیست (حداقل {args.min_bars} لازم). "
                  f"~{args.min_bars - n} کندل دیگر مانده "
                  f"(≈ {(args.min_bars - n) * (int(args.kline_interval[:-1]) * (60 if args.kline_interval.endswith('h') else 1))} دقیقه جمع‌آوری).")
            continue

        df = res["rows"]
        enough_total += 1
        base_rate = df["target"].mean()
        print(f"   نرخ پایه‌ی صعود: {base_rate*100:.1f}%")
        print(f"   {'ویژگی':<18} {'AUC':>7} {'جهت':>10} {'n':>5}")
        print("   " + "-" * 46)
        for feat in FEATURES:
            auc, cnt = _univariate_auc(df[feat], df["target"])
            if auc is None:
                print(f"   {feat:<18} {'—':>7} {'کم‌داده':>10} {cnt:>5}")
                continue
            direction = "مستقیم ✅" if auc > 0.52 else ("وارونه 🔄" if auc < 0.48 else "خنثی")
            print(f"   {feat:<18} {auc:>7.4f} {direction:>10} {cnt:>5}")

    print("\n" + "=" * 80)
    if enough_total == 0:
        print("⏳ هنوز داده‌ی کافی نیست. کلکتور را روشن نگه دار؛ هر ساعت "
              "~۱۲ کندل 5m اضافه می‌شود و همین اسکریپت را دوباره اجرا کن.")
    else:
        print("📊 نتیجه‌ی فعلی را با احتیاط بخوان: داده‌ی کم = نوسان زیاد در AUC.")
    print("=" * 80)


if __name__ == "__main__":
    main()
