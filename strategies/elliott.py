"""
elliott.py
----------
فاز ۱ — ساختار موج (Elliott) به‌شکل عینی، تکرارپذیر و بدون نگاه به آینده.

فلسفه:
  «شمارش موج» برچسبِ حقیقت ندارد (دو تحلیلگر، دو شمارش). پس اینجا هیچ شمارشی
  تولید نمی‌شود؛ فقط **ویژگی‌های ساختاری** از پیوت‌های تأییدشده ساخته می‌شود:
  جهت/دامنه/طولِ پا، عمق اصلاح فیبوناچی، و سه قاعده‌ی سختِ موج انگیزشی.

ضد نگاه-به-آینده (مهم‌ترین بخش):
  هر پیوت فقط وقتی «تأیید» می‌شود که قیمت به اندازه‌ی k×ATR از آن برگردد
  (confirm). ویژگی هر کندل t فقط از پیوت‌هایی ساخته می‌شود که confirm <= t.
  تابع verify_no_lookahead() همین را با بازمحاسبه‌ی پیشوندی تصادفی تست می‌کند.

خودآزمون (آفلاین، بدون نیاز به شبکه):
    venv\\Scripts\\python.exe strategies\\elliott.py --file data_cache\\BTCUSDT_15m.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

# --- ستون‌هایی که این ماژول به دیتاست اضافه می‌کند (فاز ۲) ------------------
ELLIOTT_FEATURES = [
    "ew_leg_dir",        # جهت آخرین پا (1=صعودی، -1=نزولی)
    "ew_leg_atr",        # دامنه‌ی پا بر حسب ATR
    "ew_leg_bars",       # طول زمانی پا (کندل)
    "ew_retr_depth",     # عمق اصلاح فعلی نسبت به همان پا
    "ew_fib_zone",       # کد ناحیه‌ی فیبوناچی (0..5)
    "ew_leg_age",        # سن پای فعلی / میانه‌ی طول پاهای قبلی
    "ew_impulse_ok",     # 1 = سه قاعده‌ی سخت نقض نشده (NaN = داده کافی نیست)
    "ew_w4_overlap",     # 1 = موج ۴ وارد قلمرو موج ۱ شده (نقض قاعده)
    "ew_w3_shortest",    # 1 = موج ۳ کوتاه‌ترین موج انگیزشی است (نقض)
]

FIB_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.786)
DEFAULT_K = 2.0            # آستانه‌ی برگشت بر حسب ATR
DEFAULT_ATR_PERIOD = 14
MAX_LEG_MEMORY = 8         # برای میانگین طول پاها


def _atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> np.ndarray:
    """ATR ساده (اگر ستون ATR در داده باشد، همان استفاده می‌شود)."""
    if "ATR" in df.columns:
        return df["ATR"].astype(float).to_numpy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean().to_numpy()


def find_pivots(high, low, atr, k: float = DEFAULT_K) -> list:
    """
    زیگزاگ ATR-محور با «تأیید برگشتی».

    خروجی: لیستی از dict با کلیدهای
        idx     : کندلِ خودِ پیوت
        price   : قیمت پیوت
        kind    : +1 سقف، -1 کف
        confirm : کندلی که در آن پیوت تأیید شد (همیشه >= idx)

    ⚠️ استفاده از idx به‌تنهایی = نشتی اطلاعات. همیشه باید از confirm استفاده شود.
    """
    n = len(high)
    pivots = []
    if n < 3:
        return pivots
    thr = np.maximum(np.asarray(atr, dtype=float) * float(k), 1e-12)

    up = True                       # دنبال سقف می‌گردیم (بعد از یک کف)
    hi_i, hi_p = 0, float(high[0])
    lo_i, lo_p = 0, float(low[0])
    for i in range(1, n):
        if up:
            if high[i] > hi_p:
                hi_i, hi_p = i, float(high[i])
            if hi_p - float(low[i]) >= thr[i]:      # برگشت کافی → سقف تأیید شد
                pivots.append({"idx": hi_i, "price": hi_p, "kind": 1, "confirm": i})
                up = False
                lo_i, lo_p = i, float(low[i])
        else:
            if low[i] < lo_p:
                lo_i, lo_p = i, float(low[i])
            if float(high[i]) - lo_p >= thr[i]:     # برگشت کافی → کف تأیید شد
                pivots.append({"idx": lo_i, "price": lo_p, "kind": -1, "confirm": i})
                up = True
                hi_i, hi_p = i, float(high[i])
    return pivots


def _fib_zone(depth) -> float:
    """کد ناحیه‌ی فیبوناچی برای عمق اصلاح: 0=<0.236 ... 5=>=0.786."""
    if depth is None or not np.isfinite(depth):
        return np.nan
    d = float(depth)
    for z, lvl in enumerate(FIB_LEVELS):
        if d < lvl:
            return float(z)
    return float(len(FIB_LEVELS))


def _rule_flags(piv: list):
    """
    سه قاعده‌ی سخت الیوت روی ۶ پیوت آخر (۵ پا).
    خروجی: (impulse_ok, w4_overlap, w3_shortest) — و NaN اگر داده کافی نباشد.
    """
    if len(piv) < 6:
        return np.nan, np.nan, np.nan
    p = piv[-6:]
    px = [x["price"] for x in p]
    s = 1.0 if px[-1] > px[0] else -1.0
    d = [abs(px[i + 1] - px[i]) for i in range(5)]
    d1, d2, d3, d5 = d[0], d[1], d[2], d[4]

    rule1 = d2 < d1                       # موج ۲ بیشتر از ۱۰۰٪ موج ۱ را برنگرداند
    rule2 = d3 > min(d1, d5)              # موج ۳ کوتاه‌ترین نباشد
    overlap = (px[4] <= px[1]) if s > 0 else (px[4] >= px[1])   # موج ۴ و محدوده‌ی موج ۱
    ok = rule1 and rule2 and (not overlap)
    return (1.0 if ok else 0.0), (1.0 if overlap else 0.0), (0.0 if rule2 else 1.0)


def elliott_features(df: pd.DataFrame, k: float = DEFAULT_K,
                     atr_period: int = DEFAULT_ATR_PERIOD) -> pd.DataFrame:
    """
    محاسبه‌ی ۹ ویژگی ساختار موج برای هر کندل — فقط با اطلاعات تا همان کندل.
    خروجی: DataFrame هم‌ایندکس با df و ستون‌های ELLIOTT_FEATURES.
    """
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    atr = _atr(df, atr_period)
    pivots = find_pivots(high, low, atr, k=k)

    n = len(df)
    out = np.full((n, len(ELLIOTT_FEATURES)), np.nan, dtype=float)
    c = {name: j for j, name in enumerate(ELLIOTT_FEATURES)}

    seen = []            # پیوت‌های تأییدشده تا کندل جاری
    leg_bars = []        # طول پاهای تکمیل‌شده (برای میانگین)
    pi = 0
    for t in range(n):
        while pi < len(pivots) and pivots[pi]["confirm"] <= t:
            if seen:
                leg_bars.append(pivots[pi]["idx"] - seen[-1]["idx"])
                if len(leg_bars) > MAX_LEG_MEMORY:
                    leg_bars.pop(0)
            seen.append(pivots[pi])
            pi += 1
        if len(seen) < 2:
            continue

        last, prev = seen[-1], seen[-2]
        leg = last["price"] - prev["price"]
        if leg == 0.0:
            continue
        a_conf = float(atr[last["confirm"]]) or np.nan
        out[t, c["ew_leg_dir"]] = 1.0 if leg > 0 else -1.0
        out[t, c["ew_leg_atr"]] = abs(leg) / a_conf if np.isfinite(a_conf) and a_conf > 0 else np.nan
        out[t, c["ew_leg_bars"]] = float(last["idx"] - prev["idx"])

        # عمق اصلاح قیمت از آخرین پیوت نسبت به طول همان پا
        depth = (last["price"] - close[t]) / leg
        out[t, c["ew_retr_depth"]] = depth if abs(depth) <= 3.0 else np.nan
        out[t, c["ew_fib_zone"]] = _fib_zone(depth)

        # سن پا (چند کندل از پیوت گذشته) نسبت به میانگین طول پاهای قبلی
        if leg_bars:
            avg = float(np.mean(leg_bars))
            out[t, c["ew_leg_age"]] = (t - last["idx"]) / avg if avg > 0 else np.nan

        ok, overlap, w3s = _rule_flags(seen)
        out[t, c["ew_impulse_ok"]] = ok
        out[t, c["ew_w4_overlap"]] = overlap
        out[t, c["ew_w3_shortest"]] = w3s

    return pd.DataFrame(out, index=df.index, columns=ELLIOTT_FEATURES)


def verify_no_lookahead(df: pd.DataFrame, n_checks: int = 5, seed: int = 0,
                        k: float = DEFAULT_K,
                        atr_period: int = DEFAULT_ATR_PERIOD):
    """
    تست نشتی: ویژگی‌های ردیف t را یک‌بار روی کل داده و یک‌بار روی داده‌ی
    بریده‌شده تا خودِ t حساب می‌کنیم؛ باید دقیقاً یکی باشند.
    خروجی: (ok: bool, message: str)
    """
    if len(df) < 200:
        return True, "داده برای تست نشتی کافی نیست (حداقل ۲۰۰ کندل)."
    rng = np.random.default_rng(seed)
    full = elliott_features(df, k=k, atr_period=atr_period)
    worst = 0.0
    for t in rng.integers(150, len(df), size=int(n_checks)):
        part = elliott_features(df.iloc[:int(t) + 1], k=k, atr_period=atr_period)
        a = full.iloc[int(t)].to_numpy(dtype=float)
        b = part.iloc[-1].to_numpy(dtype=float)
        diff = np.abs(a - b)
        diff = diff[np.isfinite(diff)]
        if diff.size:
            worst = max(worst, float(diff.max()))
    ok = worst < 1e-9
    return ok, f"حداکثر اختلاف پیشوند/کل: {worst:.3e} → " + (
        "بدون نشتی ✅" if ok else "نشتی دارد ❌")


# ----------------------------------------------------------------------
# خودآزمون
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="فاز ۱: خودآزمون ساختار موج الیوت (آفلاین)")
    p.add_argument("--file", default=os.path.join("data_cache", "BTCUSDT_15m.csv"))
    p.add_argument("--k", type=float, default=DEFAULT_K,
                   help="آستانه‌ی برگشت بر حسب ATR (پیش‌فرض ۲.۰)")
    p.add_argument("--atr-period", type=int, default=DEFAULT_ATR_PERIOD)
    p.add_argument("--tail", type=int, default=20000,
                   help="فقط N کندل آخر برای سرعت (۰ = همه)")
    p.add_argument("--no-check", action="store_true", help="بدون تست نشتی")
    args = p.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"❌ فایل پیدا نشد: {args.file}")
    df = pd.read_csv(args.file, index_col=0, parse_dates=True)
    missing = {"high", "low", "close"} - set(df.columns)
    if missing:
        raise SystemExit(f"❌ ستون‌های لازم نیست: {missing}")
    if args.tail and len(df) > args.tail:
        df = df.tail(int(args.tail))

    print("=" * 72)
    print(f"فاز ۱ — خودآزمون الیوت | {os.path.basename(args.file)} | "
          f"k={args.k} | ATR={args.atr_period}")
    print("=" * 72)

    atr = _atr(df, args.atr_period)
    piv = find_pivots(df["high"].to_numpy(float), df["low"].to_numpy(float),
                      atr, k=args.k)
    print(f"• کندل‌ها: {len(df)}")
    print(f"• پیوت‌های تأییدشده: {len(piv)}")
    if len(piv) > 1:
        lag = np.array([x["confirm"] - x["idx"] for x in piv], dtype=float)
        gaps = np.diff([x["idx"] for x in piv])
        print(f"• تأخیر تأیید (کندل): میانگین {lag.mean():.1f} | "
              f"میانه {np.median(lag):.0f} | بیشینه {lag.max():.0f}")
        print(f"• فاصله‌ی پیوت‌ها: میانگین {gaps.mean():.1f} کندل | "
              f"میانه {np.median(gaps):.0f}")
        kinds = np.array([x["kind"] for x in piv])
        print(f"• سقف/کف: سقف {int((kinds > 0).sum())} | کف {int((kinds < 0).sum())}")

    feats = elliott_features(df, k=args.k, atr_period=args.atr_period)
    print("\n• پوشش هر ویژگی (درصد کندل‌های غیرتهی):")
    for c in ELLIOTT_FEATURES:
        cover = 100.0 * feats[c].notna().mean()
        print(f"    {c:<16} {cover:6.1f}٪")

    z = feats["ew_fib_zone"].value_counts().sort_index()
    print("\n• توزیع ناحیه‌ی فیبوناچی (0=<0.236 … 5=>=0.786):")
    for zone, cnt in z.items():
        print(f"    zone {int(zone)}: {int(cnt)}")

    for c in ("ew_impulse_ok", "ew_w4_overlap", "ew_w3_shortest"):
        s = feats[c].dropna()
        if len(s):
            print(f"• {c}: {100.0 * (s > 0.5).mean():.1f}٪ از "
                  f"{len(s)} کندل تعریف‌شده")

    if not args.no_check:
        print()
        ok, msg = verify_no_lookahead(df, k=args.k, atr_period=args.atr_period)
        print(f"• تست نشتی: {msg}")
        if not ok:
            raise SystemExit("❌ ویژگی‌ها نشتی دارند؛ قبل از آموزش حل شود.")
    print("=" * 72)


if __name__ == "__main__":
    main()
