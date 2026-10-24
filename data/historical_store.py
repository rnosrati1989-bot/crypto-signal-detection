"""
historical_store.py
-------------------
مخزن محلی داده‌ی تاریخی فیوچرز بایننس (USDT-M) — روی آرشیو رسمی Binance Vision.

چرا این مخزن لازم است؟
  * API عمومی بایننس برای Open Interest فقط ~۳۰ روز تاریخچه می‌دهد (محدودیت
    مستند خود بایننس)؛ برای آموزش مدل روی تایم‌فریم‌های ریز (5m/1m) یا
    نمادهای بیشتر به ماه‌ها/سال‌ها داده‌ی پیوسته نیاز داریم.
  * Binance Vision (data.binance.vision) آرشیو رسمی و رایگان با عمق چندساله
    است: کندل و فاندینگ به‌صورت فایل ماهانه (zip)، و Open Interest (metrics)
    به‌صورت فایل روزانه. بدون ریت‌لیمیت API.

این ماژول فقط «دانلود/به‌روزرسانی تدریجی» را مدیریت می‌کند؛ خروجی ترکیبی
load_combined دقیقاً با همان قالب get_futures_dataset ساخته می‌شود تا بقیه‌ی
پایپ‌لاین (fetch_futures_data / build_dataset / train) بدون تغییر کار کند.

ساختار ذخیره:
    data_cache/historical/{SYMBOL}/{INTERVAL}/
        klines.csv        تمام ماه‌های دانلودشده (OHLCV + quote + taker_buy)
        funding.csv       نرخ فاندینگ ماهانه
        metrics.csv       Open Interest روزهای اخیر (اختیاری)
        meta.json         پوشش دانلود (ماه‌ها/روزها/بازه‌ها)
"""

import io
import json
import logging
import os
import time
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

VISION_BASE = "https://data.binance.vision/data/futures/um"
STORE_ROOT = os.path.join("data_cache", "historical")

REQUEST_TIMEOUT_SECONDS = 90   # فایل‌های ماهانه‌ی کندل می‌توانند چند مگابایت باشند
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5
REQUEST_DELAY_SECONDS = 0.2    # مکث کوتاه بین درخواست‌ها برای کاهش قطعی اتصال

# ستون‌های خروجی نهایی (همان قالب get_futures_dataset / fetch_futures_data)
COMBINED_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_volume",
    "funding_rate", "funding_rate_pct",
    "open_interest", "open_interest_value",
]

# یک Session مشترک (keep-alive) — به‌شدت تعداد خطاهای اتصال را کم می‌کند
_SESSION = requests.Session()


class ArchiveError(Exception):
    """خطای شبکه/سرور هنگام دانلود از آرشیو (متفاوت از 404=ناموجود)."""
    pass


# ----------------------------------------------------------------------
# کمکی‌های ذخیره‌سازی
# ----------------------------------------------------------------------
def _store_dir(symbol: str, interval: str) -> str:
    return os.path.join(STORE_ROOT, symbol.upper(), interval)


def _meta_path(symbol: str, interval: str) -> str:
    return os.path.join(_store_dir(symbol, interval), "meta.json")


def _load_meta(symbol: str, interval: str) -> dict:
    try:
        with open(_meta_path(symbol, interval), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"months": [], "funding_months": [], "metrics_days": []}


def _save_meta(symbol: str, interval: str, meta: dict) -> None:
    os.makedirs(_store_dir(symbol, interval), exist_ok=True)
    with open(_meta_path(symbol, interval), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def clear_metrics_days(symbol: str, interval: str) -> None:
    """پاک‌کردن روزهای ثبت‌شده‌ی metrics تا دوباره دانلود شوند."""
    symbol = symbol.upper()
    meta = _load_meta(symbol, interval)
    meta["metrics_days"] = []
    _save_meta(symbol, interval, meta)


def clear_funding_months(symbol: str, interval: str) -> None:
    """پاک‌کردن ماه‌های ثبت‌شده‌ی فاندینگ (بعد از فیکس ستون) تا دوباره دانلود شود."""
    symbol = symbol.upper()
    meta = _load_meta(symbol, interval)
    meta["funding_months"] = []
    _save_meta(symbol, interval, meta)


def _next_month(ym: str) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    m += 1
    if m > 12:
        m = 1
        y += 1
    return f"{y:04d}-{m:02d}"


# ----------------------------------------------------------------------
# دانلود
# ----------------------------------------------------------------------
def _get_bytes(url: str):
    """
    دانلود بایت‌های یک فایل با retry/backoff.

    برمی‌گرداند:
      * بایت‌های فایل در صورت موفقیت
      * None فقط اگر فایل واقعاً وجود نداشته باشد (HTTP 404)
      * در صورت خطای شبکه/سرور پس از همه‌ی تلاش‌ها → ArchiveError پرتاب می‌شود
        (صداکننده باید بداند «ناموجود» نیست، «مشکل موقت» است)
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _SESSION.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                return None
            # ریت‌لیمیت یا خطای موقت سرور → تلاش مجدد
            wait = RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                f"خطای {resp.status_code} ({url}) — تلاش مجدد پس از {wait}s")
            time.sleep(wait)
            continue
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    f"خطای اتصال ({url}): {str(e)[:120]} — تلاش مجدد پس از {wait}s")
                time.sleep(wait)
                continue
            raise ArchiveError(f"عدم امکان دانلود {url}: {str(e)[:120]}")
    raise ArchiveError(f"عدم امکان دانلود {url} پس از {MAX_RETRIES} تلاش")


def _read_zip_csv(url: str):
    """
    دانلود یک zip آرشیو و خواندن CSV داخل آن.

    برمی‌گرداند:
      * DataFrame خام CSV در صورت موفقیت
      * None اگر فایل 404 باشد (ناموجود)
      * در خطای شبکه → ArchiveError (از _get_bytes منتشر می‌شود)
    اگر zip باز نشود، warning می‌دهد و DataFrame خالی برمی‌گرداند.
    """
    raw = _get_bytes(url)
    if raw is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                return pd.read_csv(f)
    except Exception as e:
        logger.warning(f"خطا در باز کردن zip {url}: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# نرمال‌سازی قالب‌های مختلف CSV آرشیو
# (انعطاف‌پذیر: نام ستون‌ها را بر اساس الگو تشخیص می‌دهد تا به تغییرات
#  هدر بایننس وابسته نباشد؛ فایل‌های قدیمیِ بدون هدر هم پشتیبانی می‌شوند.)
# ----------------------------------------------------------------------
def _is_headerless(df: pd.DataFrame) -> bool:
    # فایل‌های قدیمی آرشیو بدون هدر: پانداس نام ستون‌ها را رشته‌ی عددی نگه می‌دارد
    if df.empty:
        return True
    c0 = df.columns[0]
    if isinstance(c0, (int, np.integer)):
        return True
    return isinstance(c0, str) and c0.replace(".", "", 1).isdigit()


def _normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if _is_headerless(df):
        names = ["timestamp", "open", "high", "low", "close", "volume",
                 "close_time", "quote_volume", "count",
                 "taker_buy_volume", "taker_buy_quote", "ignore"]
        df = df.rename(columns={c: names[i] for i, c in enumerate(df.columns)
                                if i < len(names)})
    cols = {str(c).lower(): str(c) for c in df.columns}
    keep = ["timestamp", "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_volume"]
    out = pd.DataFrame()
    for k in keep:
        col = (cols.get(k) or next((v for kk, v in cols.items()
                                    if kk in ("open_time", "quote_asset_volume",
                                              "taker_buy_base_asset_volume")
                                    and k in kk or kk == k), None))
        if col is None and k == "timestamp":
            col = next((v for kk, v in cols.items() if "time" in kk), None)
        out[k] = df[col].values if col is not None else float("nan")
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms", errors="coerce")
    for c in keep[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return (out.dropna(subset=["timestamp"])
              .set_index("timestamp").sort_index())


def _normalize_funding(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if _is_headerless(df):
        names = ["timestamp", "funding_rate", "mark_price", "extra"]
        df = df.rename(columns={c: names[i] for i, c in enumerate(df.columns)
                                if i < len(names)})
    cols = {str(c).lower(): str(c) for c in df.columns}

    tcol = (cols.get("fundingtime") or cols.get("timestamp")
            or next((v for k, v in cols.items() if "time" in k), None))
    rcol = (next((v for k, v in cols.items()
                 if "rate" in k), None)
            or next((v for k, v in cols.items()
                     if "fund" in k), None))
    if tcol is None or rcol is None:
        logger.warning(f"⚠️ ستون‌های فاندینگ شناسایی نشد: {list(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[tcol], unit="ms", errors="coerce"),
        "funding_rate": pd.to_numeric(df[rcol], errors="coerce"),
    })
    return (out.dropna(subset=["timestamp"])
              .set_index("timestamp").sort_index())


def _normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if _is_headerless(df):
        names = ["timestamp", "symbol", "sumOpenInterest", "sumOpenInterestValue",
                 "countTakerLong", "countTakerShort", "longAccount",
                 "shortAccount", "longShortRatio"]
        df = df.rename(columns={c: names[i] for i, c in enumerate(df.columns)
                                if i < len(names)})
    cols = {str(c).lower(): str(c) for c in df.columns}

    tcol = cols.get("timestamp") or next(
        (v for k, v in cols.items() if "time" in k), None)
    oi_col = (cols.get("sum_open_interest") or cols.get("sumopeninterest"))
    oiv_col = (cols.get("sum_open_interest_value") or cols.get("sumopeninterestvalue"))
    if oi_col is None or oiv_col is None:
        for k, v in cols.items():
            kk = k.replace("_", "")
            if "openinterest" in kk and "value" not in kk and oi_col is None:
                oi_col = v
            elif "openinterest" in kk and "value" in kk and oiv_col is None:
                oiv_col = v
    if tcol is None or oi_col is None or oiv_col is None:
        logger.warning(f"⚠️ ستون‌های metrics شناسایی نشد: {list(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[tcol], unit="ms", errors="coerce"),
        "open_interest": pd.to_numeric(df[oi_col], errors="coerce"),
        "open_interest_value": pd.to_numeric(df[oiv_col], errors="coerce"),
    })
    return (out.dropna(subset=["timestamp"])
              .set_index("timestamp").sort_index())


def _append_and_dedupe(path: str, new_df: pd.DataFrame) -> int:
    """
    افزودن داده به CSV محلی با حذف زمان‌های تکراری.

    برمی‌گرداند: تعداد ردیف‌های «جدید» (ردیف‌هایی که قبلاً در فایل نبودند).
    """
    if new_df is None or new_df.empty:
        return 0
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        n_old = len(old)
        frame = pd.concat([old, new_df])
    else:
        n_old = 0
        frame = new_df
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, encoding="utf-8")
    return max(0, len(frame) - n_old)


# ----------------------------------------------------------------------
# به‌روزرسانی تدریجی
# ----------------------------------------------------------------------
def update_symbol(symbol: str, interval: str, from_ym: str = "2020-01",
                  metrics_days: int = 120) -> dict:
    """
    دانلود/به‌روزرسانی داده‌ی تاریخی یک نماد:
      1) فایل‌های ماهانه‌ی klines + fundingRate از آرشیو Vision (از from_ym
         تا آخرین ماهِ کامل)
      2) فایل‌های روزانه‌ی metrics (Open Interest) برای `metrics_days` روز آخر

    idempotent است: ماه/روزهای موفق در meta.json ذخیره می‌شوند و اجرای دوباره
    فقط داده‌ی جاافتاده/جدید را می‌گیرد.
    خطای شبکه ماه را «انجام‌شده» ثبت نمی‌کند تا اجرای بعدی دوباره تلاش کند؛
    404 واقعی «ناموجود» است و بی‌صداتر رد می‌شود.
    """
    symbol = symbol.upper()
    d = _store_dir(symbol, interval)
    os.makedirs(d, exist_ok=True)
    meta = _load_meta(symbol, interval)
    done_months = set(meta.get("months", []))
    done_funding = set(meta.get("funding_months", []))
    done_metrics = set(meta.get("metrics_days", []))

    now_utc = datetime.now(timezone.utc)
    current_ym = now_utc.strftime("%Y-%m")
    got = {"klines": 0, "funding": 0, "metrics": 0}

    # --- کندل + فاندینگ: ماه به ماه ---
    ym = from_ym
    while ym < current_ym:
        month_done = ym in done_months
        funding_done = ym in done_funding
        if month_done and funding_done:
            ym = _next_month(ym)
            continue

        kurl = (f"{VISION_BASE}/monthly/klines/{symbol}/{interval}/"
                f"{symbol}-{interval}-{ym}.zip")
        try:
            kraw = _read_zip_csv(kurl)
        except ArchiveError as e:
            logger.error(f"❌ {symbol}/{interval} ماه {ym}: {e}")
            ym = _next_month(ym)
            continue  # موقت است؛ ماه را done نمی‌کنیم تا اجرای بعدی تلاش کند

        if kraw is None:
            # فایل وجود ندارد: اگر هنوز هیچ ماهی نداریم یعنی قبل از لیستینگ
            # هستیم و رد می‌شویم؛ در میانه‌ی پوشش هم یعنی پایان آرشیو.
            if not done_months:
                ym = _next_month(ym)
                continue
            ym = _next_month(ym)
            continue  # ماه در آرشیو نیست (قبل از لیستینگ نماد) — ماه‌های بعدی را هم باید پردازش کرد

        if not month_done:
            kdf = _normalize_klines(kraw)
            if kdf.empty:
                logger.error(f"⚠️ کندل‌های ماه {ym} قابل‌پردازش نبود.")
                ym = _next_month(ym)
                continue
            got["klines"] += _append_and_dedupe(os.path.join(d, "klines.csv"), kdf)
            done_months.add(ym)

        # --- فاندینگ همان ماه (به‌صورت مستقل — اگر قبلاً جا مانده، بک‌فیل می‌شود) ---
        if not funding_done:
            furl = (f"{VISION_BASE}/monthly/fundingRate/{symbol}/"
                    f"{symbol}-fundingRate-{ym}.zip")
            try:
                fraw = _read_zip_csv(furl)
            except ArchiveError as e:
                logger.error(f"❌ فاندینگ {symbol} ماه {ym}: {e}")
                ym = _next_month(ym)
                continue  # done نمی‌شود → اجرای بعدی فقط فاندینگ را می‌گیرد
            if fraw is not None:
                fdf = _normalize_funding(fraw)
                got["funding"] += _append_and_dedupe(
                    os.path.join(d, "funding.csv"), fdf)
                done_funding.add(ym)
            else:
                done_funding.add(ym)  # 404 واقعی — این ماه فاندینگ ندارد

        meta["months"] = sorted(done_months)
        meta["funding_months"] = sorted(done_funding)
        _save_meta(symbol, interval, meta)
        logger.info(f"{symbol}/{interval}: ماه {ym} آماده شد "
                    f"(کندل کل +{got['klines']} | فاندینگ +{got['funding']}).")
        time.sleep(REQUEST_DELAY_SECONDS)
        ym = _next_month(ym)

    # --- Open Interest: روزهای اخیر (metrics روزانه) ---
    if metrics_days > 0:
        start_date = now_utc.date() - timedelta(days=metrics_days)
        yesterday = now_utc.date() - timedelta(days=1)
        day = start_date
        while day <= yesterday:
            ds = day.isoformat()
            if ds not in done_metrics:
                murl = (f"{VISION_BASE}/daily/metrics/{symbol}/"
                        f"{symbol}-metrics-{ds}.zip")
                try:
                    mraw = _read_zip_csv(murl)
                except ArchiveError as e:
                    logger.error(f"❌ metrics {symbol} روز {ds}: {e}")
                    day += timedelta(days=1)
                    continue  # done نمی‌شود → اجرای بعدی دوباره تلاش می‌کند
                if mraw is not None:
                    mdf = _normalize_metrics(mraw)
                    got["metrics"] += _append_and_dedupe(
                        os.path.join(d, "metrics.csv"), mdf)
                    done_metrics.add(ds)
                else:
                    done_metrics.add(ds)  # 404 واقعی → تکرار نکن
                time.sleep(REQUEST_DELAY_SECONDS)
            day += timedelta(days=1)
        meta["metrics_days"] = sorted(done_metrics)
        _save_meta(symbol, interval, meta)
        logger.info(f"{symbol}/{interval}: OI روزهای اخیر به‌روز شد "
                    f"(+{got['metrics']} ردیف).")

    return got


# ----------------------------------------------------------------------
# خواندن ترکیبی (همان قالب get_futures_dataset)
# ----------------------------------------------------------------------
def load_combined(symbol: str, interval: str) -> pd.DataFrame:
    """
    کندل + فاندینگ + OI محلی را روی محور زمانی ترکیب می‌کند (merge_asof
    با جهت backward — بدون look-ahead) و DataFrame با ستون‌های:
    open, high, low, close, volume, quote_volume, taker_buy_volume,
    funding_rate, funding_rate_pct, open_interest, open_interest_value
    برمی‌گرداند. اگر مخزن خالی باشد → DataFrame خالی.
    """
    symbol = symbol.upper()
    d = _store_dir(symbol, interval)
    kpath = os.path.join(d, "klines.csv")
    if not os.path.exists(kpath):
        return pd.DataFrame()
    out = pd.read_csv(kpath, index_col=0, parse_dates=True).sort_index()
    if out.empty:
        return out

    fpath = os.path.join(d, "funding.csv")
    if os.path.exists(fpath):
        funding = pd.read_csv(fpath, index_col=0, parse_dates=True).sort_index()
        if not funding.empty:
            out = pd.merge_asof(
                out.reset_index(), funding.reset_index(),
                on="timestamp", direction="backward",
            ).set_index("timestamp")
    if "funding_rate" not in out.columns:
        out["funding_rate"] = float("nan")
    out["funding_rate_pct"] = out["funding_rate"] * 100.0

    mpath = os.path.join(d, "metrics.csv")
    if os.path.exists(mpath):
        metrics = pd.read_csv(mpath, index_col=0, parse_dates=True).sort_index()
        if not metrics.empty:
            out = pd.merge_asof(
                out.reset_index(), metrics.reset_index(),
                on="timestamp", direction="backward",
            ).set_index("timestamp")
    if "open_interest" not in out.columns:
        out["open_interest"] = float("nan")
        out["open_interest_value"] = float("nan")

    return out[COMBINED_COLUMNS]


def coverage(symbol: str, interval: str) -> dict:
    """گزارش پوشش مخزن برای یک نماد/تایم‌فریم."""
    df = load_combined(symbol, interval)
    if df.empty:
        return {"symbol": symbol.upper(), "interval": interval, "rows": 0}
    miss = df.isna().mean()
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "rows": len(df),
        "first": str(df.index[0]),
        "last": str(df.index[-1]),
        "funding_na_pct": round(float(miss.get("funding_rate", 1.0)) * 100, 1),
        "oi_na_pct": round(float(miss.get("open_interest", 1.0)) * 100, 1),
    }