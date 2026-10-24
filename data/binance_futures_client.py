"""
binance_futures_client.py
-------------------------
کلاینت داده‌های فیوچرز بایننس (USDT-M Futures)

داده‌هایی که این ماژول می‌گیره، مخصوص معاملات فیوچرزه و در بازار اسپات
وجود نداره:
  1. klines          -> کندل‌های قیمتی (OHLCV + حجم + حجم quote)
  2. funding_rate    -> نرخ فاندینگ (هر ۸ ساعت یک‌بار پرداخت/دریافت می‌شه)
  3. open_interest   -> موقعیت‌های باز (Open Interest) به همراه ارزش دلاری‌شون

این سه تا با هم ستون فقرات «ویژگی‌های فیوچرز» برای مدل یادگیری ماشین هستن.
فاندینگ و OI اطلاعاتی هستن که توی اسپات اصلاً وجود ندارن و می‌تونن سیگنال‌های
منحصربه‌فردی بدن (مثلاً: افزایش سریع OI + فاندینگ خیلی مثبت اغلب یعنی بازار
اشباع‌شده از پوزیشن‌های لانگ).

همه‌ی اندپوینت‌ها عمومی هستن و به API key نیازی ندارن.
"""

import os
import time
import logging
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# --- ثابت‌های اتصال ---------------------------------------------------
FUTURES_BASE_URL = "https://fapi.binance.com"
OI_HISTORY_URL = "https://fapi.binance.com/futures/data/openInterestHist"
# سفارش‌های اجباری (لیکوییدیشن) — عمومی، بدون کلید
LIQUIDATION_URL = "https://fapi.binance.com/fapi/v1/allForceOrders"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5      # بین هر تلاش، ضربدر شماره‌ی تلاش (۵، ۱۰، ۱۵ ثانیه...)
REQUEST_TIMEOUT_SECONDS = 20

# --- پروکسی (اختیاری) --------------------------------------------------
# بایننس برای بعضی کشورها/IPها HTTP 451 (مسدودسازی جغرافیایی) برمی‌گرداند.
# اگر پروکسی/VPN محلی دارید، آدرسش را در متغیر محیطی BINANCE_PROXY بگذارید؛
# فقط همین کلاینت از آن استفاده می‌کند (اتصال تلگرام دست‌نخورده می‌ماند):
#     set "BINANCE_PROXY=http://127.0.0.1:10809"
PROXY_ENV_VARS = ("BINANCE_PROXY", "FAPI_PROXY")


def _resolve_proxy() -> str:
    """آدرس پروکسی از متغیر محیطی (خالی = اتصال مستقیم)."""
    for _name in PROXY_ENV_VARS:
        _val = (os.environ.get(_name) or "").strip()
        if _val:
            return _val
    return ""


def _blocked_hint(status_code: int) -> str:
    """پیام راهنمای خطای مسدودسازی جغرافیایی (451/403)."""
    return (
        f"بایننس دسترسی این شبکه را مسدود کرده است (HTTP {status_code}) — این "
        "خطا نتیجه‌ی کد نیست، IP/کشور شما بلاک شده. راه‌حل: پروکسی/VPN روشن کنید "
        "و BINANCE_PROXY را ست کنید "
        "(مثال: set \"BINANCE_PROXY=http://127.0.0.1:10809\").")

# --- حداکثر تعداد رکورد در هر درخواست ----------------------------------
KLINES_PAGE_LIMIT = 1000
FUNDING_PAGE_LIMIT = 1000
OI_PAGE_LIMIT = 500
LIQUIDATION_PAGE_LIMIT = 100

# --- دوره‌های معتبر برای Open Interest history -------------------------
VALID_OI_PERIODS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]


def _interval_to_hours(interval: str) -> float:
    """تبدیل تایم‌فریم به ساعت: '15m' -> 0.25 ، '4h' -> 4.0 ، '1d' -> 24.0"""
    value = int(interval[:-1])
    unit = interval[-1].lower()
    if unit == "m":
        return value / 60.0
    if unit == "h":
        return float(value)
    if unit == "d":
        return value * 24.0
    if unit == "w":
        return value * 24.0 * 7
    raise ValueError(f"تایم‌فریم نامعتبر: {interval}")


def _choose_oi_period(kline_hours: float) -> str:
    """نزدیک‌ترین دوره‌ی معتبر OI که بزرگ‌تر/مساوی تایم‌فریم کندل باشه."""
    for period in VALID_OI_PERIODS:
        if _interval_to_hours(period) >= kline_hours:
            return period
    return "1d"


class BinanceFuturesDataClient:
    """
    دریافت داده‌ی فیوچرز بایننس با retry/backoff و pagination به عقب
    (برای حجم‌های بیشتر از یک صفحه). همه‌ی متدها DataFrame برمی‌گردونن.
    """

    def __init__(self):
        self.session = requests.Session()
        # پروکسی اختیاری (فقط برای همین کلاینت) — برای دور زدن مسدودسازی 451
        self.proxy = _resolve_proxy()
        if self.proxy:
            self.session.proxies.update({"http": self.proxy, "https": self.proxy})
            logger.info(f"کلاینت بایننس از پروکسی استفاده می‌کند: {self.proxy}")
        else:
            logger.debug("پروکسی بایننس تنظیم نشده — اتصال مستقیم.")

    # ------------------------------------------------------------------
    def ping(self) -> tuple:
        """
        تست سریع اتصال به بایننس. خروجی: (ok: bool, message: str)
        برای نمایش وضعیت هنگام شروع ربات و تشخیص مسدودسازی جغرافیایی (451).
        """
        url = f"{FUTURES_BASE_URL}/fapi/v1/time"
        mode = f"از طریق پروکسی {self.proxy}" if self.proxy else "مستقیم"
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        except Exception as e:
            return False, (f"اتصال به بایننس برقرار نشد ({mode}): {str(e)[:120]}. "
                           "اگر پروکسی/VPN دارید، BINANCE_PROXY را ست کنید.")
        if resp.status_code == 200:
            return True, f"اتصال به بایننس برقرار است ({mode})."
        if resp.status_code in (451, 403):
            return False, _blocked_hint(resp.status_code) + f" (اتصال {mode})"
        return False, f"پاسخ نامنتظره از بایننس: HTTP {resp.status_code} (اتصال {mode})"

    # ------------------------------------------------------------------
    def _get_with_retry(self, url: str, params: dict = None):
        """
        یک درخواست GET با retry در برابر قطعی موقت و ریت‌لیمیت.
        در صورت شکست کامل None برمی‌گردونه (نه raise).
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
                if resp.status_code == 200:
                    return resp.json()

                # ریت‌لیمیت یا خطای موقت سرور -> تلاش مجدد
                if resp.status_code in (429, 418) or resp.status_code >= 500:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        f"خطای {resp.status_code} از بایننس (تلاش {attempt}/{MAX_RETRIES}): "
                        f"{resp.text[:120]}. تلاش مجدد پس از {wait} ثانیه..."
                    )
                    time.sleep(wait)
                    continue

                # مسدودسازی جغرافیایی (451) یا بلاک WAF (403) — تلاش مجدد بی‌فایده است
                if resp.status_code in (451, 403):
                    logger.error(_blocked_hint(resp.status_code))
                    return None

                # خطای غیرقابل‌بازگشت (مثل 400 به‌خاطر پارامتر اشتباه)
                logger.error(f"خطای غیرقابل‌بازگشت {resp.status_code} از {url}: {resp.text[:200]}")
                return None

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        f"خطای اتصال (تلاش {attempt}/{MAX_RETRIES}): {str(e)[:150]}. "
                        f"تلاش مجدد پس از {wait} ثانیه..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"خطای اتصال پس از {MAX_RETRIES} تلاش: {str(e)}")
        return None

    # ------------------------------------------------------------------
    def _fetch_paginated(self, url: str, params_base: dict, page_limit: int,
                         limit: int, time_key, stop_on_short: bool = True):
        """
        دریافت صفحه‌به‌صفحه (به عقب) از یک اندپوینت بایننس.

        :param time_key: کلید زمان در هر ردیف. برای klines که آرایه است None
                         بده تا از ایندکس ۰ استفاده بشه؛ برای dictها اسم کلید
                         (مثل 'fundingTime' یا 'timestamp').
        """
        all_rows = []
        end_time_ms = None
        remaining = limit

        while remaining > 0:
            page_limit_now = min(page_limit, remaining)
            params = dict(params_base)
            params["limit"] = page_limit_now
            if end_time_ms is not None:
                params["endTime"] = end_time_ms

            page = self._get_with_retry(url, params)
            if not page:
                break

            all_rows = page + all_rows
            remaining -= len(page)

            if stop_on_short and len(page) < page_limit_now:
                break  # به ابتدای تاریخچه رسیدیم

            # زمان قدیمی‌ترین ردیف همین صفحه - ۱ میلی‌ثانیه، برای صفحه‌ی بعد
            if isinstance(page[0], dict):
                page_time = page[0][time_key]
            else:
                page_time = page[0][0]
            end_time_ms = page_time - 1

        return all_rows

    # ------------------------------------------------------------------
    def get_futures_klines(self, symbol: str, interval: str = "1h",
                           limit: int = 1000) -> pd.DataFrame:
        """کندل‌های فیوچرز (fapi/v1/klines) — عمومی، بدون کلید."""
        rows = self._fetch_paginated(
            f"{FUTURES_BASE_URL}/fapi/v1/klines",
            {"symbol": symbol.upper(), "interval": interval},
            KLINES_PAGE_LIMIT, limit, time_key=None
        )
        if not rows:
            logger.error(f"هیچ کندلی برای {symbol}/{interval} دریافت نشد.")
            return pd.DataFrame()

        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(rows, columns=columns)
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

        for col in ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume",
                 "quote_volume", "taker_buy_volume"]]
        return df

    # ------------------------------------------------------------------
    def get_funding_rate_history(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """تاریخچه‌ی نرخ فاندینگ (fapi/v1/fundingRate) — عمومی."""
        rows = self._fetch_paginated(
            f"{FUTURES_BASE_URL}/fapi/v1/fundingRate",
            {"symbol": symbol.upper()},
            FUNDING_PAGE_LIMIT, limit, time_key="fundingTime", stop_on_short=False
        )
        if not rows:
            logger.error(f"تاریخچه‌ی فاندینگ برای {symbol} دریافت نشد.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset="fundingTime").sort_values("fundingTime")
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df[["fundingTime", "funding_rate", "markPrice"]]

    # ------------------------------------------------------------------
    def get_open_interest_history(self, symbol: str, period: str = "1h",
                                  limit: int = 500) -> pd.DataFrame:
        """تاریخچه‌ی Open Interest (futures/data/openInterestHist) — عمومی."""
        if period not in VALID_OI_PERIODS:
            raise ValueError(
                f"دوره‌ی OI نامعتبر: {period}. مقادیر معتبر: {VALID_OI_PERIODS}"
            )
        rows = self._fetch_paginated(
            OI_HISTORY_URL,
            {"symbol": symbol.upper(), "period": period},
            OI_PAGE_LIMIT, limit, time_key="timestamp"
        )
        if not rows:
            logger.error(f"تاریخچه‌ی Open Interest برای {symbol} دریافت نشد.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
        df["open_interest_value"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
        return df[["timestamp", "open_interest", "open_interest_value"]]

    # ------------------------------------------------------------------
    def get_liquidation_orders(self, symbol: str, limit: int = 1000,
                               start_time_ms: int = None,
                               end_time_ms: int = None) -> pd.DataFrame:
        """
        سفارش‌های اجباری (لیکوییدیشن) یک نماد — fapi/v1/allForceOrders.

        ⚠️ وضعیت بررسی‌شده (۲۰۲۶-۱۰): این اندپوینت دیگر روی fapi.binance.com
        وجود ندارد و HTTP 404 برمی‌گرداند (بایننس فید عمومی لیکوییدیشن را
        بازنشسته کرده است). پس در عمل این تابع DataFrame خالی برمی‌گرداند.
        پاراگراف زیر شرح طراحی اولیه است و صرفاً سابقه‌ی تاریخی محسوب می‌شود:

        ⚠️ صداقت: بایننس تاریخچه‌ی این اندپوینت را فقط برای بازه‌ی محدود
        (چند هفته/ماه اخیر) نگه می‌دارد؛ بنابراین برای آموزشِ چندساله قابل
        استفاده نیست و بیشتر برای «فشار لحظه‌ای لیکوییدیشن» در اجرای زنده
        به کار می‌آید. اگر داده نبود، DataFrame خالی برمی‌گرداند (نه خطا).

        ستون side سمت سفارشِ اجراشده است: SELL → لانگ لیکویید شده،
        BUY → شورت لیکویید شده.
        """
        params = {"symbol": symbol.upper()}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        rows = self._fetch_paginated(
            LIQUIDATION_URL, params, LIQUIDATION_PAGE_LIMIT, limit,
            time_key="time"
        )
        empty = pd.DataFrame(columns=["timestamp", "side", "price", "qty", "notional"])
        if not rows:
            logger.warning(f"لیکوییدیشنی برای {symbol} دریافت نشد "
                           "(تاریخچه‌ی کوتاه است یا نماد پشتیبانی نمی‌شود).")
            return empty

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        df["qty"] = pd.to_numeric(df.get("executedQty"), errors="coerce")
        df["notional"] = df["price"] * df["qty"]          # ارزش دلاری سفارش
        df["side"] = df["side"].astype(str).str.upper()
        df = df.drop_duplicates(subset=["timestamp", "side", "price", "qty"])
        return df[["timestamp", "side", "price", "qty", "notional"]] \
            .sort_values("timestamp")

    # ------------------------------------------------------------------
    def get_futures_dataset(self, symbol: str, interval: str = "1h",
                            limit: int = 1000) -> pd.DataFrame:
        """
        ترکیب کندل‌ها + فاندینگ + OI روی یک محور زمانی.

        فاندینگ (هر ۸ ساعت) و OI (مثلاً هر ۱ ساعت) با merge_asof و جهت
        «backward» به هر کندل چسبیده می‌شن: یعنی برای هر کندل، آخرین مقدار
        شناخته‌شده‌ی فاندینگ/OI تا *آغاز همان کندل* — بدون look-ahead bias.

        خروجی یک DataFrame با ایندکس timestamp و ستون‌های:
        open, high, low, close, volume, quote_volume, taker_buy_volume,
        funding_rate, funding_rate_pct, open_interest, open_interest_value
        """
        klines = self.get_futures_klines(symbol, interval, limit=limit)
        if klines.empty:
            return klines

        kline_hours = _interval_to_hours(interval)

        # فاندینگ هر ۸ ساعت -> به‌اندازه‌ی کافی برمی‌گردونیم
        funding_limit = max(50, int(limit / 8) + 5)
        funding = self.get_funding_rate_history(symbol, limit=funding_limit)

        # OI با نزدیک‌ترین دوره‌ی معتبر
        oi_period = _choose_oi_period(kline_hours)
        oi_period_hours = _interval_to_hours(oi_period)
        oi_limit = max(100, int(limit * oi_period_hours / kline_hours) + 5)
        oi = self.get_open_interest_history(symbol, period=oi_period, limit=oi_limit)

        df = klines.set_index("timestamp")

        # --- چسباندن فاندینگ (بدون look-ahead) ---
        if not funding.empty:
            funding = funding.rename(columns={"fundingTime": "timestamp"}) \
                             .set_index("timestamp")
            df = pd.merge_asof(
                df.reset_index(),
                funding.reset_index()[["timestamp", "funding_rate"]],
                on="timestamp", direction="backward"
            ).set_index("timestamp")
        else:
            df["funding_rate"] = float("nan")

        # --- چسباندن Open Interest (بدون look-ahead) ---
        if not oi.empty:
            df = pd.merge_asof(
                df.reset_index(),
                oi[["timestamp", "open_interest", "open_interest_value"]],
                on="timestamp", direction="backward"
            ).set_index("timestamp")
        else:
            df["open_interest"] = float("nan")
            df["open_interest_value"] = float("nan")

        # نرخ فاندینگ به درصد (خواناتر برای مدل و گزارش)
        df["funding_rate_pct"] = df["funding_rate"] * 100.0

        return df
