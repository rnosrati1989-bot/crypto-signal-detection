
import time
import pandas as pd
from binance.client import Client
from config.settings import Config
import logging

# تنظیم لاگر برای این ماژول
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5  # بین هر تلاش، ضربدر شماره‌ی تلاش (۵، ۱۰، ۱۵ ثانیه...)
REQUEST_TIMEOUT_SECONDS = 20  # بزرگ‌تر از تایم‌اوت پیش‌فرض کتابخونه، برای پروکسی/VPN های کندتر
BINANCE_PAGE_LIMIT = 1000  # حداکثر تعداد کندلی که Binance در هر درخواست برمی‌گردونه


class BinanceDataClient:
    def __init__(self):
        """
        راه‌اندازی کلاینت بایننس با استفاده از کلیدهای امن داخل Config.
        در صورت بروز قطعی موقت شبکه/پروکسی (مثلاً هنگام استفاده از VPN)، چند بار
        با فاصله‌ی زمانی تلاش می‌کنیم قبل از این‌که واقعاً شکست بخوریم.
        """
        api_key = Config.BINANCE_API_KEY
        api_secret = Config.BINANCE_API_SECRET

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if not api_key or not api_secret:
                    logger.warning("Binance API keys are missing. Running in public-data only mode.")
                    self.client = Client(requests_params={"timeout": REQUEST_TIMEOUT_SECONDS})
                else:
                    self.client = Client(
                        api_key, api_secret,
                        requests_params={"timeout": REQUEST_TIMEOUT_SECONDS}
                    )
                    logger.info("Binance Client initialized successfully.")
                return  # موفقیت‌آمیز بود، خروج از حلقه‌ی تلاش
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        f"⚠️ اتصال به Binance ناموفق بود (تلاش {attempt}/{MAX_RETRIES}): {str(e)[:150]}. "
                        f"تلاش مجدد پس از {wait} ثانیه... (اگر از VPN/پروکسی استفاده می‌کنید، از پایدار بودن آن مطمئن شوید)"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"❌ اتصال به Binance پس از {MAX_RETRIES} تلاش شکست خورد: {str(e)}")

        raise ConnectionError(f"عدم امکان اتصال به Binance API پس از {MAX_RETRIES} تلاش: {last_error}")

    def _fetch_page_with_retry(self, symbol: str, interval: str, page_limit: int, end_time_ms=None):
        """
        دریافت یک «صفحه» (حداکثر ۱۰۰۰ کندل) از کندل‌ها، با retry در برابر قطعی موقت.
        اگر end_time_ms داده بشه، کندل‌های *قبل از* اون لحظه رو می‌گیره (برای pagination به عقب).
        در صورت شکست کامل، None برمی‌گردونه (نه raise، تا صداکننده تصمیم بگیره ادامه بده یا نه).
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                kwargs = {"symbol": symbol.upper(), "interval": interval, "limit": page_limit}
                if end_time_ms is not None:
                    kwargs["endTime"] = end_time_ms

                return self.client.get_klines(**kwargs)

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        f"⚠️ دریافت یک صفحه از داده برای {symbol}/{interval} ناموفق بود "
                        f"(تلاش {attempt}/{MAX_RETRIES}): {str(e)[:150]}. تلاش مجدد پس از {wait} ثانیه..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"❌ دریافت یک صفحه از داده برای {symbol}/{interval} پس از {MAX_RETRIES} تلاش شکست خورد: {str(e)}"
                    )
        return None

    def get_historical_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        دریافت کندل‌های تاریخی (OHLCV) برای یک جفت‌ارز و تایم‌فریم مشخص.

        اگر limit <= 1000 باشه، فقط یک درخواست ساده انجام میشه (رفتار قبلی، بدون تغییر).
        اگر limit > 1000 باشه (مثلاً برای Walk-Forward Validation با چند سال داده)،
        به‌صورت خودکار چند صفحه (هر کدوم حداکثر ۱۰۰۰ کندل) به عقب pagination می‌کنه
        و همه رو به هم می‌چسبونه — چون API بایننس در هر درخواست بیشتر از ۱۰۰۰ کندل نمیده.

        :param symbol: نام جفت ارز (مثلاً BTCUSDT)
        :param interval: تایم فریم (مثلاً 1h, 15m, 1d)
        :param limit: تعداد کل کندل‌های درخواستی (می‌تونه بیشتر از ۱۰۰۰ باشه)
        :return: یک DataFrame مرتب‌شده (صعودی بر اساس زمان) از داده‌های قیمتی
        """
        all_klines = []
        end_time_ms = None
        remaining = limit

        while remaining > 0:
            page_limit = min(BINANCE_PAGE_LIMIT, remaining)
            logger.info(
                f"Fetching {page_limit} klines for {symbol} on {interval} timeframe "
                f"(مجموع تاکنون: {len(all_klines)} از {limit})..."
            )

            page = self._fetch_page_with_retry(symbol, interval, page_limit, end_time_ms)
            if not page:
                # یا خطای کامل بود، یا دیگه داده‌ی قدیمی‌تری برای برگردوندن نیست
                break

            all_klines = page + all_klines
            remaining -= len(page)

            if len(page) < page_limit:
                # یعنی به ابتدای تاریخچه‌ی موجود در بایننس رسیدیم
                break

            # برای صفحه‌ی بعدی، از قبل از اولین کندل همین صفحه شروع می‌کنیم
            earliest_open_time = page[0][0]
            end_time_ms = earliest_open_time - 1

        if not all_klines:
            logger.error(f"هیچ داده‌ای برای {symbol} دریافت نشد.")
            return pd.DataFrame()

        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]

        df = pd.DataFrame(all_klines, columns=columns)

        # حذف احتمالیِ ردیف‌های تکراری (لبه‌ی صفحات pagination) و مرتب‌سازی زمانی
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df = df[['open', 'high', 'low', 'close', 'volume']]

        # اگر مجموع صفحات کمی بیشتر از limit شد (به‌خاطر مرز صفحات)، فقط آخرین `limit` ردیف رو نگه می‌داریم
        if len(df) > limit:
            df = df.iloc[-limit:]

        logger.info(f"Successfully fetched {len(df)} rows for {symbol}.")
        return df
