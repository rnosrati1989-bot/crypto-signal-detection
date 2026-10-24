"""
smart_money.py
--------------
ویژگی‌های «پول هوشمند» (Smart Money) برای مدل یادگیری ماشین — نسخه ۲.

ایده‌ی اصلی: رفتار معامله‌گران نهادی و «پول هوشمند» معمولاً در قیمتِ ساده
دیده نمی‌شود؛ در «جریان سفارش» و «تمرکز حجم روی سطوح قیمتی» دیده می‌شود.
داده‌ی فیوچرز بایننس این امکان را می‌دهد چون ستون taker_buy_volume دارد
(حجم خریدِ فعال/تهاجمی) و می‌توان دلتای واقعی حجم (CVD) ساخت.

┌─────────────────────────────────────────────────────────────────────┐
│  چرا نسخه ۲؟ (درس‌های آزمایش sm-only روی 15m و 1h)                   │
│                                                                     │
│  آزمایش نشان داد AUC فقط با ویژگی‌های SM = 0.446 (پایین‌تر از شانس). │
│  تحلیل تک‌متغیره نشان داد «وارونگی» همه از ستون‌های سطحِ مطلق بود:   │
│  poc/va/range/cvd — این‌ها غیرایستا هستند و رابطه‌شان با هدف به      │
│  رژیم بازار (روند Train) وابسته است. برگرداندن علامتشان = overfit    │
│  به رژیم Train. پس: سطح‌های مطلق حذف شدند و CVD با نسخه‌ی ایستا      │
│  (z-score چرخان) جایگزین شد.                                        │
│                                                                     │
│  همچنین باگ Spring/Upthrust پیدا شد: رنج با پنجره‌ی شاملِ کندلِ     │
│  جاری تعریف شده بود و شرط «شکست» ریاضیاً غیرممکن بود → همیشه صفر.   │
│  حالا رنج «پیش از کندل شکست» (تا t-2) تعریف می‌شود.                │
└─────────────────────────────────────────────────────────────────────┘

چهار خانواده‌ی ویژگی (همه ایستا / بدون look-ahead):

۱) Cumulative Volume Delta (CVD) — ایستا
   delta = taker_buy_volume - (volume - taker_buy_volume)
   cvd_z = z-score چرخانِ جمع تجمعی delta (انحراف از میانگین اخیرش)
   cvd_change_{4,24} = تغییر دلتا در پنجره‌های کوتاه
   cvd_divergence_24 = واگرایی قیمت/CVD (قیمت حرکت کرد ولی دلتا تأیید نکرد)

۲) Market / Volume Profile (پروفایل حجم در قیمت)
   POC و Value Area روی پنجره‌ی چرخان — ولی فقط نسخه‌های «نسبی» خروجی
   داده می‌شوند (سطح‌های مطلق حذف شده‌اند):
   - price_to_poc: فاصله‌ی قیمت تا POC (نسبی)
   - pos_in_va: موقعیت قیمت داخل Value Area
   - va_width_pct: پهنای Value Area نسبت به قیمت
   - poc_concentration: تمرکز حجم در POC

۳) Wyckoff (ساده‌شده — باگ رفع شد)
   - spring: شکستِ کاذبِ کف رنج (رنج تا t-2) و برگشت داخل
   - upthrust: شکستِ کاذبِ سقف رنج و برگشت داخل
   - pos_in_range: موقعیت قیمت داخل رنج چرخان
   - spring_count_24 / upthrust_count_24

۴) فشار سفارش (تقریبی از taker data)
   - buy_pressure_ma, buy_pressure_z, buy_pressure_momentum

قوانین:
   - هر ویژگی در کندل t فقط از اطلاعات «تا کندل t» ساخته می‌شود.
   - مقادیر NaN دوره‌ی گرم‌آپ عمداً حذف نمی‌شوند؛ LightGBM خودش مدیریت می‌کند.

نحوه‌ی استفاده:
    from smart_money import SmartMoneyIndicators
    df = SmartMoneyIndicators.calculate_all_smart_money(df)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ستون‌هایی که این ماژول می‌سازد (برای مقایسه‌ی «با/بدون» در آزمایش‌ها)
SMART_MONEY_COLUMNS = [
    "cvd_z", "cvd_change_4", "cvd_change_24", "cvd_divergence_24",
    "price_to_poc", "pos_in_va", "va_width_pct", "poc_concentration",
    "pos_in_range", "spring", "upthrust", "spring_count_24",
    "upthrust_count_24",
    "buy_pressure_ma", "buy_pressure_z", "buy_pressure_momentum",
]


# «هسته‌ی قوی» Smart Money: ویژگی‌هایی که در هر دو تایم‌فریم (15m و 1h)
# توسط مدل استفاده شدند و Gain معنادار داشتند (بر اساس آزمایش نسخه ۲)
SMART_MONEY_CORE = [
    "va_width_pct", "cvd_z", "price_to_poc",
    "poc_concentration", "pos_in_va", "cvd_change_24",
]


def get_smart_money_columns(df: pd.DataFrame) -> list:
    """ستون‌های Smart Money که واقعاً در دیتافریم وجود دارند."""
    return [c for c in SMART_MONEY_COLUMNS if c in df.columns]


def get_core_sm_columns(df: pd.DataFrame) -> list:
    """ستون‌های «هسته‌ی قوی» Smart Money که در دیتافریم وجود دارند."""
    return [c for c in SMART_MONEY_CORE if c in df.columns]


class SmartMoneyIndicators:
    """ساخت ویژگی‌های Smart Money از داده‌ی خام فیوچرز."""

    @staticmethod
    def calculate_all_smart_money(
        df: pd.DataFrame,
        profile_window: int = 96,
        range_window: int = 48,
        pressure_window: int = 24,
    ) -> pd.DataFrame:
        """
        ساخت همه‌ی ویژگی‌های Smart Money روی DataFrame ورودی.

        :param profile_window: پنجره‌ی پروفایل حجم (96 کندل =
                               ۱ روز روی 15m / ۴ روز روی 1h)
        :param range_window:   پنجره‌ی تشخیص رنج برای Wyckoff (48 کندل)
        :param pressure_window: پنجره‌ی فشار سفارش (24 کندل)

        نیازمندی ستون‌ها: open, high, low, close, volume, taker_buy_volume
        (اگر taker_buy_volume نباشد، بخش CVD/فشار سفارش رد می‌شود.)
        """
        if df.empty or len(df) < 20:
            logger.warning("Dataframe خیلی کوتاه است؛ ویژگی‌های Smart Money ساخته نشد.")
            return df

        df = df.copy()
        has_taker = "taker_buy_volume" in df.columns

        # ---- ۱. Cumulative Volume Delta (ایستا) ----
        if has_taker:
            delta = 2.0 * df["taker_buy_volume"] - df["volume"]
            cvd = delta.cumsum()
            # z-score چرخان: انحراف CVD از میانگین اخیر خودش (ایستا)
            cvd_ma = cvd.rolling(profile_window).mean()
            cvd_std = cvd.rolling(profile_window).std().replace(0, np.nan)
            df["cvd_z"] = (cvd - cvd_ma) / cvd_std
            df["cvd_change_4"] = cvd.diff(4)
            df["cvd_change_24"] = cvd.diff(24)
            # واگرایی قیمت/CVD در ۲۴ کندل:
            #   +1 = قیمت بالا رفته ولی CVD تأیید نکرده (توزیع پنهان — هشدار نزولی)
            #   -1 = قیمت پایین رفته ولی CVD تأیید نکرده (انباشت پنهان — هشدار صعودی)
            price_chg = df["close"].diff(pressure_window)
            cvd_chg = cvd.diff(pressure_window)
            df["cvd_divergence_24"] = np.where(
                (price_chg > 0) & (cvd_chg <= 0), 1.0,
                np.where((price_chg < 0) & (cvd_chg >= 0), -1.0, 0.0)
            )
        else:
            logger.warning("taker_buy_volume پیدا نشد؛ بخش CVD و فشار سفارش رد شد.")

        # ---- ۲. Market / Volume Profile (فقط خروجی‌های نسبی) ----
        poc, va_high, va_low, poc_conc = SmartMoneyIndicators._volume_profile(
            df, profile_window)
        df["price_to_poc"] = (df["close"] - poc) / df["close"]
        va_range = np.where((va_high - va_low) == 0, np.nan, va_high - va_low)
        df["pos_in_va"] = (df["close"] - va_low) / va_range
        df["va_width_pct"] = va_range / df["close"]
        df["poc_concentration"] = poc_conc

        # ---- ۳. Wyckoff (ساده‌شده — رنج تا t-2 برای رویداد شکست) ----
        range_high = df["high"].rolling(range_window).max()
        range_low = df["low"].rolling(range_window).min()
        range_span = (range_high - range_low).replace(0, np.nan)
        df["pos_in_range"] = (df["close"] - range_low) / range_span

        # کف/سقف رنج «پیش از کندل شکست»: پنجره‌ی ۴۸ کندل تا t-2
        range_high_prior = df["high"].rolling(range_window).max().shift(2)
        range_low_prior = df["low"].rolling(range_window).min().shift(2)

        # Spring: کندل t-1 کف رنجِ تثبیت‌شده را شکست و کندل t بالای آن بسته شد
        df["spring"] = (
            (df["low"].shift(1) < range_low_prior) &
            (df["close"] > range_low_prior)
        ).astype(float)

        # Upthrust: کندل t-1 سقف رنج را شکست و کندل t زیر آن بسته شد
        df["upthrust"] = (
            (df["high"].shift(1) > range_high_prior) &
            (df["close"] < range_high_prior)
        ).astype(float)

        # تعداد اسپرینگ/آپ‌تراست در ۲۴ کندل اخیر (شدت «شکار استاپ»)
        df["spring_count_24"] = df["spring"].rolling(pressure_window).sum()
        df["upthrust_count_24"] = df["upthrust"].rolling(pressure_window).sum()

        # ---- ۴. فشار سفارش (تقریبی از taker data) ----
        if has_taker:
            if "taker_buy_ratio" not in df.columns:
                df["taker_buy_ratio"] = (
                    df["taker_buy_volume"] / df["volume"].replace(0, np.nan)
                )
            ratio_ma = df["taker_buy_ratio"].rolling(pressure_window).mean()
            ratio_std = df["taker_buy_ratio"].rolling(pressure_window).std()
            df["buy_pressure_ma"] = ratio_ma
            # Z-score: تهاجمی بودن خرید نسبت به رفتار اخیر خودش
            df["buy_pressure_z"] = (df["taker_buy_ratio"] - ratio_ma) / ratio_std.replace(0, np.nan)
            # شتاب فشار خرید: میانگین کوتاه‌مدت (۴) منهای بلندمدت (۲۴)
            df["buy_pressure_momentum"] = (
                df["taker_buy_ratio"].rolling(4).mean() -
                df["taker_buy_ratio"].rolling(pressure_window).mean()
            )

        logger.info(f"ویژگی‌های Smart Money ساخته شد: "
                    f"{len(get_smart_money_columns(df))} ستون.")
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _volume_profile(df: pd.DataFrame, window: int, n_bins: int = 40):
        """
        پروفایل حجم در قیمت روی پنجره‌ی چرخان (فقط داده‌ی گذشته).

        برای هر کندل t، پنجره‌ی [t-window+1, t] را به n_bins باند قیمتی
        تقسیم می‌کند و حجم هر کندل را به باندِ «قیمت معمول» (typical price)
        آن تخصیص می‌دهد. سپس:
          - POC: مرکز باند با بیشترین حجم
          - Value Area: باندها از POC به‌بیرون تا پوشش ۷۰٪ حجم کل
          - تمرکز POC: سهم حجم باند POC از کل

        برمی‌گرداند: (poc, va_high, va_low, poc_concentration) به‌صورت ndarray
        (این‌ها فقط برای محاسبه‌ی ویژگی‌های نسبی استفاده می‌شوند؛ سطح‌های
        مطلق به‌عنوان ویژگی به مدل داده نمی‌شوند — غیرایستا هستند.)
        """
        n = len(df)
        poc = np.full(n, np.nan)
        va_high = np.full(n, np.nan)
        va_low = np.full(n, np.nan)
        conc = np.full(n, np.nan)

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        vol = df["volume"].fillna(0.0).to_numpy(dtype=float)

        for t in range(window - 1, n):
            lo = float(np.min(low[t - window + 1:t + 1]))
            hi = float(np.max(high[t - window + 1:t + 1]))
            if hi <= lo:
                continue

            edges = np.linspace(lo, hi, n_bins + 1)
            tp = (
                high[t - window + 1:t + 1] +
                low[t - window + 1:t + 1] +
                close[t - window + 1:t + 1]
            ) / 3.0
            idx = np.clip(np.digitize(tp, edges) - 1, 0, n_bins - 1)

            vol_sum = np.zeros(n_bins)
            np.add.at(vol_sum, idx, vol[t - window + 1:t + 1])
            total = float(vol_sum.sum())
            if total <= 0:
                continue

            poc_bin = int(np.argmax(vol_sum))
            poc[t] = 0.5 * (edges[poc_bin] + edges[poc_bin + 1])
            conc[t] = vol_sum[poc_bin] / total

            # گسترش به‌دور از POC تا پوشش ۷۰٪ حجم (Value Area)
            target = 0.70 * total
            acc = float(vol_sum[poc_bin])
            lo_bin = hi_bin = poc_bin
            while acc < target and (lo_bin > 0 or hi_bin < n_bins - 1):
                left = vol_sum[lo_bin - 1] if lo_bin > 0 else -1.0
                right = vol_sum[hi_bin + 1] if hi_bin < n_bins - 1 else -1.0
                if left >= right:
                    lo_bin -= 1
                    acc += left
                else:
                    hi_bin += 1
                    acc += right
            va_low[t] = edges[lo_bin]
            va_high[t] = edges[hi_bin + 1]

        return poc, va_high, va_low, conc
