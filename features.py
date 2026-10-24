"""
features.py
-----------
ساخت «ویژگی» (Feature) برای مدل یادگیری ماشین از داده‌ی خام فیوچرز.

داده‌ی خام (خروجی binance_futures_client) فقط قیمت/حجم/فاندینگ/OI دارد.
مدل با این اعداد خام خوب یاد نمی‌گیرد؛ به «ویژگی» نیاز دارد: نسبت‌ها،
تغییرات، میانگین‌های چرخان و... که رفتار بازار را فشرده و قابل‌یادگیری کند.

قوانین ساخت ویژگی (خیلی مهم):
1. هر ویژگی در کندل t فقط از اطلاعات «تا کندل t» ساخته می‌شود (بدون look-ahead).
2. اندیکاتورهای تکنیکال از ماژول‌های موجود پروژه استفاده می‌شوند.
3. ویژگی‌های اختصاصی فیوچرز (فاندینگ، OI، فشار خرید) جدا اضافه می‌شوند.
4. تابع add_targets برچسب (label) را می‌سازد — این برچسب عمداً از آینده
   استفاده می‌کند ولی هیچ‌وقت نباید به عنوان ورودی مدل داده شود.

بازنگری: یادگیری از آزمایش Smart Money —
سطوح مطلق قیمت (EMA، PSAR، SuperTrend، ایچیموکو، VWAP، OBV، باندها) غیرایستا
هستند و مدل به‌جای «ساختار» رژیم قیمتی را حفظ می‌کند. این نسخه:
  - نسخه‌های نرمال‌شده/نسبتیِ ایستا اضافه می‌کند (فاصله از EMA بر حسب ٪، جایگاه
    در رنج، z-score حجم/OBV، فاصله از VWAP و...).
  - ستون‌های مطلقِ قیمت را از ورودی مدل حذف می‌کند.
  - الگوهای کندلی را به‌صورت «جمع‌بندی چندکندله» به مدل وصل می‌کند
    (پل زدن الگوها به سیگنال — قابل آزمون با Walk-Forward).
"""

import logging
import numpy as np
import pandas as pd

from indicators.trend import TrendIndicators
from indicators.momentum import MomentumIndicators
from indicators.volatility import VolatilityIndicators
from indicators.volume import VolumeIndicators
from smart_money import SmartMoneyIndicators

logger = logging.getLogger(__name__)

# ستون‌های خام (نه ویژگی) — برای جدا کردن ویژگی‌ها از داده‌ی اولیه
RAW_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_volume",
    "funding_rate", "open_interest",
    "open_interest_value", "funding_rate_pct",
    # ستون‌های خامِ اختیاریِ لیکوییدیشن (اگر از اندپوینت allForceOrders آمده باشند)
    "liq_long_notional", "liq_short_notional",
]

# ستون‌های «قیمت مطلق / غیرایستا» که از ورودی مدل حذف می‌شوند.
# استدلال: در آزمایش Smart Money مشخص شد سطوح مطلق قیمت به‌جای یادگیریِ
# ساختار، رژیم قیمتی را حفظ می‌کنند و در بازه‌های ندیده خراب می‌شوند.
# به‌جای این‌ها نسخه‌های نرمال‌شده (در ادامه‌ی همین تابع) ساخته می‌شوند.
ABSOLUTE_LEVEL_COLUMNS = [
    "EMA_9", "EMA_21", "EMA_50", "EMA_200",
    "BB_HIGH", "BB_MID", "BB_LOW",
    "KC_HIGH", "KC_MID", "KC_LOW",
    "DC_HIGH", "DC_MID", "DC_LOW",
    "VWAP", "OBV",
]
# پیشوند ستون‌های مطلق (پارابولیک سار، ابر ایچیموکو، سوپرترند)
ABSOLUTE_LEVEL_PREFIXES = ("ISA_", "ISB_", "ICT_", "ICS_", "IKS_", "ITS_",
                           "SUPERTREND", "PSAR")


# تاخیر تأییدِ swing (باید با swing_window در PatternDetector.get_all_chart_patterns
# یکی باشد؛ پیش‌فرض آن 5 است).
SWING_CONFIRM_LAG = 5


# ستون‌هایی که اطلاعات آینده دارند و باید همیشه حذف شوند
LEAKY_PREFIXES = ("ICS_",)


# ----------------------------------------------------------------------
# context بیتکوین (ویژگیهای متقاطع برای نمادهای آلت)
# ----------------------------------------------------------------------
# آلتکوین‌ها تحت «بتای» بیتکوین حرکت می‌کنند؛ بازده و ADX بیتکوین روی همان
# تایم‌فریم اطلاعاتی می‌دهد که در داده‌ی خودِ آلت وجود ندارد (جهتِ بازار).
BTC_CONTEXT_SYMBOL = "BTCUSDT"

# ستون‌هایی که attach_btc_context روی هر نماد می‌چسباند
BTC_CONTEXT_COLUMNS = [
    "btc_ret_1", "btc_ret_4", "btc_ret_24", "btc_ret_96",
    "btc_adx", "btc_trend", "btc_atr_pct", "btc_range_pos", "btc_vol_z",
]

# ----------------------------------------------------------------------
# لیکوییدیشن (فقط وقتی داده‌ی خامش از allForceOrders آمده باشد)
# ----------------------------------------------------------------------
LIQUIDATION_RAW_COLUMNS = ("liq_long_notional", "liq_short_notional")
LIQUIDATION_WINDOWS = (24, 96)

# هشدار «لیکوییدیشن خواستی ولی داده نبود» فقط یک‌بار چاپ شود (وگرنه در حلقه‌ی
# ربات برای هر نماد در هر بررسی تکرار می‌شود).
_LIQ_WARNED = False


def _is_datetime_index(df: pd.DataFrame) -> bool:
    return isinstance(df.index, pd.DatetimeIndex)


def build_features(df: pd.DataFrame,
                   include_smart_money: bool = False,
                   include_chart_patterns: bool = False,
                   include_scores: bool = False,
                   include_absolute_levels: bool = False,
                   include_elliott: bool = False,
                   include_liquidations: bool = False) -> pd.DataFrame:
    """
    تبدیل داده‌ی خام فیوچرز به DataFrame کامل ویژگی‌ها.

    ورودی باید حداقل ستون‌های: open, high, low, close, volume,
    taker_buy_volume, funding_rate, open_interest, open_interest_value
    داشته باشد.
    """
    if df.empty:
        return df

    df = df.copy()

    # ۱. اندیکاتورهای تکنیکال (از ماژول‌های موجود پروژه)
    df = TrendIndicators.calculate_all_trend(df)
    df = MomentumIndicators.calculate_all_momentum(df)
    df = VolatilityIndicators.calculate_all_volatility(df)
    df = VolumeIndicators.calculate_all_volume(df)

    close = df["close"]

    # ۲. ویژگی‌های قیمتی پایه (همه فقط از داده‌ی گذشته/حال)
    df["ret_1"] = close.pct_change(1)
    df["ret_4"] = close.pct_change(4)
    df["ret_24"] = close.pct_change(24)
    df["ret_96"] = close.pct_change(96)
    df["volatility_24"] = df["ret_1"].rolling(24).std()
    df["range_pct"] = (df["high"] - df["low"]) / close

    # ۳. ویژگی‌های اختصاصی فیوچرز — این‌ها همان چیزی هستند که مدل را
    #    از «اندیکاتورهای معمولی که همه دارند» متمایز می‌کنند.
    volume = df["volume"].replace(0, np.nan)

    # فشار خرید: نسبت حجم خریدِ فعال (taker buy) به کل حجم
    df["taker_buy_ratio"] = df["taker_buy_volume"] / volume
    df["taker_buy_ratio_ma_24"] = df["taker_buy_ratio"].rolling(24).mean()

    # فاندینگ: مقدار فعلی + میانگین چرخان + تغییر نسبت به یک سیکل قبل
    df["funding_ma_32"] = df["funding_rate"].rolling(32).mean()  # 32 کندل 15m = 8 ساعت
    df["funding_change_32"] = df["funding_rate"] - df["funding_rate"].shift(32)

    # Open Interest: تغییرات درصدی در پنجره‌های مختلف
    df["oi_change_4"] = df["open_interest"].pct_change(4)     # 1 ساعت
    df["oi_change_24"] = df["open_interest"].pct_change(24)   # 6 ساعت
    df["oi_change_96"] = df["open_interest"].pct_change(96)   # 24 ساعت
    df["oi_value_change_24"] = df["open_interest_value"].pct_change(24)

    # نسبت OI به حجم: تقریبی از «اهرم بازار» — هرچه بیشتر، موقعیت‌های باز
    # بیشتری نسبت به حجم معامله وجود دارد (می‌تواند نشانه‌ی اشباع باشد)
    df["oi_to_volume"] = df["open_interest"] / volume

    # تعامل فاندینگ و OI: شاخصی از «سقوط اهرم» — وقتی فاندینگ خیلی مثبت و OI
    # در حال کاهش است، احتمال لیکوییدیشن لانگ‌ها (شکار استاپ) بالاست.
    df["funding_oi"] = df["funding_rate"] * df["oi_change_4"]

    # ۳الف. جریان سفارش (Order Flow) — «چه کسی دارد می‌خرد/می‌فروشد»
    # ----------------------------------------------------------
    # taker_buy_volume = حجمی که با «خرید بازار» (تاکر/تهاجمی) معامله شده.
    # taker_sell = حجم کل - تیکر بای → حجمی که با «فروش بازار» خورده شده.
    # taker_delta جریان خالص را در بازه‌ی [-1, +1] خلاصه می‌کند:
    #   +1 = تمام حجم خرید تهاجمی | -1 = تمام حجم فروش تهاجمی.
    taker_sell_volume = (df["volume"] - df["taker_buy_volume"]).clip(lower=0.0)
    df["taker_sell_ratio"] = taker_sell_volume / volume
    df["taker_delta"] = (df["taker_buy_volume"] - taker_sell_volume) / volume

    # انحراف فشار خرید از میانگین خودش (ایستا) — «غیرعادی بودن» جریان
    df["taker_buy_ratio_ma_8"] = df["taker_buy_ratio"].rolling(8).mean()
    df["taker_buy_ratio_dev"] = (df["taker_buy_ratio"]
                                 - df["taker_buy_ratio_ma_24"])

    # z-score جریان خالص (۹۶ کندل = ۲۴ ساعت روی 15m) — اوج‌های غیرعادی جریان
    td_ma96 = df["taker_delta"].rolling(96).mean()
    td_sd96 = df["taker_delta"].rolling(96).std()
    df["taker_delta_z"] = ((df["taker_delta"] - td_ma96) / td_sd96).clip(-3, 3)

    # جریان تجمعی (شبیه CVD) در پنجره‌های ۶ و ۲۴ ساعت
    df["taker_delta_cum_24"] = df["taker_delta"].rolling(24).sum()
    df["taker_delta_cum_96"] = df["taker_delta"].rolling(96).sum()

    # همبستگی جریان سفارش با بازده — «آیا خرید تهاجمی واقعاً قیمت را بالا می‌برد؟»
    df["flow_ret_corr_96"] = df["taker_delta"].rolling(96).corr(df["ret_1"])

    # ۳ث. فاندینگ قوی‌تر: z-score، سالانه‌شده، تجمعی و شیب
    # ----------------------------------------------------------
    df["funding_annualized"] = df["funding_rate"] * 3.0 * 365.0  # ۸ساعت → سالانه
    df["funding_cum_32"] = df["funding_rate"].rolling(32).sum()
    df["funding_cum_96"] = df["funding_rate"].rolling(96).sum()
    df["funding_slope_32"] = df["funding_ma_32"] - df["funding_ma_32"].shift(32)
    fr_ma96 = df["funding_rate"].rolling(96).mean()
    fr_sd96 = df["funding_rate"].rolling(96).std()
    df["funding_z"] = ((df["funding_rate"] - fr_ma96) / fr_sd96).clip(-3, 3)

    # ۳ج. Open Interest قوی‌تر: z-score، شتاب و واگرایی با قیمت
    # ----------------------------------------------------------
    df["oi_change_1"] = df["open_interest"].pct_change(1)
    oi_ma96 = df["open_interest"].rolling(96).mean()
    oi_sd96 = df["open_interest"].rolling(96).std()
    df["oi_z"] = ((df["open_interest"] - oi_ma96) / oi_sd96).clip(-3, 3)
    df["oi_accel_4"] = df["oi_change_4"] - df["oi_change_4"].shift(4)
    # واگرایی OI و قیمت: «پول تازه وارد شده ولی قیمت تکان نخورده؟»
    df["oi_price_div_24"] = df["oi_change_24"] - df["ret_24"]
    df["oi_to_volume_change_24"] = df["oi_to_volume"].pct_change(24)

    # ۳ح. لیکوییدیشن (اختیاری — فقط اگر داده‌ی خامش موجود باشد)
    # ----------------------------------------------------------
    # ستون‌های خام liq_* را با liquidations_to_candles از اندپوینت
    # /fapi/v1/allForceOrders می‌سازیم و به داده‌ی خام می‌چسبانیم؛ اینجا
    # فقط به ویژگی تبدیل می‌شوند.
    # ⚠️ ۲۰۲۶-۱۰: اندپوینت عمومی /fapi/v1/allForceOrders بایننس بازنشسته شده
    # (HTTP 404) و تاریخچه‌ی عمومی لیکوییدیشن از API بایننس در دسترس نیست.
    # پس این بلوک فقط وقتی فعال می‌شود که ستون‌های liq_* از منبع دیگری
    # (مثلاً خروجی یک سرویس ثالث) در داده‌ی خام موجود باشند؛ وگرنه هشدار می‌دهد.
    if include_liquidations:
        if all(c in df.columns for c in LIQUIDATION_RAW_COLUMNS):
            liq_long_raw = df["liq_long_notional"].fillna(0.0)
            liq_short_raw = df["liq_short_notional"].fillna(0.0)
            for w in LIQUIDATION_WINDOWS:
                lw = liq_long_raw.rolling(w).sum()
                sw = liq_short_raw.rolling(w).sum()
                df[f"liq_long_{w}"] = lw
                df[f"liq_short_{w}"] = sw
                tot = (lw + sw).replace(0, np.nan)
                # مثبت = لیکویید شدن شورت‌ها بیشتر (فشار صعودی/squeeze)
                df[f"liq_imbalance_{w}"] = (sw - lw) / tot
            vol_sum = df["volume"].rolling(24).sum().replace(0, np.nan)
            df["liq_to_volume_24"] = ((liq_long_raw + liq_short_raw)
                                      .rolling(24).sum() / vol_sum)
        else:
            global _LIQ_WARNED
            if not _LIQ_WARNED:
                _LIQ_WARNED = True
                logger.warning("include_liquidations=True ولی ستون‌های خام liq_* در داده "
                           "نیست — بلوک لیکوییدیشن ساخته نشد.")

    # ۳ب. ویژگی‌های نرمال‌شده (ایستا) — جایگزین سطوح مطلق قیمت
    # ----------------------------------------------------------
    # فاصله‌ی قیمت از میانگین‌های متحرک (درصد) — وضعیت روند نسبی
    df["ema9_dist"] = close / df["EMA_9"] - 1.0
    df["ema21_dist"] = close / df["EMA_21"] - 1.0
    df["ema50_dist"] = close / df["EMA_50"] - 1.0
    df["ema200_dist"] = close / df["EMA_200"] - 1.0

    # نسبت EMA سریع به کند — قدرت هم‌سویی روند کوتاه‌مدت
    df["ema9_21_cross"] = df["EMA_9"] / df["EMA_21"] - 1.0

    # نوسان نسبی (ATR ٪) + رژیم نوسان (انبساط/انقباض)
    df["atr_pct"] = df["ATR"] / close
    df["vol_regime"] = df["ATR"] / df["ATR"].rolling(48).mean()

    # جایگاه قیمت در رنج ۲۰ کندلی اخیر (۰ تا ۱) — نزدیک سقف یا کف؟
    hi20 = df["high"].rolling(20).max()
    lo20 = df["low"].rolling(20).min()
    df["range_pos"] = (close - lo20) / (hi20 - lo20).replace(0, np.nan)

    # شتاب حجم (z-score) — موج حجمی غیرعادی
    vol_ma24 = volume.rolling(24).mean()
    vol_sd24 = volume.rolling(24).std()
    df["vol_z"] = ((volume - vol_ma24) / vol_sd24).clip(-3, 3)

    # OBV نرمال‌شده (z-score) — جریان پول نسبت به میانگین اخیر
    df["obv_z"] = ((df["OBV"] - df["OBV"].rolling(48).mean())
                   / df["OBV"].rolling(48).std()).clip(-3, 3)

    # فاصله از VWAP (درصد) — جایگاه قیمت نسبت به میانگین وزنیِ روز
    df["vwap_dist"] = close / df["VWAP"] - 1.0

    # جایگاه داخل کانال کلتنر و دونچیان (۰ تا ۱، ایستا)
    df["kc_pos"] = ((close - df["KC_LOW"])
                    / (df["KC_HIGH"] - df["KC_LOW"]).replace(0, np.nan))
    df["dc_pos"] = ((close - df["DC_LOW"])
                    / (df["DC_HIGH"] - df["DC_LOW"]).replace(0, np.nan))

    # شیب پهنای بولینگر — انبساط/انقباض نوسان
    df["bbw_slope"] = df["BB_BANDWIDTH"] - df["BB_BANDWIDTH"].shift(1)

    # هیستوگرام MACD نرمال‌شده با ATR + شیب آن (شتاب مومنتوم)
    if "MACDh_12_26_9" in df.columns:
        df["macdh_norm"] = df["MACDh_12_26_9"] / df["ATR"]
        df["macdh_slope"] = (df["MACDh_12_26_9"] - df["MACDh_12_26_9"].shift(1)) / df["ATR"]

    # شیب RSI (شتاب مومنتوم)
    df["rsi_slope"] = df["RSI"] - df["RSI"].shift(1)

    # فاصله از PSAR به واحد ATR + جهت SuperTrend (هر دو ایستا)
    psarl = df.get("PSARl_0.02_0.2")
    psars = df.get("PSARs_0.02_0.2")
    if psarl is not None and psars is not None:
        psar_line = psarl.fillna(psars)
        df["psar_dist_atr"] = (close - psar_line) / df["ATR"]

    st_cols = [c for c in df.columns if str(c).startswith("SUPERTREND")]
    st_dir_cols = [c for c in st_cols if str(c).endswith("d")]
    st_line_cols = [c for c in st_cols if not str(c).endswith("d")]
    if st_line_cols:
        st_line = df[st_line_cols[0]]
        df["st_dist"] = close / st_line - 1.0
    if st_dir_cols:
        df["st_dir"] = df[st_dir_cols[0]].astype(float)  # +1 روند صعودی، -1 نزولی

    # زمان روز (اثر جلسه‌ی معاملاتی) — به‌صورت سینوس/کسینوس تا چرخه‌ای باشد
    if _is_datetime_index(df):
        hour = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # پایداری مومنتوم: چند کندل پیاپی هم‌جهت بوده‌اند؟ (+3 یعنی ۳ کندل صعودی)
    sign_s = np.sign(close.diff()).fillna(0)
    grp = (sign_s != sign_s.shift()).cumsum()
    df["streak"] = sign_s * (sign_s.groupby(grp).cumcount() + 1)

    # ۳پ. جمع‌بندی الگوهای کندلی اخیر — پل زدن الگوها به مدل
    # (با try/except تا باگ احتمالی الگوها نتواند ساخت دیتاست را بشکند)
    try:
        from strategies.patterns import PatternDetector as _PD
        dpat = _PD.detect_candle_patterns(df)
        pat_cols = [c for c in dpat.columns if c.startswith("PATTERN_")]
        if pat_cols:
            pat_frame = dpat[pat_cols].fillna(0)
            pat_net = pat_frame.sum(axis=1)  # خالص جهت الگوهای همین کندل
            df["pat_net_3"] = pat_net.rolling(3).sum()
            df["pat_net_12"] = pat_net.rolling(12).sum()
    except Exception as e:  # pragma: no cover
        logger.warning(f"الگوهای کندلی به ویژگی اضافه نشدند: {e}")

    # ۳پ.۲ الگوهای نموداری (Double Top/Bottom، H&S، مثلث، Flag/Pennant، Gartley)
    #      هر الگو فقط در لحظه‌ی «تکمیل» خودش (end_time) معلوم می‌شود → بدون look-ahead.
    #      تشخیص‌دهنده‌ها را یک‌بار روی کل تاریخ اجرا می‌کنیم و رویدادها را به شمارش
    #      ایستای «الگوهای تکمیل‌شده در ۲۴/۹۶ کندل اخیر» تبدیل می‌کنیم.
    if include_chart_patterns and _is_datetime_index(df):
        try:
            from strategies.patterns import PatternDetector as _PDC
            # فقط تشخیص‌دهنده‌هایی که کل تاریخچه را اسکن می‌کنند (Double Top/Bottom، H&S).
            # مثلث/Flag/Pennant/Gartley فقط «انتهای داده» را می‌بینند → در آموزش تقریباً
            # هرگز روشن نمی‌شدند ولی در حالت زنده روی آخرین کندل روشن می‌شدند (train/serve skew).
            pats = (_PDC.detect_double_top_bottom(df, swing_window=SWING_CONFIRM_LAG)
                    + _PDC.detect_head_and_shoulders(df, swing_window=SWING_CONFIRM_LAG))
            if pats:
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index()
                pdf = pd.DataFrame(pats)
                ends = pd.to_datetime(pdf["end_time"], errors="coerce").to_numpy()
                dirstr = pdf["direction"].astype(str).str.lower()
                dirs = np.where(dirstr.str.startswith("bull"), 1.0,
                                np.where(dirstr.str.startswith("bear"), -1.0, 0.0))
                ok = ~pd.isna(ends)
                ends, dirs = ends[ok], dirs[ok]
                if len(ends):
                    # 🔒 ضدِ look-ahead: swing فقط وقتی تأیید می‌شود که `swing_window`
                    # کندل بعد از آن هم دیده شده باشد (تعریف مرکزی در find_swing_points).
                    # پس الگو نباید در زمان خودِ swing، بلکه SWING_CONFIRM_LAG کندل
                    # بعد از end_time فعال شود.
                    pos = np.clip(df.index.searchsorted(pd.DatetimeIndex(ends))
                                  + SWING_CONFIRM_LAG, 0, len(df) - 1)
                    sig = np.zeros(len(df))
                    for p, d in zip(pos, dirs):
                        sig[p] += d
                    sig_s = pd.Series(sig, index=df.index)
                    df["chart_pat_net_24"] = sig_s.rolling(24, min_periods=1).sum()
                    df["chart_pat_net_96"] = sig_s.rolling(96, min_periods=1).sum()
                    df["chart_pat_count_96"] = sig_s.abs().rolling(96, min_periods=1).sum()
        except Exception as e:  # pragma: no cover
            logger.warning(f"الگوهای نموداری به ویژگی اضافه نشدند: {e}")

    # ۴. ویژگی‌های Smart Money (پول هوشمند): CVD، پروفایل حجم، Wyckoff، فشار سفارش
    if include_smart_money:
        # ویژگیهای Smart Money طبق آزمایش مقایسهای روی این داده edge اضافه
        # نمیکنند؛ بنابراین بهصورت پیشفرض خاموشاند (برای آزمایش: روشن کنید).
        df = SmartMoneyIndicators.calculate_all_smart_money(df)

    # ۴ب. لایه‌ی رژیم/اسکورینگ — امتیاز SignalScorer برای هر کندل
    #      (رژیم، قدرت روند، جهت مجاز فیلتر، امتیاز ۰-۱۰۰ و اجزای آن — همه فقط از کندل t).
    #      اجرای per-row با chunk تا حافظه محدود بماند؛ خروجی dict به ستون‌های عددی
    #      تبدیل می‌شود (رژیم one-hot، جهت عددی، filter_passed، امتیاز و دیتیل‌ها).
    if include_scores:
        try:
            from strategies.scoring import SignalScorer as _SS
            score_needed = {"close", "EMA_9", "EMA_21", "EMA_50", "EMA_200",
                            "ADX_14", "DMP_14", "DMN_14", "ATR", "BB_BANDWIDTH"}
            if not (score_needed - set(df.columns)):
                results = []
                step = 20_000
                for s in range(0, len(df), step):
                    recs = df.iloc[s:s + step].to_dict("records")
                    results.extend(_SS.score_row(r) for r in recs)
                res = pd.DataFrame(results)
                res.index = df.index  # ⚠️ هم‌ترازی ایندکس ضروریه وگرنه انتصاب NaN می‌شه
                df["score_0_100"] = res["score"].astype(float)
                df["score_raw"] = res["raw_score"].astype(float)
                df["score_dir"] = res["direction"].map(
                    {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0})
                df["score_conf"] = res["confidence"].astype(float)
                df["regime_strength"] = res["regime_strength"].astype(float)
                df["trend_dir_num"] = res["trend_direction"].map(
                    {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0})
                df["filter_passed"] = res["regime_filter"].apply(
                    lambda d: float(bool(d.get("passed"))))
                for rr in ("strong_trend", "weak_trend", "ranging", "high_volatility"):
                    df[f"regime_{rr}"] = (res["regime"] == rr).astype(float)
                det = res["details"].apply(pd.Series)
                for c in ("trend", "momentum", "volatility", "volume", "confluence"):
                    if c in det.columns:
                        df[f"score_{c}"] = det[c].astype(float)
        except Exception as e:  # pragma: no cover
            logger.warning(f"لایه‌ی رژیم/اسکورینگ به ویژگی اضافه نشد: {e}")

    # ۴پ. 🔒 محافظ نهایی: ستون‌های نشتی (Chikou Span = قیمت آینده) هرگز نمی‌مانند
    leak_cols = [c for c in df.columns if str(c).startswith(LEAKY_PREFIXES)]
    if leak_cols:
        df = df.drop(columns=leak_cols)

    # ۵. حذف ستون‌های قیمت مطلق/غیرایستا از ورودی مدل
    # --- فاز ۲: بلوک ساختار موج (Elliott) — اختیاری و کاملاً گذشته‌نگر ----------
    # فقط ویژگی‌های ساختاری از پیوت‌های تأییدشده (بدون شمارش ذهنی موج).
    # pd.concat استفاده می‌شود تا هشدار DataFrame fragmentation ندهد.
    if include_elliott:
        try:
            from strategies.elliott import elliott_features
            df = pd.concat([df, elliott_features(df)], axis=1)
        except Exception as _e:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                f"بلوک موج الیوت ساخته نشد (بی‌اثر روی مدل): {_e}")

    # وقتی include_absolute_levels=True، سطوح مطلق قیمت نگه‌داشته می‌شوند (ورودی مدل)
    drop_cols = ([] if include_absolute_levels
                 else [c for c in df.columns if c in ABSOLUTE_LEVEL_COLUMNS])
    if not include_absolute_levels:
        drop_cols += [c for c in df.columns if str(c).startswith(ABSOLUTE_LEVEL_PREFIXES)]
    drop_cols = list(dict.fromkeys(drop_cols))  # بدون تکرار، حفظ ترتیب
    if drop_cols:
        df = df.drop(columns=drop_cols)

    logger.info(f"ویژگی‌ها ساخته شد: {len(df)} ردیف، "
                f"{len(get_feature_columns(df))} ویژگی "
                f"({'سطوح مطلق نگه‌داشته شد' if include_absolute_levels else 'سطح مطلق حذف: ' + str(len(drop_cols))}).")
    return df


def add_targets(df: pd.DataFrame, horizon: int = 4,
                up_threshold: float = 0.002,
                down_threshold: float = 0.002) -> pd.DataFrame:
    """
    ساخت برچسب (label) برای یادگیری: در horizon کندل آینده قیمت چقدر
    بالا/پایین می‌رود.

    :param horizon: افق پیش‌بینی (تعداد کندل). پیش‌فرض 4 = یک ساعت روی 15m.
    :param up_threshold: آستانه‌ی صعود (به کسری؛ 0.002 = 0.2%)
    :param down_threshold: آستانه‌ی نزول

    ستون‌های ساخته‌شده:
        fwd_ret_{h}      : بازده واقعی آینده (برای ارزیابی/بک‌تست)
        target_up_{h}    : 1 اگر fwd_ret > up_threshold
        target_down_{h}  : 1 اگر fwd_ret < -down_threshold

    ⚠️ این ستون‌ها برچسب هستند؛ هنگام آموزش هرگز نباید به عنوان ویژگی
    به مدل داده شوند (باعث look-ahead و نتایج دروغین می‌شود).
    """
    df = df.copy()
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    df[f"fwd_ret_{horizon}"] = fwd
    df[f"target_up_{horizon}"] = (fwd > up_threshold).astype(int)
    df[f"target_down_{horizon}"] = (fwd < -down_threshold).astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """ستون‌هایی که می‌توانند به عنوان ورودی مدل استفاده شوند
    (همه‌ی ستون‌ها به‌جز داده‌ی خام و برچسب‌ها)."""
    label_cols = [c for c in df.columns
                  if c.startswith(("target_", "fwd_ret_"))]
    return [c for c in df.columns
            if c not in RAW_COLUMNS and c not in label_cols]


# ----------------------------------------------------------------------
# context بیت‌کوین — ویژگی‌های متقاطع برای نمادهای آلت
# ----------------------------------------------------------------------
def build_btc_context(btc_raw: pd.DataFrame) -> pd.DataFrame:
    """
    از داده‌ی خامِ بیت‌کوین (روی همان تایم‌فریم) چند ویژگی «context» می‌سازد:
    بازده در چند افق، ADX، نسبت EMA سریع/کند، ATR٪، جایگاه در رنج و z حجم.

    ⚠️ همه فقط از کندل‌های تا لحظه‌ی t ساخته می‌شوند (بدون look-ahead).
    خروجی: DataFrame با ایندکس timestamp و ستون‌های BTC_CONTEXT_COLUMNS
    (اگر داده کافی نباشد: DataFrame خالی).
    """
    if btc_raw is None or len(btc_raw) < 60:
        return pd.DataFrame()

    b = TrendIndicators.calculate_all_trend(btc_raw.copy())
    b = VolatilityIndicators.calculate_all_volatility(b)
    if "ATR" not in b.columns or "EMA_21" not in b.columns or "ADX_14" not in b.columns:
        logger.warning("ساخت context بیت‌کوین ناموفق بود (ستون اندیکاتور ساخته نشد).")
        return pd.DataFrame()

    close = b["close"]
    ctx = pd.DataFrame(index=b.index)
    ctx["btc_ret_1"] = close.pct_change(1)
    ctx["btc_ret_4"] = close.pct_change(4)
    ctx["btc_ret_24"] = close.pct_change(24)
    ctx["btc_ret_96"] = close.pct_change(96)
    ctx["btc_adx"] = b["ADX_14"]
    ctx["btc_trend"] = b["EMA_9"] / b["EMA_21"] - 1.0
    ctx["btc_atr_pct"] = b["ATR"] / close
    hi20 = b["high"].rolling(20).max()
    lo20 = b["low"].rolling(20).min()
    ctx["btc_range_pos"] = (close - lo20) / (hi20 - lo20).replace(0, np.nan)
    bvol = b["volume"].replace(0, np.nan)
    ctx["btc_vol_z"] = ((bvol - bvol.rolling(24).mean())
                        / bvol.rolling(24).std()).clip(-3, 3)
    ctx = ctx[BTC_CONTEXT_COLUMNS]
    ctx.index.name = getattr(btc_raw.index, "name", None) or "timestamp"
    return ctx


def attach_btc_context(df: pd.DataFrame, btc_context: pd.DataFrame) -> pd.DataFrame:
    """
    چسباندن context بیت‌کوین به DataFrame یک نماد با merge_asof جهت backward
    (برای هر کندل، آخرین context بیت‌کوینِ ≤ همان زمان — بدون look-ahead).

    اگر context خالی باشد، همه‌ی ستون‌ها NaN می‌شوند تا *شکل* دیتاست ثابت بماند
    (و آموزش و اجرای زنده دقیقاً یک مجموعه ستون داشته باشند).
    """
    df = df.copy()
    cols = list(BTC_CONTEXT_COLUMNS)
    if btc_context is None or btc_context.empty or not _is_datetime_index(df):
        for c in cols:
            df[c] = np.nan
        return df

    time_col = df.index.name or "timestamp"
    left = df.reset_index()
    left = left.rename(columns={left.columns[0]: time_col})
    right = btc_context.reset_index()
    right.columns = [time_col] + cols
    left = left.sort_values(time_col)
    right = right.sort_values(time_col)
    merged = (pd.merge_asof(left, right, on=time_col, direction="backward")
                .set_index(time_col)
                .reindex(df.index))
    for c in cols:
        df[c] = merged[c].to_numpy()

    # قدرت نسبی آلت در برابر بیت‌کوین (وقتی بازده خودِ آلت موجود باشد)
    if "ret_24" in df.columns:
        df["rel_strength_24"] = df["ret_24"] - df["btc_ret_24"]
    return df


def load_btc_context(interval: str = "15m",
                     cache_dir: str = "data_cache") -> pd.DataFrame:
    """خواندن داده‌ی خام بیت‌کوین از data_cache و ساخت context (برای build_dataset)."""
    import os  # فقط همین تابع به مسیر فایل نیاز دارد
    path = os.path.join(cache_dir, f"{BTC_CONTEXT_SYMBOL}_{interval}.csv")
    if not os.path.exists(path):
        logger.warning(f"داده‌ی خام بیت‌کوین پیدا نشد: {path} — context بیت‌کوین خالی می‌ماند.")
        return pd.DataFrame()
    raw = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    ctx = build_btc_context(raw)
    logger.info(f"context بیت‌کوین ساخته شد: {len(ctx)} ردیف ({interval}).")
    return ctx


# ----------------------------------------------------------------------
# لیکوییدیشن — تبدیل سفارش‌های اجباری به ستون‌های هر کندل
# ----------------------------------------------------------------------
def liquidations_to_candles(orders: pd.DataFrame,
                            candle_index,
                            interval: str = "15m") -> pd.DataFrame:
    """
    سفارش‌های لیکوییدیشن (خروجی get_liquidation_orders / allForceOrders) را
    به مقدار دلاریِ لیکوییدشده در هر کندل تبدیل می‌کند.

    ستون side در پاسخ بایننس «سمت سفارشِ اجراشده» است:
      - SELL → پوزیشن لانگ لیکویید شده (برای بستن لانگ باید فروخت)
      - BUY  → پوزیشن شورت لیکویید شده (برای بستن شورت باید خرید)
    خروجی: liq_long_notional و liq_short_notional روی ایندکس کندل‌ها.
    """
    out = pd.DataFrame(0.0, index=candle_index,
                       columns=list(LIQUIDATION_RAW_COLUMNS))
    if orders is None or orders.empty or len(candle_index) == 0:
        return out

    return _aggregate_liquidations(orders, candle_index, interval)


def load_liquidation_candles(symbol: str, candle_index, interval: str = "15m",
                             liq_dir: str = "data_cache/liquidations") -> pd.DataFrame:
    """
    خواندن مخزن محلی لیکوییدیشن (که liquidations_collector.py می‌سازد) و تبدیل
    به دو ستون دلاریِ هر کندل: liq_long_notional / liq_short_notional.

    ⚠️ تاریخچه‌ی لیکوییدیشن از هیچ API عمومی قابل بازسازی نیست؛ این مخزن فقط
    از لحظه‌ای که collector اجرا شده داده دارد (به همین دلیل همراه
    build_dataset باید --tail-days داده شود).

    اگر فایل یا داده نبود، DataFrame خالی برمی‌گرداند (build_dataset هشدار می‌دهد).
    """
    import os  # فقط همین تابع به مسیر فایل نیاز دارد
    path = os.path.join(liq_dir, f"{symbol.upper()}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        orders = pd.read_csv(path, on_bad_lines="skip")
    except Exception as e:  # pragma: no cover — ممکن است فایل هم‌زمان در حال نوشتن باشد
        logger.warning(f"خواندن مخزن لیکوییدیشن {symbol} ناموفق بود: {e}")
        return pd.DataFrame()

    if orders.empty or not {"timestamp", "side", "notional"}.issubset(orders.columns):
        return pd.DataFrame()

    orders["timestamp"] = pd.to_datetime(orders["timestamp"], errors="coerce")
    orders = orders.dropna(subset=["timestamp"])
    if orders.empty:
        return pd.DataFrame()

    idx = pd.DatetimeIndex(candle_index)
    if len(idx) == 0:
        return pd.DataFrame()
    # فقط سفارش‌های داخل (و کمی اطراف) بازه‌ی کندل‌ها — سریع‌تر و بدون صفرِ اضافی
    span = pd.Timedelta(days=1)
    orders = orders[(orders["timestamp"] >= idx.min() - span)
                    & (orders["timestamp"] <= idx.max() + span)]
    if orders.empty:
        return pd.DataFrame()

    return liquidations_to_candles(orders, idx, interval)

def _aggregate_liquidations(orders, candle_index, interval):
    """تجمیع سفارش‌های لیکوییدیشن در سطل‌های کندلی (دلاری، به‌ازای هر کندل)."""
    out = pd.DataFrame(0.0, index=candle_index,
                       columns=list(LIQUIDATION_RAW_COLUMNS))
    delta = pd.to_timedelta(interval)
    ts = pd.to_datetime(orders["timestamp"]).dt.floor(delta)
    side = orders["side"].astype(str).str.upper()
    notional = pd.to_numeric(orders["notional"], errors="coerce").fillna(0.0)

    long_amt = notional.where(side == "SELL", 0.0)
    short_amt = notional.where(side == "BUY", 0.0)
    ag = pd.DataFrame({"liq_long_notional": long_amt.to_numpy(),
                       "liq_short_notional": short_amt.to_numpy()},
                      index=pd.DatetimeIndex(ts))
    ag = ag.groupby(level=0).sum()
    ag = ag.reindex(pd.DatetimeIndex(candle_index), fill_value=0.0)
    out["liq_long_notional"] = ag["liq_long_notional"].to_numpy()
    out["liq_short_notional"] = ag["liq_short_notional"].to_numpy()
    return out
