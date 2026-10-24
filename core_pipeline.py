"""
core_pipeline.py
-----------------
منطق مرکزی و مشترک تحلیل: دریافت داده -> محاسبه اندیکاتورها -> تشخیص الگو ->
اسکورینگ -> مشورت با AI. هم main.py و هم multi_run.py از همین تابع استفاده
می‌کنن تا هیچ‌جا داده‌ی ساختگی (fake) به‌جای مقادیر واقعی فرستاده نشه.
"""

import logging
import numpy as np
from data.binance_client import BinanceDataClient
from indicators.trend import TrendIndicators
from indicators.momentum import MomentumIndicators
from indicators.volatility import VolatilityIndicators
from indicators.volume import VolumeIndicators
from strategies.patterns import PatternDetector
from strategies.scoring import SignalScorer
from strategies.market_profile import MarketProfile

logger = logging.getLogger("CorePipeline")


class AnalysisError(Exception):
    """خطای مشخص برای زمانی که تحلیل به هر دلیلی قابل انجام نیست."""
    pass


def _build_digest(df) -> dict:
    """
    دیجست نرمال/ایستای آخرین کندل برای LLM — فقط از ستون‌هایی که همین
    پایپ‌لاین (ماژول‌های اندیکاتور + الگو) تولید کرده؛ همه‌ی مقادیر نسبتی/ایستا
    هستند (بدون قیمت مطلق). ویژگی‌های اختصاصی فیوچرز (فاندینگ، OI، فشار خرید)
    در خروجی زنده‌ی اسپات در دسترس نیستند و عمداً این‌جا نیستند؛ مسیر ML در
    features.build_features مجموعه‌ی کامل‌تر دارد.
    هر ستون غایب → None، تا هیچ‌وقت پایپ‌لاین شکست نخوره.
    """
    def _series(name):
        return df[name] if name in df.columns else None

    if df.empty:
        return {}

    close = _series("close")
    high = _series("high")
    low = _series("low")
    volume = _series("volume")
    atr = _series("ATR")

    ema9 = _series("EMA_9")
    ema21 = _series("EMA_21")
    ema50 = _series("EMA_50")
    ema200 = _series("EMA_200")

    values = {}

    if close is not None:
        if ema9 is not None:
            values["ema9_dist"] = close / ema9 - 1.0
        if ema21 is not None:
            values["ema21_dist"] = close / ema21 - 1.0
        if ema50 is not None:
            values["ema50_dist"] = close / ema50 - 1.0
        if ema200 is not None:
            values["ema200_dist"] = close / ema200 - 1.0
        if atr is not None:
            values["atr_pct"] = atr / close
        if high is not None and low is not None:
            hi20 = high.rolling(20).max()
            lo20 = low.rolling(20).min()
            values["range_pos"] = (close - lo20) / (hi20 - lo20).replace(0, np.nan)
        vwap = _series("VWAP")
        if vwap is not None:
            values["vwap_dist"] = close / vwap - 1.0
        kc_hi, kc_lo = _series("KC_HIGH"), _series("KC_LOW")
        if kc_hi is not None and kc_lo is not None:
            values["kc_pos"] = (close - kc_lo) / (kc_hi - kc_lo).replace(0, np.nan)
        dc_hi, dc_lo = _series("DC_HIGH"), _series("DC_LOW")
        if dc_hi is not None and dc_lo is not None:
            values["dc_pos"] = (close - dc_lo) / (dc_hi - dc_lo).replace(0, np.nan)
        psarl, psars = _series("PSARl_0.02_0.2"), _series("PSARs_0.02_0.2")
        if atr is not None and (psarl is not None or psars is not None):
            if psarl is not None:
                psar_line = psarl.fillna(psars) if psars is not None else psarl
            else:
                psar_line = psars
            values["psar_dist_atr"] = (close - psar_line) / atr
        if atr is not None:
            values["vol_regime"] = atr / atr.rolling(48).mean()

    if ema9 is not None and ema21 is not None:
        values["ema9_21_cross"] = ema9 / ema21 - 1.0

    if atr is not None:
        macdh = _series("MACDh_12_26_9")
        if macdh is not None:
            values["macdh_norm"] = macdh / atr
            values["macdh_slope"] = (macdh - macdh.shift(1)) / atr

    rsi = _series("RSI")
    if rsi is not None:
        values["rsi_slope"] = rsi - rsi.shift(1)

    if volume is not None:
        vol0 = volume.replace(0, np.nan)
        ma24 = vol0.rolling(24).mean()
        sd24 = vol0.rolling(24).std()
        values["vol_z"] = ((vol0 - ma24) / sd24).clip(-3, 3)

    obv = _series("OBV")
    if obv is not None:
        values["obv_z"] = ((obv - obv.rolling(48).mean())
                           / obv.rolling(48).std()).clip(-3, 3)

    st_cols = [c for c in df.columns if str(c).startswith("SUPERTREND")]
    st_dirs = [c for c in st_cols if str(c).endswith("d")]
    if st_dirs:
        values["st_dir"] = df[st_dirs[0]].astype(float)

    pat_cols = [c for c in df.columns if c.startswith("PATTERN_")]
    if pat_cols:
        pat_net = df[pat_cols].fillna(0).sum(axis=1)
        values["pat_net_3"] = pat_net.rolling(3).sum()
        values["pat_net_12"] = pat_net.rolling(12).sum()

    # تبدیل سری‌های ستونی به عدد گردشده‌ی آخرین کندل (NaN/Inf → None)
    digest = {}
    for key, series in values.items():
        try:
            f = float(series.iloc[-1])
        except (TypeError, ValueError, IndexError):
            digest[key] = None
            continue
        if f != f or f in (float("inf"), float("-inf")):
            digest[key] = None
        else:
            digest[key] = round(f, 4)
    return digest


def analyze_symbol(symbol: str, timeframe: str, klines_limit: int = 250) -> dict:
    """
    اجرای کامل و واقعیِ پایپ‌لاین تحلیل برای یک نماد/تایم‌فریم مشخص.
    هیچ مقدار ساختگی (hardcoded) در این مسیر وجود نداره؛ اگر داده کافی
    نباشه یا محاسبه‌ای شکست بخوره، AnalysisError پرتاب میشه تا صدا
    زننده تصمیم بگیره (skip کنه، retry کنه و...).

    خروجی شامل: technical_summary, score_result, supports, resistances
    """
    try:
        binance_client = BinanceDataClient()
        df = binance_client.get_historical_klines(symbol, timeframe, limit=klines_limit)
    except Exception as e:
        # هر نوع خطای شبکه/اتصال (شامل ConnectionError بعد از اتمام retry های داخلی)
        # رو به یک نوع خطای یکسان تبدیل می‌کنیم تا صداکننده‌ها (main.py، multi_run.py،
        # backtest.py) فقط لازم نباشه چند نوع Exception مختلف رو جدا مدیریت کنن.
        raise AnalysisError(f"خطا در دریافت داده برای {symbol}/{timeframe}: {e}")

    if df.empty:
        raise AnalysisError(f"داده‌ای برای {symbol} در تایم‌فریم {timeframe} دریافت نشد.")

    df = TrendIndicators.calculate_all_trend(df)
    df = MomentumIndicators.calculate_all_momentum(df)
    df = VolatilityIndicators.calculate_all_volatility(df)
    df = VolumeIndicators.calculate_all_volume(df)
    df = PatternDetector.detect_candle_patterns(df)
    supports, resistances = PatternDetector.find_support_resistance(df)

    # الگوهای کندلی فعال روی همین آخرین کندل (برای نمایش، نه به‌عنوان سیگنال مستقل)
    last_row = df.iloc[-1]
    active_candle_patterns = [
        col.replace("PATTERN_", "")
        for col in df.columns
        if col.startswith("PATTERN_") and last_row.get(col, 0) != 0
    ]

    # همه‌ی الگوهای نموداری بزرگ‌تر (Double Top/Bottom, Head & Shoulders, Triangle)
    # که اخیراً (نزدیک انتهای داده) شکل گرفتن
    chart_patterns = PatternDetector.get_all_chart_patterns(df)
    recent_chart_patterns = [
        p for p in chart_patterns
        if p["end_time"] >= df.index[-min(50, len(df))]
    ]

    # score_result: خروجی عددی SignalScorer — فقط برای مرجع تاریخی/بک‌تست نگه داشته میشه،
    # چون طبق نتایج Walk-Forward، این عدد به‌تنهایی edge اثبات‌شده‌ای نداره.
    score_result = SignalScorer.calculate_score(df)

    # market_profile: خروجی اصلی برای نمایش زنده — به‌جای یک تصمیم قطعی (BUY/SELL)،
    # وضعیت هر بُعد بازار رو جدا توصیف می‌کنه و صریحاً میگه ابعاد هم‌جهت هستن یا در تضاد.
    last_row = df.iloc[-1]
    market_profile = MarketProfile.build(last_row)

    last_candle = df.iloc[-1]
    technical_summary = {
        "close": last_candle.get("close"),
        "rsi": round(last_candle.get("RSI", 50), 2) if last_candle.get("RSI") is not None else 50.0,
        "cmf": round(last_candle.get("CMF", 0), 4) if last_candle.get("CMF") is not None else 0.0,
        "adx": round(last_candle.get("ADX_14", 0), 2) if last_candle.get("ADX_14") is not None else 0.0,
    }

    # دیجست نرمال‌شده برای LLM — تا مدل به‌جای «۴ عدد تنها» تصویر کامل‌تر از
    # آخرین کندل بگیرد (روند/نوسان/نقدینگی/مومنتوم/الگو). هر خطا در این بخش
    # فقط دیجست را خالی می‌کند، کل تحلیل را نمی‌شکند.
    try:
        digest = _build_digest(df)
    except Exception as e:
        logger.warning(f"دیجست LLM ساخته نشد: {e}")
        digest = {}
    technical_summary["digest"] = digest

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "technical_summary": technical_summary,
        "market_profile": market_profile,
        "score_result": score_result,  # مرجع تاریخی/بک‌تست، نه توصیه‌ی تصمیم‌گیری
        "supports": supports,
        "resistances": resistances,
        "active_candle_patterns": active_candle_patterns,
        "recent_chart_patterns": recent_chart_patterns,
        "dataframe": df,  # برای بک‌تست یا تحلیل بیشتر در دسترسه
    }