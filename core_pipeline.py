"""
core_pipeline.py
-----------------
منطق مرکزی و مشترک تحلیل: دریافت داده -> محاسبه اندیکاتورها -> تشخیص الگو ->
اسکورینگ -> مشورت با AI. هم main.py و هم multi_run.py از همین تابع استفاده
می‌کنن تا هیچ‌جا داده‌ی ساختگی (fake) به‌جای مقادیر واقعی فرستاده نشه.
"""

import logging
from data.binance_client import BinanceDataClient
from indicators.trend import TrendIndicators
from indicators.momentum import MomentumIndicators
from indicators.volatility import VolatilityIndicators
from indicators.volume import VolumeIndicators
from strategies.patterns import PatternDetector
from strategies.scoring import SignalScorer

logger = logging.getLogger("CorePipeline")


class AnalysisError(Exception):
    """خطای مشخص برای زمانی که تحلیل به هر دلیلی قابل انجام نیست."""
    pass


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

    score_result = SignalScorer.calculate_score(df)

    last_candle = df.iloc[-1]
    technical_summary = {
        "close": last_candle.get("close"),
        "rsi": round(last_candle.get("RSI", 50), 2) if last_candle.get("RSI") is not None else 50.0,
        "cmf": round(last_candle.get("CMF", 0), 4) if last_candle.get("CMF") is not None else 0.0,
        "adx": round(last_candle.get("ADX_14", 0), 2) if last_candle.get("ADX_14") is not None else 0.0,
    }

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "technical_summary": technical_summary,
        "score_result": score_result,
        "supports": supports,
        "resistances": resistances,
        "active_candle_patterns": active_candle_patterns,
        "recent_chart_patterns": recent_chart_patterns,
        "dataframe": df,  # برای بک‌تست یا تحلیل بیشتر در دسترسه
    }
