import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SignalScorer:
    @staticmethod
    def score_row(row, buy_threshold: float = 70, sell_threshold: float = 30) -> dict:
        """
        محاسبه نمره‌ی سیگنال برای یک ردیف (کندل) مشخص از دیتافریم.
        این متد جدا شده تا هم روی «آخرین کندل زنده» (calculate_score) و هم
        روی «تک‌تک کندل‌های تاریخی» (برای بک‌تست/بهینه‌سازی) قابل استفاده باشه.

        buy_threshold/sell_threshold قابل تنظیم‌ان تا در optimize_backtest.py
        بشه بهترین آستانه رو بر اساس داده‌ی تاریخی (train set) پیدا کرد، به‌جای
        این‌که این اعداد همیشه ثابت و دلبخواهی (۷۰ و ۳۰) باشن.
        """
        trend_score = 0
        momentum_score = 0
        volatility_score = 0
        volume_score = 0

        # ۱. بخش روند (وزن کل: ۴۰ نمره)
        if row.get('close') > row.get('EMA_50', 0): trend_score += 10
        if row.get('EMA_9', 0) > row.get('EMA_21', 0): trend_score += 10
        if row.get('SUPERTREND_SUPERT', 0) > 0: trend_score += 10
        if row.get('ADX_14', 0) > 25 and row.get('DMP_14', 0) > row.get('DMN_14', 0): trend_score += 10

        # ۲. بخش مومنتوم (وزن کل: ۳۰ نمره)
        rsi = row.get('RSI', 50)
        if rsi < 35: momentum_score += 15
        elif 50 < rsi < 70: momentum_score += 10

        stoch_k = row.get('STOCH_RSI_K', 50)
        stoch_d = row.get('STOCH_RSI_D', 50)
        if stoch_k > stoch_d and stoch_k < 30: momentum_score += 15

        # ۳. بخش نوسان (وزن کل: ۱۵ نمره)
        close = row.get('close', 0)
        bb_low = row.get('BB_LOW', 0)
        if close <= bb_low * 1.01:
            volatility_score += 15

        # ۴. بخش حجم (وزن کل: ۱۵ نمره)
        cmf = row.get('CMF', 0)
        if cmf > 0.05: volume_score += 15

        total_score = trend_score + momentum_score + volatility_score + volume_score

        direction = "HOLD"
        if total_score >= buy_threshold:
            direction = "BUY"
        elif total_score <= sell_threshold and total_score > 0:
            direction = "SELL"

        return {
            "score": total_score,
            "direction": direction,
            "details": {
                "trend": trend_score,
                "momentum": momentum_score,
                "volatility": volatility_score,
                "volume": volume_score
            }
        }

    @staticmethod
    def calculate_score(df: pd.DataFrame) -> dict:
        """
        محاسبه نمره سیگنال خرید/فروش نهایی بین 0 تا 100 برای آخرین کندل دیتافریم.
        (رفتار این متد کاملاً مثل قبل باقی مونده، فقط منطق داخلیش به score_row منتقل شده)
        """
        if df.empty:
            return {"score": 0, "direction": "HOLD", "details": {}}

        last_row = df.iloc[-1]
        return SignalScorer.score_row(last_row)
