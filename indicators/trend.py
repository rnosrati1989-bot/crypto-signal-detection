import pandas as pd
import pandas_ta as ta
import logging

# تنظیم لاگر برای این ماژول
logger = logging.getLogger(__name__)

class TrendIndicators:
    @staticmethod
    def calculate_all_trend(df: pd.DataFrame) -> pd.DataFrame:
        """
        محاسبه تمام اندیکاتورهای روند بر روی DataFrame ورودی.
        دیتابیس ورودی باید شامل ستون‌های open, high, low, close, volume باشد.
        """
        if df.empty or len(df) < 50: # برای اندیکاتورهایی مثل ایچیموکو به حداقل داده نیاز داریم
            logger.warning("Dataframe is too short to calculate trend indicators.")
            return df
        
        try:
            # کپی از دیتابیس اصلی برای جلوگیری از تغییر در دیتای اولیه
            df_indicators = df.copy()

            # ۱. میانگین‌های متحرک (EMA Multi-timeframe)
            df_indicators['EMA_9'] = ta.ema(df_indicators['close'], length=9)
            df_indicators['EMA_21'] = ta.ema(df_indicators['close'], length=21)
            df_indicators['EMA_50'] = ta.ema(df_indicators['close'], length=50)
            df_indicators['EMA_200'] = ta.ema(df_indicators['close'], length=200)

            # ۲. اندیکاتور MACD
            macd_df = ta.macd(df_indicators['close'], fast=12, slow=26, signal=9)
            if macd_df is not None:
                df_indicators = pd.concat([df_indicators, macd_df], axis=1)

            # ۳. شاخص میانگین حرکت جهت‌دار (ADX + DMI)
            adx_df = ta.adx(df_indicators['high'], df_indicators['low'], df_indicators['close'], length=14)
            if adx_df is not None:
                df_indicators = pd.concat([df_indicators, adx_df], axis=1)

            # ۴. سوپرترند (SuperTrend)
            supertrend_df = ta.supertrend(df_indicators['high'], df_indicators['low'], df_indicators['close'], period=7, multiplier=3)
            if supertrend_df is not None:
                super_cols = {col: f"SUPERTREND_{col.split('_')[0]}" for col in supertrend_df.columns}
                supertrend_df.rename(columns=super_cols, inplace=True)
                df_indicators = pd.concat([df_indicators, supertrend_df], axis=1)

            # ۵. پارابولیک سار (Parabolic SAR)
            psar_df = ta.psar(df_indicators['high'], df_indicators['low'], df_indicators['close'], af=0.02, max_af=0.2)
            if psar_df is not None:
                df_indicators = pd.concat([df_indicators, psar_df], axis=1)

            # ۶. نرخ تغییرات قیمت (ROC)
            df_indicators['ROC'] = ta.roc(df_indicators['close'], length=12)

            # ۷. ابر ایچیموکو (Ichimoku Cloud)
            ichi_df, ichi_span = ta.ichimoku(df_indicators['high'], df_indicators['low'], df_indicators['close'])
            if ichi_df is not None:
                df_indicators = pd.concat([df_indicators, ichi_df], axis=1)

            logger.info("Trend indicators calculated successfully.")
            return df_indicators

        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
            return df