import pandas as pd
import pandas_ta as ta
import logging

# تنظیم لاگر برای این ماژول
logger = logging.getLogger(__name__)

class MomentumIndicators:
    @staticmethod
    def calculate_all_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """
        محاسبه تمام اندیکاتورهای مومنتوم و اسیلاتورها بر روی DataFrame ورودی.
        دیتابیس ورودی باید شامل ستون‌های high, low, close, volume باشد.
        """
        if df.empty or len(df) < 30: # برای اکثر اسیلاتورها حداقل ۳۰ کندل نیاز داریم
            logger.warning("Dataframe is too short to calculate momentum indicators.")
            return df
        
        try:
            # کپی از دیتابیس اصلی برای جلوگیری از تغییر در دیتای اولیه
            df_indicators = df.copy()

            # ۱. شاخص قدرت نسبی (RSI 14)
            df_indicators['RSI'] = ta.rsi(df_indicators['close'], length=14)

            # ۲. استوکاستیک آر‌اس‌آی (Stochastic RSI)
            stoch_rsi = ta.stochrsi(df_indicators['close'], length=14, rsi_length=14, k=3, d=3)
            if stoch_rsi is not None:
                stoch_rsi.columns = ['STOCH_RSI_K', 'STOCH_RSI_D']
                df_indicators = pd.concat([df_indicators, stoch_rsi], axis=1)

            # ۳. شاخص کانال کالا (CCI 20)
            df_indicators['CCI'] = ta.cci(df_indicators['high'], df_indicators['low'], df_indicators['close'], length=20)

            # ۴. اسیلاتور آلتیمیت (Ultimate Oscillator)
            df_indicators['UO'] = ta.uo(df_indicators['high'], df_indicators['low'], df_indicators['close'])

            # ۵. شاخص ویلیامز (Williams %R)
            df_indicators['WILLR'] = ta.willr(df_indicators['high'], df_indicators['low'], df_indicators['close'], length=14)

            # ۶. اسیلاتور عالی (Awesome Oscillator)
            df_indicators['AO'] = ta.ao(df_indicators['high'], df_indicators['low'])

            # ۷. تریکس (TRIX)
            trix_df = ta.trix(df_indicators['close'], length=18, signal=9)
            if trix_df is not None:
                trix_df.columns = ['TRIX', 'TRIX_SIGNAL']
                df_indicators = pd.concat([df_indicators, trix_df], axis=1)

            # ۸. اسیلاتور KDJ
            kdj_df = ta.kdj(df_indicators['high'], df_indicators['low'], df_indicators['close'])
            if kdj_df is not None:
                kdj_df.columns = ['KDJ_K', 'KDJ_D', 'KDJ_J']
                df_indicators = pd.concat([df_indicators, kdj_df], axis=1)

            logger.info("Momentum indicators calculated successfully.")
            return df_indicators

        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return df