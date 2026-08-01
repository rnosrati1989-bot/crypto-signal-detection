import pandas as pd
import pandas_ta as ta
import logging

# تنظیم لاگر برای این ماژول
logger = logging.getLogger(__name__)

class VolatilityIndicators:
    @staticmethod
    def calculate_all_volatility(df: pd.DataFrame) -> pd.DataFrame:
        """
        محاسبه تمام اندیکاتورهای نوسان بر روی DataFrame ورودی.
        دیتابیس ورودی باید شامل ستون‌های high, low, close باشد.
        """
        if df.empty or len(df) < 30:
            logger.warning("Dataframe is too short to calculate volatility indicators.")
            return df
        
        try:
            # کپی از دیتابیس اصلی برای جلوگیری از تغییر در دیتای اولیه
            df_indicators = df.copy()

            # ۱. باندهای بولینگر (Bollinger Bands - 20, 2)
            # خروجی شامل باندهای بالایی (BBU)، میانی (BBM) و پایینی (BBL) است
            bb_df = ta.bbands(df_indicators['close'], length=20, std=2)
            if bb_df is not None:
                # تغییر نام ستون‌ها برای دسترسی آسان‌تر
                bb_df.columns = ['BB_LOW', 'BB_MID', 'BB_HIGH', 'BB_BANDWIDTH', 'BB_PERCENT']
                df_indicators = pd.concat([df_indicators, bb_df], axis=1)

            # ۲. کانال‌های کلتنر (Keltner Channels)
            # تفاوت این کانال با بولینگر در استفاده از ATR به جای انحراف معیار است
            kc_df = ta.kc(df_indicators['high'], df_indicators['low'], df_indicators['close'], length=20, scalar=2)
            if kc_df is not None:
                kc_df.columns = ['KC_LOW', 'KC_MID', 'KC_HIGH']
                df_indicators = pd.concat([df_indicators, kc_df], axis=1)

            # ۳. میانگین محدوده واقعی (ATR 14)
            # بهترین ابزار برای محاسبه نوسان مطلق و تنظیم فاصله Stop-Loss
            df_indicators['ATR'] = ta.atr(df_indicators['high'], df_indicators['low'], df_indicators['close'], length=14)

            # ۴. کانال‌های دونچیان (Donchian Channels)
            # بالاترین سقف و پایین‌ترین کف در دوره مشخص (عالی برای استراتژی‌های شکست سقف/کف)
            dc_df = ta.donchian(df_indicators['high'], df_indicators['low'], lower_length=20, upper_length=20)
            if dc_df is not None:
                dc_df.columns = ['DC_LOW', 'DC_MID', 'DC_HIGH']
                df_indicators = pd.with_columns(dc_df) if hasattr(pd, 'with_columns') else pd.concat([df_indicators, dc_df], axis=1)

            # ۵. پهنای باندهای بولینگر (BB Width) و درصد بی (%B)
            # اگر این مقدار محاسبه نشده باشد، مجدداً فرمول‌نویسی دستی می‌کنیم
            if 'BB_BANDWIDTH' not in df_indicators.columns and 'BB_MID' in df_indicators.columns:
                df_indicators['BB_BANDWIDTH'] = (df_indicators['BB_HIGH'] - df_indicators['BB_LOW']) / df_indicators['BB_MID']
            
            if 'BB_PERCENT' not in df_indicators.columns and 'BB_LOW' in df_indicators.columns:
                df_indicators['BB_PERCENT'] = (df_indicators['close'] - df_indicators['BB_LOW']) / (df_indicators['BB_HIGH'] - df_indicators['BB_LOW'])

            logger.info("Volatility indicators calculated successfully.")
            return df_indicators

        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            return df
