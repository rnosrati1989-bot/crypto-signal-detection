import numpy as np
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
            #    پیاده‌سازی دستی به‌جای ta.adx — در pandas_ta 0.4.71b0 مقدار
            #    k = 100/ATR در گرم‌آپ باعث DMP/DMN > 100 (غیرممکن) می‌شود.
            #    این پیاده‌سازی همان فرمول استاندارد را می‌سازد ولی DMI را به
            #    بازه‌ی منطقی [0, 100] محدود می‌کند (مثل گارد CMF).
            adx_length = 14
            adx_signal_length = 14       # signal_length = length (مثل کتابخانه)
            adxr_length = 2
            scalar = 100.0

            high_s = df_indicators['high']
            low_s = df_indicators['low']
            close_s = df_indicators['close']

            # True Range + ATR (Wilder/RMA: ewm با alpha = 1/طول)
            prev_close = close_s.shift(1)
            tr = pd.concat([
                high_s - low_s,
                (high_s - prev_close).abs(),
                (low_s - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_rma = tr.ewm(alpha=1.0 / adx_length, adjust=False).mean()

            # حرکت جهت‌دار مثبت/منفی (روش استاندارد Directional Movement)
            up = high_s - high_s.shift(1)
            dn = low_s.shift(1) - low_s
            pos = ((up > dn) & (up > 0)) * up
            neg = ((dn > up) & (dn > 0)) * dn
            pos = pos.clip(lower=0).fillna(0)
            neg = neg.clip(lower=0).fillna(0)

            # DMI (DMP/DMN) = 100 * RMA_{length}( movement ) / ATR
            # با clip به [0, 100] تا مقدار غیرممکن ساخته نشود
            with np.errstate(divide='ignore', invalid='ignore'):
                dmp = (scalar * pos.ewm(alpha=1.0 / adx_length, adjust=False).mean()
                       / atr_rma).clip(0, 100)
                dmn = (scalar * neg.ewm(alpha=1.0 / adx_length, adjust=False).mean()
                       / atr_rma).clip(0, 100)

                # ADX = 100 * |DMP - DMN| / (DMP + DMN) سپس RMA (سیگنال)
                denom = dmp + dmn
                dx = scalar * (dmp - dmn).abs() / denom.where(denom != 0)
                adx = dx.ewm(alpha=1.0 / adx_signal_length, adjust=False).mean()
                adxr = 0.5 * (adx + adx.shift(adxr_length))

            adx_df = pd.DataFrame({
                f"ADX_{adx_signal_length}": adx,
                f"ADXR_{adx_signal_length}_{adxr_length}": adxr,
                f"DMP_{adx_length}": dmp,
                f"DMN_{adx_length}": dmn,
            })
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