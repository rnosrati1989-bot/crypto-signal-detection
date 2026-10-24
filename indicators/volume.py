import pandas as pd
import pandas_ta as ta
import logging

# تنظیم لاگر برای این ماژول
logger = logging.getLogger(__name__)

class VolumeIndicators:
    @staticmethod
    def calculate_all_volume(df: pd.DataFrame) -> pd.DataFrame:
        """
        محاسبه تمام اندیکاتورهای حجم و جریان پول بر روی DataFrame ورودی.
        دیتابیس ورودی باید شامل ستون‌های open, high, low, close, volume باشد.
        """
        if df.empty or len(df) < 20:
            logger.warning("Dataframe is too short to calculate volume indicators.")
            return df
        
        try:
            # کپی از دیتابیس اصلی برای جلوگیری از تغییر در دیتای اولیه
            df_indicators = df.copy()

            # ۱. حجم تعادلی (On-Balance Volume - OBV)
            # برای تشخیص اینکه آیا حجم در روزهای مثبت بیشتر است یا روزهای منفی
            df_indicators['OBV'] = ta.obv(df_indicators['close'], df_indicators['volume'])

            # ۲. شاخص جریان نقدینگی چایکین (Chaikin Money Flow - CMF)
            # جریان ورود و خروج پول را در یک بازه مشخص (معمولاً ۲۰ روزه) اندازه می‌گیرد
            #
            # ⚠️ نکته‌ی مهم: از تابع آماده‌ی ta.cmf() استفاده نمی‌کنیم چون یک باگ شناخته‌شده
            # داره — وقتی high == low بشه (کندل با رنج صفر، مخصوصاً روی تایم‌فریم‌های پایین)،
            # فرمول داخلیش تقسیم‌بر-صفر تقریبی می‌ده و مقادیر عظیم و بی‌معنی (مثلاً 1e19)
            # تولید می‌کنه، درحالی‌که CMF باید همیشه بین -1 و +1 باشه.
            # به‌جاش خودمون با گارد صریح در برابر رنج صفر محاسبه‌ش می‌کنیم:
            high_low_range = df_indicators['high'] - df_indicators['low']
            money_flow_multiplier = (
                (df_indicators['close'] - df_indicators['low']) -
                (df_indicators['high'] - df_indicators['close'])
            ) / high_low_range.replace(0, pd.NA)  # وقتی رنج صفره، multiplier نامعتبره
            money_flow_multiplier = money_flow_multiplier.fillna(0)  # کندل بدون رنج = بدون اطلاعات جهت‌دار، نه بی‌نهایت

            money_flow_volume = money_flow_multiplier * df_indicators['volume']
            cmf_raw = (
                money_flow_volume.rolling(window=20).sum() /
                df_indicators['volume'].rolling(window=20).sum().replace(0, pd.NA)
            )
            # اطمینان نهایی: CMF ریاضاً باید همیشه در بازه‌ی [-1, 1] باشه؛ هر مقدار خارج از این
            # بازه (مثلاً به‌خاطر داده‌ی ناقص) رو کلیپ می‌کنیم تا هیچ‌وقت به AI/کاربر عدد
            # بی‌معنی نمایش داده نشه.
            df_indicators['CMF'] = cmf_raw.clip(lower=-1, upper=1)

            # ۳. شاخص جریان پول (Money Flow Index - MFI)
            # همان RSI است، اما حجم معاملات را هم در فرمول خود دخالت می‌دهد (RSI وزنی با حجم)
            df_indicators['MFI'] = ta.mfi(
                df_indicators['high'], df_indicators['low'], 
                df_indicators['close'], df_indicators['volume'], length=14
            )

            # ۴. قیمت میانگین وزنی حجمی (VWAP)
            # یکی از مهم‌ترین خطوط تعادلی برای معامله‌گران نهادی و الگوریتمی
            # توجه: VWAP معمولاً روی داده‌های درون‌روزی (Intraday) بهترین کارایی را دارد
            try:
                df_indicators['VWAP'] = ta.vwap(
                    df_indicators['high'], df_indicators['low'], 
                    df_indicators['close'], df_indicators['volume']
                )
            except Exception as e:
                logger.warning(f"Could not calculate VWAP (commonly needs intraday timestamps): {str(e)}")
                # یک تقریب ساده از VWAP در صورت بروز خطای کتابخانه
                typical_price = (df_indicators['high'] + df_indicators['low'] + df_indicators['close']) / 3
                df_indicators['VWAP'] = (typical_price * df_indicators['volume']).cumsum() / df_indicators['volume'].cumsum()

            # ۵. نرخ تغییرات حجم (Volume Rate of Change - VROC)
            # سرعت شتاب ورود و خروج حجم معاملات را بررسی می‌کند
            df_indicators['VROC'] = df_indicators['volume'].pct_change(periods=14) * 100

            logger.info("Volume indicators calculated successfully.")
            return df_indicators

        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return df
