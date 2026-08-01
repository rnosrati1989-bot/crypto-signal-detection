import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PatternDetector:
    @staticmethod
    def detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        تشخیص الگوهای کندل‌استیک به‌صورت برداری (vectorized، نه حلقه‌ی for) روی کل دیتافریم.
        هر الگو یک ستون PATTERN_<NAME> میسازه: 1 = صعودی (bullish)، -1 = نزولی (bearish)، 0 = هیچ.

        این الگوها صرفاً «برچسب» هستن، نه سیگنال خرید/فروش مستقل — برای این‌که
        کاربر/AI بفهمه چه شکلی از کندل‌ها اخیراً دیده شده، نه این‌که خودکار
        معامله باز کنه.
        """
        if df.empty or len(df) < 5:
            return df

        d = df.copy()

        o, h, l, c = d['open'], d['high'], d['low'], d['close']
        body = (c - o).abs()
        candle_range = (h - l).replace(0, 1e-9)
        upper_shadow = h - d[['open', 'close']].max(axis=1)
        lower_shadow = d[['open', 'close']].min(axis=1) - l
        is_bullish = c > o
        is_bearish = c < o
        body_pct = body / candle_range

        # مقادیر کندل قبلی/دو کندل قبلی (برای الگوهای چند-کندلی)
        o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
        body1 = (c1 - o1).abs()
        is_bullish1 = c1 > o1
        is_bearish1 = c1 < o1

        o2, c2 = o.shift(2), c.shift(2)
        body2 = (c2 - o2).abs()
        is_bearish2 = c2 < o2
        is_bullish2 = c2 > o2

        # ---------- الگوهای تک‌کندلی ----------

        # ۱. Doji: بدنه‌ی خیلی کوچیک نسبت به کل رنج کندل
        doji = (body_pct < 0.1)

        # ۲. Hammer (صعودی، معمولاً بعد از روند نزولی): سایه‌ی پایینی بلند، بدنه بالا
        hammer = (lower_shadow > 2 * body) & (upper_shadow < 0.5 * body) & (body > 0)

        # ۳. Hanging Man: از نظر شکل دقیقاً مثل Hammer، ولی وقتی بعد از روند صعودی بیاد نزولیه
        #    (چون تشخیص "روند قبلی" نیاز به context بیشتری داره، همون شکل رو با پرچم جدا نگه می‌داریم)
        hanging_man_shape = hammer  # شکل هندسی یکسانه؛ جهت واقعی به context بازار بستگی داره

        # ۴. Inverted Hammer / Shooting Star: سایه‌ی بالایی بلند، بدنه پایین
        inverted_shape = (upper_shadow > 2 * body) & (lower_shadow < 0.5 * body) & (body > 0)

        # ۵. Marubozu: تقریباً بدون سایه (بدنه تقریباً کل رنج کندل رو پر کرده)
        marubozu_bull = (body_pct > 0.95) & is_bullish
        marubozu_bear = (body_pct > 0.95) & is_bearish

        # ۶. Spinning Top: بدنه‌ی کوچیک با سایه‌های تقریباً مساوی و قابل‌توجه از دو طرف
        spinning_top = (body_pct > 0.1) & (body_pct < 0.3) & \
                       (upper_shadow > body) & (lower_shadow > body)

        # ---------- الگوهای دو-کندلی ----------

        # ۷. Bullish/Bearish Engulfing
        bullish_engulfing = is_bullish & is_bearish1 & (c >= o1) & (o <= c1)
        bearish_engulfing = is_bearish & is_bullish1 & (c <= o1) & (o >= c1)

        # ۸. Bullish/Bearish Harami: کندل دوم کاملاً داخل بدنه‌ی کندل اول جا میشه (برعکس Engulfing)
        bullish_harami = is_bullish & is_bearish1 & (c <= o1) & (o >= c1) & (body < body1)
        bearish_harami = is_bearish & is_bullish1 & (c >= o1) & (o <= c1) & (body < body1)

        # ۹. Piercing Line (صعودی): کندل اول نزولی قوی، کندل دوم صعودی و بیشتر از نیمه‌ی بدنه‌ی اول رو می‌پوشونه
        midpoint1 = (o1 + c1) / 2
        piercing_line = is_bearish1 & is_bullish & (o < c1) & (c > midpoint1) & (c < o1)

        # ۱۰. Dark Cloud Cover (نزولی): برعکس Piercing Line
        dark_cloud_cover = is_bullish1 & is_bearish & (o > c1) & (c < midpoint1) & (c > o1)

        # ---------- الگوهای سه‌کندلی ----------

        # ۱۱. Morning Star (صعودی): نزولی قوی -> کندل کوچیک (تردید) -> صعودی قوی
        small_body_mid = body1 < (body2 * 0.5)
        morning_star = is_bearish2 & small_body_mid & is_bullish & (c > (o2 + c2) / 2)

        # ۱۲. Evening Star (نزولی): برعکس Morning Star
        evening_star = is_bullish2 & small_body_mid & is_bearish & (c < (o2 + c2) / 2)

        # ۱۳. Three White Soldiers (صعودی قوی): سه کندل صعودی پیاپی، هرکدوم بالاتر از قبلی بسته میشه
        three_white_soldiers = is_bullish & is_bullish1 & is_bullish2 & (c > c1) & (c1 > c2)

        # ۱۴. Three Black Crows (نزولی قوی): برعکس Three White Soldiers
        three_black_crows = is_bearish & is_bearish1 & is_bearish2 & (c < c1) & (c1 < c2)

        def set_pattern(name, bull_cond=None, bear_cond=None, shape_cond=None):
            col = f"PATTERN_{name}"
            d[col] = 0
            if shape_cond is not None:
                d.loc[shape_cond.fillna(False), col] = 1  # فقط شکل هندسی، بدون جهت قطعی
            if bull_cond is not None:
                d.loc[bull_cond.fillna(False), col] = 1
            if bear_cond is not None:
                d.loc[bear_cond.fillna(False), col] = -1
            return col

        set_pattern("DOJI", shape_cond=doji)
        set_pattern("HAMMER", shape_cond=hammer)
        set_pattern("HANGING_MAN_SHAPE", shape_cond=hanging_man_shape)
        set_pattern("INVERTED_HAMMER_OR_SHOOTING_STAR", shape_cond=inverted_shape)
        set_pattern("MARUBOZU", bull_cond=marubozu_bull, bear_cond=marubozu_bear)
        set_pattern("SPINNING_TOP", shape_cond=spinning_top)
        set_pattern("ENGULFING", bull_cond=bullish_engulfing, bear_cond=bearish_engulfing)
        set_pattern("HARAMI", bull_cond=bullish_harami, bear_cond=bearish_harami)
        set_pattern("PIERCING_LINE", bull_cond=piercing_line)
        set_pattern("DARK_CLOUD_COVER", bear_cond=dark_cloud_cover)
        set_pattern("MORNING_STAR", bull_cond=morning_star)
        set_pattern("EVENING_STAR", bear_cond=evening_star)
        set_pattern("THREE_WHITE_SOLDIERS", bull_cond=three_white_soldiers)
        set_pattern("THREE_BLACK_CROWS", bear_cond=three_black_crows)

        logger.info("Candlestick patterns detected (14 patterns, vectorized).")
        return d

    @staticmethod
    def find_support_resistance(df: pd.DataFrame, window: int = 20) -> tuple:
        """
        یافتن سطوح حمایت و مقاومت کلیدی بر اساس سقف و کف‌های تاریخی (Pivots)
        """
        if len(df) < window * 2:
            return [], []

        supports = []
        resistances = []

        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == df['low'].iloc[i-window: i+window].min():
                supports.append(df['low'].iloc[i])
            if df['high'].iloc[i] == df['high'].iloc[i-window: i+window].max():
                resistances.append(df['high'].iloc[i])

        supports = sorted(list(set(np.round(supports, 2))))[-5:]
        resistances = sorted(list(set(np.round(resistances, 2))))[:5]

        return supports, resistances

    @staticmethod
    def find_swing_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        شناسایی نقاط چرخش محلی (Swing High / Swing Low) — پایه و اساس تشخیص
        الگوهای نموداری بزرگ‌تر مثل Double Top/Bottom و Head & Shoulders.

        یک کندل، Swing High محسوب میشه اگر سقفش از `window` کندل قبل و بعدش بیشتر باشه.
        خروجی: دیتافریمی با ستون‌های SWING_HIGH و SWING_LOW (NaN یا مقدار قیمت).
        """
        d = df.copy()
        d['SWING_HIGH'] = np.nan
        d['SWING_LOW'] = np.nan

        if len(d) < window * 2 + 1:
            return d

        highs = d['high'].values
        lows = d['low'].values

        for i in range(window, len(d) - window):
            local_high = highs[i - window: i + window + 1]
            local_low = lows[i - window: i + window + 1]
            if highs[i] == local_high.max():
                d.iloc[i, d.columns.get_loc('SWING_HIGH')] = highs[i]
            if lows[i] == local_low.min():
                d.iloc[i, d.columns.get_loc('SWING_LOW')] = lows[i]

        return d

    @staticmethod
    def detect_double_top_bottom(df: pd.DataFrame, swing_window: int = 5, tolerance_pct: float = 0.015) -> list:
        """
        تشخیص ساده‌ی الگوی Double Top / Double Bottom بر اساس دو Swing High (یا Low)
        متوالی که تقریباً هم‌سطح‌ان (در حدود tolerance_pct از هم).

        خروجی: لیستی از دیکشنری‌ها با نوع الگو، زمان دو نقطه، و قیمت.
        این یک ابزار تشخیصیه (برای نمایش/تحلیل)، نه سیگنال معاملاتی مستقل.
        """
        d = PatternDetector.find_swing_points(df, window=swing_window)
        patterns = []

        swing_highs = d[d['SWING_HIGH'].notna()][['SWING_HIGH']].reset_index()
        swing_lows = d[d['SWING_LOW'].notna()][['SWING_LOW']].reset_index()

        # Double Top: دو سقف متوالی نزدیک به هم
        for i in range(len(swing_highs) - 1):
            p1 = swing_highs.iloc[i]['SWING_HIGH']
            p2 = swing_highs.iloc[i + 1]['SWING_HIGH']
            if abs(p1 - p2) / p1 <= tolerance_pct:
                patterns.append({
                    "pattern": "DOUBLE_TOP",
                    "direction": "bearish",
                    "point1_time": swing_highs.iloc[i]['timestamp'],
                    "point2_time": swing_highs.iloc[i + 1]['timestamp'],
                    "end_time": swing_highs.iloc[i + 1]['timestamp'],
                    "price": round((p1 + p2) / 2, 2),
                })

        # Double Bottom: دو کف متوالی نزدیک به هم
        for i in range(len(swing_lows) - 1):
            p1 = swing_lows.iloc[i]['SWING_LOW']
            p2 = swing_lows.iloc[i + 1]['SWING_LOW']
            if abs(p1 - p2) / p1 <= tolerance_pct:
                patterns.append({
                    "pattern": "DOUBLE_BOTTOM",
                    "direction": "bullish",
                    "point1_time": swing_lows.iloc[i]['timestamp'],
                    "point2_time": swing_lows.iloc[i + 1]['timestamp'],
                    "end_time": swing_lows.iloc[i + 1]['timestamp'],
                    "price": round((p1 + p2) / 2, 2),
                })

        return patterns

    @staticmethod
    def detect_head_and_shoulders(df: pd.DataFrame, swing_window: int = 5, shoulder_tolerance_pct: float = 0.04) -> list:
        """
        تشخیص الگوی Head & Shoulders (سقف، نزولی) و نسخه‌ی معکوسش Inverse H&S (کف، صعودی).

        شرط: سه Swing متوالی که وسطی‌ (Head) به‌وضوح از دو کناری (Shoulders) بلندتر/کوتاه‌تر
        باشه، و دو Shoulder تقریباً هم‌سطح باشن (در حدود shoulder_tolerance_pct از هم).

        خروجی: لیستی از دیکشنری‌ها با نوع الگو، سه نقطه‌ی زمانی، و قیمت Head — صرفاً برای
        نمایش/تحلیل، نه سیگنال معاملاتی مستقل.
        """
        d = PatternDetector.find_swing_points(df, window=swing_window)
        patterns = []

        swing_highs = d[d['SWING_HIGH'].notna()][['SWING_HIGH']].reset_index()
        swing_lows = d[d['SWING_LOW'].notna()][['SWING_LOW']].reset_index()

        # Head & Shoulders (نزولی): سه سقف متوالی، وسطی از هر دو کناری بلندتر
        for i in range(len(swing_highs) - 2):
            left = swing_highs.iloc[i]['SWING_HIGH']
            head = swing_highs.iloc[i + 1]['SWING_HIGH']
            right = swing_highs.iloc[i + 2]['SWING_HIGH']
            if head > left and head > right and abs(left - right) / left <= shoulder_tolerance_pct:
                patterns.append({
                    "pattern": "HEAD_AND_SHOULDERS",
                    "direction": "bearish",
                    "left_shoulder_time": swing_highs.iloc[i]['timestamp'],
                    "head_time": swing_highs.iloc[i + 1]['timestamp'],
                    "right_shoulder_time": swing_highs.iloc[i + 2]['timestamp'],
                    "end_time": swing_highs.iloc[i + 2]['timestamp'],
                    "price": round(head, 2),
                })

        # Inverse Head & Shoulders (صعودی): سه کف متوالی، وسطی از هر دو کناری پایین‌تر
        for i in range(len(swing_lows) - 2):
            left = swing_lows.iloc[i]['SWING_LOW']
            head = swing_lows.iloc[i + 1]['SWING_LOW']
            right = swing_lows.iloc[i + 2]['SWING_LOW']
            if head < left and head < right and abs(left - right) / left <= shoulder_tolerance_pct:
                patterns.append({
                    "pattern": "INVERSE_HEAD_AND_SHOULDERS",
                    "direction": "bullish",
                    "left_shoulder_time": swing_lows.iloc[i]['timestamp'],
                    "head_time": swing_lows.iloc[i + 1]['timestamp'],
                    "right_shoulder_time": swing_lows.iloc[i + 2]['timestamp'],
                    "end_time": swing_lows.iloc[i + 2]['timestamp'],
                    "price": round(head, 2),
                })

        return patterns

    @staticmethod
    def detect_triangles(df: pd.DataFrame, swing_window: int = 5, lookback_points: int = 4,
                          flat_slope_tolerance: float = 0.0008) -> list:
        """
        تشخیص الگوهای مثلثی بر اساس شیب خط‌روند سقف‌ها و کف‌های اخیر:
        - Ascending Triangle (صعودی): سقف‌ها تقریباً صاف (مقاومت افقی)، کف‌ها صعودی
        - Descending Triangle (نزولی): کف‌ها تقریباً صاف (حمایت افقی)، سقف‌ها نزولی
        - Symmetrical Triangle (خنثی/فشردگی): سقف‌ها نزولی و کف‌ها صعودی (همگرایی)

        شیب‌ها با یک رگرسیون خطی ساده (numpy.polyfit) روی آخرین `lookback_points`
        Swing High/Low محاسبه میشن. flat_slope_tolerance به‌صورت نسبی (شیب / میانگین
        قیمت) سنجیده میشه تا مستقل از مقیاس قیمت (BTC در برابر یک آلت‌کوین ارزون) باشه.

        خروجی: صفر یا یک دیکشنری (آخرین/جاری‌ترین مثلث)، چون این الگو معمولاً روی
        بازه‌ی اخیر معنی داره، نه در طول کل تاریخچه.
        """
        d = PatternDetector.find_swing_points(df, window=swing_window)

        swing_highs = d[d['SWING_HIGH'].notna()][['SWING_HIGH']].reset_index().tail(lookback_points)
        swing_lows = d[d['SWING_LOW'].notna()][['SWING_LOW']].reset_index().tail(lookback_points)

        if len(swing_highs) < lookback_points or len(swing_lows) < lookback_points:
            return []

        avg_price = df['close'].tail(50).mean()

        # از timestamp به عدد ترتیبی (0,1,2,...) تبدیل می‌کنیم چون polyfit با تاریخ کار نمی‌کنه
        high_x = np.arange(len(swing_highs))
        high_slope = np.polyfit(high_x, swing_highs['SWING_HIGH'].values, 1)[0] / avg_price

        low_x = np.arange(len(swing_lows))
        low_slope = np.polyfit(low_x, swing_lows['SWING_LOW'].values, 1)[0] / avg_price

        high_flat = abs(high_slope) <= flat_slope_tolerance
        low_flat = abs(low_slope) <= flat_slope_tolerance

        pattern_name, direction = None, None
        if high_flat and low_slope > flat_slope_tolerance:
            pattern_name, direction = "ASCENDING_TRIANGLE", "bullish"
        elif low_flat and high_slope < -flat_slope_tolerance:
            pattern_name, direction = "DESCENDING_TRIANGLE", "bearish"
        elif high_slope < -flat_slope_tolerance and low_slope > flat_slope_tolerance:
            pattern_name, direction = "SYMMETRICAL_TRIANGLE", "neutral (بستگی به جهت breakout داره)"

        if pattern_name is None:
            return []

        return [{
            "pattern": pattern_name,
            "direction": direction,
            "start_time": swing_highs.iloc[0]['timestamp'],
            "end_time": max(swing_highs.iloc[-1]['timestamp'], swing_lows.iloc[-1]['timestamp']),
            "price": round(df['close'].iloc[-1], 2),
        }]

    @staticmethod
    def get_all_chart_patterns(df: pd.DataFrame, swing_window: int = 5) -> list:
        """
        اجرای همه‌ی تشخیص‌دهنده‌های الگوی نموداری (Double Top/Bottom، Head & Shoulders،
        Triangle) و برگردوندن یک لیست ترکیبی — برای این‌که core_pipeline فقط یک تابع
        رو صدا بزنه و نگران اضافه‌شدن الگوی جدید در آینده نباشه.
        """
        patterns = []
        patterns.extend(PatternDetector.detect_double_top_bottom(df, swing_window=swing_window))
        patterns.extend(PatternDetector.detect_head_and_shoulders(df, swing_window=swing_window))
        patterns.extend(PatternDetector.detect_triangles(df, swing_window=swing_window))
        return patterns
