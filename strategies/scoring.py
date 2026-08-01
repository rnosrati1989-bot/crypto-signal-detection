import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RegimeFilter:
    """
    فیلتر رژیم بازار: این کلاس تشخیص می‌دهد که آیا شرایط بازار برای معامله مناسب است یا نه.
    
    بر اساس نتایج Walk-Forward، سیستم فقط در شرایط خاصی edge داره:
    - روند قوی (ADX > 25)
    - نوسان کافی (ATR > 1%, BB Width > 0.015)
    - هم‌راستایی نسبی EMA ها (حداقل 2 از 3)
    
    نسخه 2: فیلترهای انعطاف‌پذیرتر (حذف VROC و CMF که خیلی سخت بودن)
    """
    
    @staticmethod
    def check_adx_filter(row, min_adx: float = 25.0) -> tuple:
        """
        فیلتر ADX: فقط روندهای قوی
        
        Returns:
            (passed: bool, reason: str)
        """
        adx = row.get('ADX_14', 0)
        if adx < min_adx:
            return False, f"ADX={adx:.1f} < {min_adx} (روند ضعیف)"
        return True, f"ADX={adx:.1f} ✓"
    
    @staticmethod
    def check_volatility_expansion(row) -> tuple:
        """
        فیلتر نوسان: ATR نباید خیلی کم باشه
        
        Returns:
            (passed: bool, reason: str)
        """
        bb_width = row.get('BB_BANDWIDTH', 0)
        atr = row.get('ATR', 0)
        close = row.get('close', 1)
        atr_pct = (atr / close * 100) if close > 0 else 0
        
        # شرایط تسهیل‌شده: یکی از دو شرط کافیه
        if bb_width < 0.010:  # کاهش از 0.015 به 0.010
            if atr_pct < 0.8:  # کاهش از 1.0 به 0.8
                return False, f"BB_Width={bb_width:.4f} و ATR={atr_pct:.2f}% (هر دو خیلی کم)"
        
        return True, f"BB_Width={bb_width:.4f}, ATR={atr_pct:.2f}% ✓"
    
    @staticmethod
    def check_trend_alignment(row, regime_info: dict) -> tuple:
        """
        فیلتر هم‌راستایی EMA: حداقل 2 از 3 EMA باید هم‌راستا باشن (تسهیل‌شده)
        
        Returns:
            (passed: bool, reason: str, allowed_direction: str)
        """
        ema_9 = row.get('EMA_9', 0)
        ema_21 = row.get('EMA_21', 0)
        ema_50 = row.get('EMA_50', 0)
        close = row.get('close', 0)
        
        # چک کردن ترتیب کامل (ایده‌آل)
        perfect_bullish = ema_9 > ema_21 > ema_50 and close > ema_50
        perfect_bearish = ema_9 < ema_21 < ema_50 and close < ema_50
        
        if perfect_bullish:
            return True, "EMA Perfect Bullish ✓", "BUY"
        elif perfect_bearish:
            return True, "EMA Perfect Bearish ✓", "SELL"
        
        # چک کردن هم‌راستایی جزئی (حداقل 2 از 3)
        bullish_count = sum([
            ema_9 > ema_21,
            ema_21 > ema_50,
            close > ema_21
        ])
        
        bearish_count = sum([
            ema_9 < ema_21,
            ema_21 < ema_50,
            close < ema_21
        ])
        
        if bullish_count >= 2:
            return True, f"EMA Partial Bullish ({bullish_count}/3) ✓", "BUY"
        elif bearish_count >= 2:
            return True, f"EMA Partial Bearish ({bearish_count}/3) ✓", "SELL"
        else:
            return False, "EMA ها هم‌راستا نیستن (بازار مردد)", "HOLD"
    
    @staticmethod
    def apply_filters(row, regime_info: dict, enable_filters: bool = True, 
                     min_adx: float = 25.0) -> dict:
        """
        اعمال فیلترهای انعطاف‌پذیر (نسخه 2)
        
        تغییرات نسبت به نسخه 1:
        - ❌ حذف: VROC و CMF (خیلی سخت بودن)
        - ✅ نگهداری: ADX >= 25
        - ✅ تسهیل: BB Width و ATR (آستانه‌های پایین‌تر)
        - ✅ تسهیل: EMA Alignment (حداقل 2 از 3 به‌جای 3 از 3)
        
        Returns:
            dict با کلیدها:
            - passed: آیا تمام فیلترها pass شدن؟
            - allowed_direction: "BUY" | "SELL" | "HOLD"
            - reasons: لیست دلایل (برای log/debug)
        """
        if not enable_filters:
            return {
                "passed": True,
                "allowed_direction": "ANY",
                "reasons": ["Filters disabled"],
                "filter_summary": "⚪ FILTERS OFF"
            }
        
        reasons = []
        all_passed = True
        allowed_direction = "ANY"
        
        # ۱. فیلتر ADX (بدون تغییر)
        adx_pass, adx_reason = RegimeFilter.check_adx_filter(row, min_adx=min_adx)
        reasons.append(f"ADX: {adx_reason}")
        if not adx_pass:
            all_passed = False
        
        # ۲. فیلتر نوسان (تسهیل‌شده)
        vola_pass, vola_reason = RegimeFilter.check_volatility_expansion(row)
        reasons.append(f"Volatility: {vola_reason}")
        if not vola_pass:
            all_passed = False
        
        # ۳. فیلتر هم‌راستایی EMA (تسهیل‌شده: حداقل 2 از 3)
        align_pass, align_reason, align_dir = RegimeFilter.check_trend_alignment(row, regime_info)
        reasons.append(f"Trend: {align_reason}")
        if not align_pass:
            all_passed = False
        else:
            allowed_direction = align_dir
        
        # خلاصه نهایی
        if all_passed:
            summary = f"✅ ALL PASS ({allowed_direction} only)"
        else:
            summary = "❌ BLOCKED"
            allowed_direction = "HOLD"  # اگر هر فیلتری fail شد، فقط HOLD
        
        return {
            "passed": all_passed,
            "allowed_direction": allowed_direction,
            "reasons": reasons,
            "filter_summary": summary
        }


class MarketRegime:
    """
    تشخیص رژیم بازار: این کلاس تشخیص می‌دهد که آیا بازار در روند صعودی/نزولی قوی است،
    یا در رنج (نوسان بدون جهت مشخص)، یا در وضعیت نوسان بالا.
    
    این تشخیص به ما کمک می‌کند که وزن اندیکاتورها را متناسب با رژیم تنظیم کنیم.
    """
    
    @staticmethod
    def detect(row) -> dict:
        """
        تشخیص رژیم بازار بر اساس ADX، ATR، و Bollinger Band Width
        
        Returns:
            dict با کلیدهای:
            - regime: "strong_trend" | "weak_trend" | "ranging" | "high_volatility"
            - trend_direction: "bullish" | "bearish" | "neutral"
            - strength: 0-100 (قدرت رژیم فعلی)
        """
        adx = row.get('ADX_14', 0)
        dmp = row.get('DMP_14', 0)
        dmn = row.get('DMN_14', 0)
        atr = row.get('ATR', 0)
        bb_width = row.get('BB_BANDWIDTH', 0)
        close = row.get('close', 0)
        ema_50 = row.get('EMA_50', 0)
        ema_200 = row.get('EMA_200', 0)
        
        # تشخیص جهت روند
        trend_direction = "neutral"
        if close > ema_50 > ema_200 and dmp > dmn:
            trend_direction = "bullish"
        elif close < ema_50 < ema_200 and dmn > dmp:
            trend_direction = "bearish"
        
        # تشخیص قدرت روند
        if adx > 30:
            regime = "strong_trend"
            strength = min(100, adx * 2)
        elif adx > 20:
            regime = "weak_trend"
            strength = adx * 3
        else:
            regime = "ranging"
            strength = 30
        
        # تشخیص نوسان بالا
        if bb_width > 0.1 or (atr / close if close > 0 else 0) > 0.03:
            regime = "high_volatility"
            strength = 80
        
        return {
            "regime": regime,
            "trend_direction": trend_direction,
            "strength": strength,
            "adx": adx
        }


class SignalScorer:
    """
    موتور اسکورینگ پیشرفته با قابلیت‌های:
    1. تشخیص رژیم بازار و وزن‌دهی پویا
    2. Confluence (همگرایی چند اندیکاتور)
    3. آستانه‌های پویا بر اساس قدرت روند
    4. منطق SHORT و LONG متوازن
    5. فیلترهای رژیم بازار (نسخه 2: انعطاف‌پذیرتر!)
    """
    
    # وزن‌های پایه (در رژیم strong_trend)
    WEIGHTS_STRONG_TREND = {
        "trend": 0.45,      # در روند قوی، اندیکاتورهای روند مهم‌ترند
        "momentum": 0.25,
        "volatility": 0.15,
        "volume": 0.15
    }
    
    # وزن‌های برای بازار رنج
    WEIGHTS_RANGING = {
        "trend": 0.15,      # در رنج، روند اهمیت کمتری داره
        "momentum": 0.40,   # اسیلاتورها (RSI, Stoch) مهم‌تر میشن
        "volatility": 0.25,
        "volume": 0.20
    }
    
    # وزن‌های برای نوسان بالا
    WEIGHTS_HIGH_VOLATILITY = {
        "trend": 0.30,
        "momentum": 0.20,
        "volatility": 0.35,  # در نوسان بالا، Bollinger و ATR خیلی مهمن
        "volume": 0.15
    }
    
    @staticmethod
    def _calculate_trend_score(row, regime_info: dict) -> float:
        """
        محاسبه امتیاز روند با توجه به رژیم بازار
        خروجی: -100 تا +100 (منفی = نزولی، مثبت = صعودی)
        """
        score = 0
        close = row.get('close', 0)
        
        # ۱. موقعیت نسبت به EMA ها (وزن: 30)
        ema_9 = row.get('EMA_9', 0)
        ema_21 = row.get('EMA_21', 0)
        ema_50 = row.get('EMA_50', 0)
        ema_200 = row.get('EMA_200', 0)
        
        if close > ema_200:
            score += 30
        elif close < ema_200:
            score -= 30
        
        # ۲. ترتیب EMA ها (وزن: 25)
        if ema_9 > ema_21 > ema_50:
            score += 25
        elif ema_9 < ema_21 < ema_50:
            score -= 25
        
        # ۳. MACD (وزن: 20)
        macd = row.get('MACD_12_26_9', 0)
        macd_signal = row.get('MACDs_12_26_9', 0)
        if macd > macd_signal and macd > 0:
            score += 20
        elif macd < macd_signal and macd < 0:
            score -= 20
        
        # ۴. ADX + DMI (وزن: 15)
        adx = regime_info.get('adx', 0)
        dmp = row.get('DMP_14', 0)
        dmn = row.get('DMN_14', 0)
        if adx > 25:
            if dmp > dmn:
                score += 15
            elif dmn > dmp:
                score -= 15
        
        # ۵. SuperTrend (وزن: 10)
        supertrend_dir = row.get('SUPERTd_7_3.0', 0)  # 1 = صعودی, -1 = نزولی
        if supertrend_dir > 0:
            score += 10
        elif supertrend_dir < 0:
            score -= 10
        
        return score
    
    @staticmethod
    def _calculate_momentum_score(row, regime_info: dict) -> float:
        """
        محاسبه امتیاز مومنتوم
        خروجی: -100 تا +100
        """
        score = 0
        
        # ۱. RSI (وزن: 35)
        rsi = row.get('RSI', 50)
        if rsi < 30:
            score += 35  # oversold = فرصت خرید
        elif rsi > 70:
            score -= 35  # overbought = فرصت فروش
        elif 40 < rsi < 60:
            score += 0  # neutral
        elif rsi < 40:
            score += 15
        elif rsi > 60:
            score -= 15
        
        # ۲. Stochastic RSI (وزن: 30)
        stoch_k = row.get('STOCH_RSI_K', 50)
        stoch_d = row.get('STOCH_RSI_D', 50)
        if stoch_k < 20 and stoch_k > stoch_d:
            score += 30  # bullish crossover در oversold
        elif stoch_k > 80 and stoch_k < stoch_d:
            score -= 30  # bearish crossover در overbought
        elif stoch_k < 20:
            score += 15
        elif stoch_k > 80:
            score -= 15
        
        # ۳. CCI (وزن: 20)
        cci = row.get('CCI', 0)
        if cci < -100:
            score += 20
        elif cci > 100:
            score -= 20
        
        # ۴. Williams %R (وزن: 15)
        willr = row.get('WILLR', -50)
        if willr < -80:
            score += 15
        elif willr > -20:
            score -= 15
        
        return score
    
    @staticmethod
    def _calculate_volatility_score(row) -> float:
        """
        محاسبه امتیاز نوسان
        خروجی: -100 تا +100
        """
        score = 0
        close = row.get('close', 0)
        
        # ۱. Bollinger Bands (وزن: 50)
        bb_low = row.get('BB_LOW', 0)
        bb_high = row.get('BB_HIGH', 0)
        bb_pct = row.get('BB_PERCENT', 0.5)
        
        if close <= bb_low * 1.005:  # نزدیک باند پایین = oversold
            score += 50
        elif close >= bb_high * 0.995:  # نزدیک باند بالا = overbought
            score -= 50
        elif bb_pct < 0.2:
            score += 25
        elif bb_pct > 0.8:
            score -= 25
        
        # ۲. Keltner Channels (وزن: 30)
        kc_low = row.get('KC_LOW', 0)
        kc_high = row.get('KC_HIGH', 0)
        if kc_low > 0 and close < kc_low:
            score += 30
        elif kc_high > 0 and close > kc_high:
            score -= 30
        
        # ۳. BB Width (وزن: 20)
        bb_width = row.get('BB_BANDWIDTH', 0)
        if bb_width < 0.02:  # فشردگی = احتمال انفجار
            score += 20
        
        return score
    
    @staticmethod
    def _calculate_volume_score(row) -> float:
        """
        محاسبه امتیاز حجم و جریان پول
        خروجی: -100 تا +100
        """
        score = 0
        
        # ۱. CMF (وزن: 40)
        cmf = row.get('CMF', 0)
        if cmf > 0.15:
            score += 40
        elif cmf < -0.15:
            score -= 40
        elif cmf > 0.05:
            score += 20
        elif cmf < -0.05:
            score -= 20
        
        # ۲. MFI (وزن: 35)
        mfi = row.get('MFI', 50)
        if mfi < 20:
            score += 35
        elif mfi > 80:
            score -= 35
        elif mfi < 30:
            score += 15
        elif mfi > 70:
            score -= 15
        
        # ۳. Volume ROC (وزن: 25)
        vroc = row.get('VROC', 0)
        if vroc > 50:  # حجم در حال افزایش = تایید حرکت
            score += 25
        elif vroc < -30:
            score -= 25
        
        return score
    
    @staticmethod
    def _calculate_confluence_bonus(trend_score: float, momentum_score: float, 
                                     volatility_score: float, volume_score: float) -> float:
        """
        محاسبه امتیاز همگرایی (Confluence): وقتی چند اندیکاتور هم‌جهت باشن،
        امتیاز اضافه می‌گیره (تایید چند لایه‌ای)
        
        خروجی: -20 تا +20
        """
        signals = [trend_score, momentum_score, volatility_score, volume_score]
        
        # چک کن همه مثبت (bullish) یا همه منفی (bearish) باشن
        all_positive = all(s > 20 for s in signals)
        all_negative = all(s < -20 for s in signals)
        
        if all_positive:
            return 20  # همه bullish = تایید قوی
        elif all_negative:
            return -20  # همه bearish = تایید قوی
        
        # حداقل 3 از 4 هم‌جهت باشن
        positive_count = sum(1 for s in signals if s > 20)
        negative_count = sum(1 for s in signals if s < -20)
        
        if positive_count >= 3:
            return 10
        elif negative_count >= 3:
            return -10
        
        return 0
    
    @staticmethod
    def score_row(row, buy_threshold: float = None, sell_threshold: float = None,
                  enable_regime_filter: bool = True, min_adx: float = 25.0) -> dict:
        """
        محاسبه اسکور پیشرفته با توجه به رژیم بازار + فیلترهای اضافی (نسخه 2)
        
        Args:
            enable_regime_filter: فعال/غیرفعال کردن فیلتر رژیم (برای مقایسه)
            min_adx: حداقل ADX برای pass کردن فیلتر
        
        خروجی: -100 تا +100 (منفی = SHORT, مثبت = LONG)
        """
        # ۱. تشخیص رژیم بازار
        regime_info = MarketRegime.detect(row)
        regime = regime_info['regime']
        
        # ۲. اعمال فیلترهای رژیم بازار (نسخه 2: انعطاف‌پذیرتر!)
        filter_result = RegimeFilter.apply_filters(
            row, regime_info, 
            enable_filters=enable_regime_filter,
            min_adx=min_adx
        )
        
        # ۳. انتخاب وزن‌ها بر اساس رژیم
        if regime == "strong_trend":
            weights = SignalScorer.WEIGHTS_STRONG_TREND
        elif regime == "ranging":
            weights = SignalScorer.WEIGHTS_RANGING
        else:  # high_volatility یا weak_trend
            weights = SignalScorer.WEIGHTS_HIGH_VOLATILITY
        
        # ۴. محاسبه امتیاز هر بخش (-100 تا +100)
        trend_score = SignalScorer._calculate_trend_score(row, regime_info)
        momentum_score = SignalScorer._calculate_momentum_score(row, regime_info)
        volatility_score = SignalScorer._calculate_volatility_score(row)
        volume_score = SignalScorer._calculate_volume_score(row)
        
        # ۵. محاسبه امتیاز همگرایی
        confluence_bonus = SignalScorer._calculate_confluence_bonus(
            trend_score, momentum_score, volatility_score, volume_score
        )
        
        # ۶. وزن‌دهی نهایی
        weighted_score = (
            trend_score * weights["trend"] +
            momentum_score * weights["momentum"] +
            volatility_score * weights["volatility"] +
            volume_score * weights["volume"]
        )
        
        # اضافه کردن امتیاز همگرایی
        final_score = weighted_score + confluence_bonus
        
        # ۷. نرمال‌سازی به 0-100 برای سازگاری با کد قبلی
        normalized_score = (final_score + 100) / 2  # تبدیل -100..+100 به 0..100
        
        # ۸. تعیین آستانه‌های پویا بر اساس قدرت روند
        if buy_threshold is None or sell_threshold is None:
            strength = regime_info['strength']
            if regime == "strong_trend":
                # در روند قوی، آستانه سخت‌گیرانه‌تر (فقط سیگنال‌های قوی)
                buy_threshold = 70 + (strength - 60) * 0.2
                sell_threshold = 30 - (strength - 60) * 0.2
            elif regime == "ranging":
                # در رنج، آستانه‌های محافظه‌کارانه‌تر
                buy_threshold = 75
                sell_threshold = 25
            else:
                # پیش‌فرض
                buy_threshold = 70
                sell_threshold = 30
        
        # ۹. تص��یم‌گیری نهایی (با در نظر گرفتن فیلترها)
        direction = "HOLD"
        confidence = abs(final_score) / 100  # 0 تا 1
        
        # اگر فیلتر رژیم pass نشد، فقط HOLD
        if not filter_result["passed"]:
            direction = "HOLD"
        else:
            # فیلتر pass شد، حالا بر اساس اسکور و آستانه تصمیم می‌گیریم
            allowed_dir = filter_result["allowed_direction"]
            
            if normalized_score >= buy_threshold and allowed_dir in ["BUY", "ANY"]:
                direction = "BUY"
            elif normalized_score <= sell_threshold and allowed_dir in ["SELL", "ANY"]:
                direction = "SELL"
            else:
                direction = "HOLD"
        
        # در رژیم ranging، محتاط‌تر باش (حتی اگر فیلتر pass شد)
        if regime == "ranging" and confidence < 0.5:
            direction = "HOLD"
        
        return {
            "score": round(normalized_score, 2),
            "raw_score": round(final_score, 2),  # نمره خام (-100 تا +100)
            "direction": direction,
            "confidence": round(confidence, 2),
            "regime": regime,
            "regime_strength": round(regime_info['strength'], 2),
            "trend_direction": regime_info['trend_direction'],
            "regime_filter": filter_result,  # اضافه کردن نتیجه فیلترها
            "details": {
                "trend": round(trend_score, 2),
                "momentum": round(momentum_score, 2),
                "volatility": round(volatility_score, 2),
                "volume": round(volume_score, 2),
                "confluence": round(confluence_bonus, 2)
            },
            "thresholds": {
                "buy": round(buy_threshold, 2),
                "sell": round(sell_threshold, 2)
            }
        }
    
    @staticmethod
    def calculate_score(df: pd.DataFrame) -> dict:
        """
        محاسبه نمره سیگنال برای آخرین کندل (سازگاری با API قبلی)
        """
        if df.empty:
            return {
                "score": 50, 
                "direction": "HOLD", 
                "confidence": 0,
                "regime": "unknown",
                "details": {}
            }
        
        last_row = df.iloc[-1]
        return SignalScorer.score_row(last_row)
