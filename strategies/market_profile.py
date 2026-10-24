"""
market_profile.py
-------------------
لایه‌ی «دستیار تصمیم‌گیری»: خروجی SignalScorer (که یه امتیاز خطی و یه
BUY/SELL/HOLD قطعیه) رو به یه تصویر چندبعدی و قابل‌فهم تبدیل می‌کنه.

چرا این‌جوری؟
بعد از چند دور بک‌تست جدی (backtest.py، optimize_backtest.py، walk_forward.py،
auto_walk_forward.py)، مشخص شد که فرمول ترکیبی امتیازدهی (SignalScorer) به‌تنهایی
edge آماری قابل‌اتکایی نداره. پس دیگه نباید طوری رفتار کنیم که انگار یه
تصمیم قطعی (BUY/SELL) قابل‌اعتماده. به‌جاش این ماژول یه «پروفایل بازار»
می‌سازه: وضعیت هر بُعد (روند/مومنتوم/نوسان/حجم) به زبون ساده، به‌علاوه یه
سنجه‌ی صریح از اینکه ابعاد مختلف با هم هم‌جهت هستن یا در تضادن.

تصمیم نهایی همیشه دست کاربر (یا AI که فقط تفسیر می‌کنه، نه حکم می‌ده) می‌مونه.
"""

from strategies.scoring import SignalScorer, MarketRegime


class MarketProfile:
    @staticmethod
    def label_trend(score: float) -> str:
        if score > 50:
            return "روند صعودی قوی"
        elif score > 15:
            return "روند صعودی ملایم"
        elif score < -50:
            return "روند نزولی قوی"
        elif score < -15:
            return "روند نزولی ملایم"
        else:
            return "بدون روند مشخص (خنثی)"

    @staticmethod
    def label_momentum(score: float) -> str:
        if score > 50:
            return "اشباع فروش نسبی (احتمال برگشت به بالا بیشتره)"
        elif score > 15:
            return "مومنتوم مثبت ملایم"
        elif score < -50:
            return "اشباع خرید نسبی (احتمال برگشت به پایین بیشتره)"
        elif score < -15:
            return "مومنتوم منفی ملایم"
        else:
            return "مومنتوم خنثی"

    @staticmethod
    def label_volatility(score: float) -> str:
        if score > 40:
            return "نزدیک کف محدوده‌ی نوسان اخیر"
        elif score < -40:
            return "نزدیک سقف محدوده‌ی نوسان اخیر"
        elif abs(score) <= 15:
            return "نوسان عادی، بدون سیگنال خاص"
        else:
            return "نوسان نسبتاً کشیده به یک‌سمت"

    @staticmethod
    def label_volume(score: float) -> str:
        if score > 40:
            return "ورود قابل‌توجه نقدینگی"
        elif score > 15:
            return "ورود ملایم نقدینگی"
        elif score < -40:
            return "خروج قابل‌توجه نقدینگی"
        elif score < -15:
            return "خروج ملایم نقدینگی"
        else:
            return "حجم خنثی، بدون سیگنال خاص"

    @staticmethod
    def label_regime(regime_info: dict) -> str:
        mapping = {
            "strong_trend": "روند قوی",
            "weak_trend": "روند ضعیف",
            "ranging": "بازار رنج (بدون روند مشخص)",
            "high_volatility": "نوسان بالا / کمتر قابل‌پیش‌بینی",
        }
        return mapping.get(regime_info.get("regime"), regime_info.get("regime", "نامشخص"))

    @staticmethod
    def build(row) -> dict:
        """
        ساخت پروفایل کامل بازار برای یک کندل (ردیف) مشخص.
        هیچ‌جای این تابع یک تصمیم قطعی (BUY/SELL) صادر نمی‌کنه — فقط توصیف می‌کنه.
        """
        regime_info = MarketRegime.detect(row)

        trend_score = SignalScorer._calculate_trend_score(row, regime_info)
        momentum_score = SignalScorer._calculate_momentum_score(row, regime_info)
        volatility_score = SignalScorer._calculate_volatility_score(row)
        volume_score = SignalScorer._calculate_volume_score(row)

        dimensions = {
            "trend": {
                "score": round(trend_score, 1),
                "label": MarketProfile.label_trend(trend_score),
            },
            "momentum": {
                "score": round(momentum_score, 1),
                "label": MarketProfile.label_momentum(momentum_score),
            },
            "volatility": {
                "score": round(volatility_score, 1),
                "label": MarketProfile.label_volatility(volatility_score),
            },
            "volume": {
                "score": round(volume_score, 1),
                "label": MarketProfile.label_volume(volume_score),
            },
        }

        # سنجش هم‌جهتی بین ابعاد مختلف — این مهم‌ترین بخش برای صداقت با کاربره
        dim_scores = {
            "trend": trend_score,
            "momentum": momentum_score,
            "volatility": volatility_score,
            "volume": volume_score,
        }
        # اسم بُعدها رو صریحاً توی هر دسته می‌ذاریم (نه فقط تعداد)، چون وقتی این اطلاعات
        # رو به AI می‌دیم، نباید مجبورش کنیم خودش از روی امتیازهای خام حساب کنه کدوم بُعد
        # کجا افتاده — مدل‌های کوچیک (مثل Mistral-7B) توی این نوع محاسبه‌ی ساده هم خطا
        # می‌کنن (مثلاً یه بُعد رو اشتباه دسته‌بندی می‌کنن یا حتی یه بُعد ساختگی می‌سازن).
        bullish_dims = [name for name, s in dim_scores.items() if s > 20]
        bearish_dims = [name for name, s in dim_scores.items() if s < -20]
        neutral_dims = [name for name in dim_scores if name not in bullish_dims and name not in bearish_dims]

        bullish_count = len(bullish_dims)
        bearish_count = len(bearish_dims)
        neutral_count = len(neutral_dims)

        uncertainty_note = None
        if bullish_count >= 3:
            agreement_label = "اکثر ابعاد هم‌جهت صعودی‌ان"
        elif bearish_count >= 3:
            agreement_label = "اکثر ابعاد هم‌جهت نزولی‌ان"
        elif bullish_count >= 2 and bearish_count >= 2:
            agreement_label = "⚠️ ابعاد مختلف در تضاد آشکارن"
            uncertainty_note = (
                "بعضی ابعاد (مثلاً روند) صعودی‌ان ولی بعضی دیگه (مثلاً حجم یا مومنتوم) "
                "نزولی‌ان. این وضعیت معمولاً یعنی بازار در حال تصمیم‌گیریه و مسیر روشنی نداره — "
                "نه اینکه یه سیگنال ضعیف باشه که بشه نادیده گرفت."
            )
        elif neutral_count >= 3:
            agreement_label = "اکثر ابعاد خنثی/کم‌سیگنال‌ان"
            uncertainty_note = "بازار احتمالاً بی‌جهت یا در حال استراحته؛ شواهد کافی برای هیچ جهتی نیست."
        elif bullish_count == 2 and bearish_count == 0:
            agreement_label = "گرایش ملایم به صعودی (بدون تضاد، ولی شواهد کامل/قوی نیست)"
        elif bearish_count == 2 and bullish_count == 0:
            agreement_label = "گرایش ملایم به نزولی (بدون تضاد، ولی شواهد کامل/قوی نیست)"
        else:
            agreement_label = "سیگنال‌های ابعاد ضعیف و پراکنده‌ان"

        return {
            "regime": MarketProfile.label_regime(regime_info),
            "regime_raw": regime_info.get("regime"),
            "dimensions": dimensions,
            "agreement": {
                "bullish_dimensions": bullish_count,
                "bearish_dimensions": bearish_count,
                "neutral_dimensions": neutral_count,
                "bullish_dimension_names": bullish_dims,
                "bearish_dimension_names": bearish_dims,
                "neutral_dimension_names": neutral_dims,
                "label": agreement_label,
            },
            "uncertainty_note": uncertainty_note,
        }
