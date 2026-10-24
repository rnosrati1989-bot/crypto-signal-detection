import logging
import re
from openai import OpenAI
from config.settings import Config

logger = logging.getLogger(__name__)

# قالب خط جمع‌بندی که از هر مدل خواسته می‌شود در انتهای پاسخ بنویسد
# (توصیفی است، نه سیگنال معاملاتی)
VERDICT_TEMPLATE = (
    "VERDICT_LINE: bias=bullish|neutral|bearish|mixed | "
    "confidence=0.0-1.0 | watch=<one concrete thing to watch>"
)

# پارسر خط جمع‌بندی — مقاوم در برابر متن اضافه
_VERDICT_RE = re.compile(
    r"bias\s*=\s*(\w+).*?confidence\s*=\s*([\d.]+|\w+).*?watch\s*=\s*([^\n]*)",
    re.IGNORECASE | re.DOTALL,
)

# برچسب‌های دیجست برای پرامپت LLM — ترتیب نمایش را هم مشخص می‌کند.
# همه‌ی این مقادیر نرمال/نسبتی/ایستا هستند (نه قیمت مطلق) و از
# core_pipeline.technical_summary["digest"] می‌آیند.
_DIGEST_LABELS = (
    ("ema9_dist", "close distance to EMA9 (relative)"),
    ("ema21_dist", "close distance to EMA21 (relative)"),
    ("ema50_dist", "close distance to EMA50 (relative)"),
    ("ema200_dist", "close distance to EMA200 (relative)"),
    ("ema9_21_cross", "EMA9/EMA21 ratio - 1 (short-term trend alignment)"),
    ("atr_pct", "ATR as fraction of price (volatility magnitude)"),
    ("vol_regime", "ATR / 48-bar mean ATR (>1 expanding, <1 compressing)"),
    ("range_pos", "close position in last 20 bars (0 = low, 1 = high)"),
    ("vol_z", "volume z-score (clipped to [-3, 3])"),
    ("obv_z", "OBV z-score (money flow pressure, clipped)"),
    # taker_buy_ratio / funding_* are futures-data only and are NOT in the live
    # spot digest (core_pipeline._build_digest); add them back if the live path
    # ever switches to the futures client.
    ("macdh_norm", "MACD histogram / ATR (momentum magnitude)"),
    ("macdh_slope", "MACD histogram 1-bar change / ATR"),
    ("rsi_slope", "RSI 1-bar change"),
    ("vwap_dist", "close distance to VWAP (relative)"),
    ("kc_pos", "Keltner channel position (0 = lower, 1 = upper band)"),
    ("dc_pos", "Donchian channel position (0..1)"),
    ("psar_dist_atr", "close-to-PSAR distance, in ATR units"),
    ("st_dir", "SuperTrend direction (+1 up / -1 down)"),
    ("pat_net_3", "net candle-pattern direction, last 3 bars"),
    ("pat_net_12", "net candle-pattern direction, last 12 bars"),
)


# مدل‌های embedding فقط بردار می‌سازند و قابلیت چت (نظرخواهی) ندارند.
# LM Studio این‌ها را کنار مدل‌های LLM هم لیست می‌کند؛ این کلیدواژه‌ها
# (مقایسه با نادیده‌گرفتن بزرگی/کوچکی حروف) قبل از انتخاب دو مدل برای
# نظرخواهی فیلتر می‌شوند تا یک مدل embedding به‌جای یک LLM واقعی کوئری نشود.
_EMBEDDING_KEYWORDS = ("embed", "bge", "gte", "minilm", "e5-")


# تقریب اندازهی مدل از روی نام (بیلیون پارامتر) — برای انتخاب دو مدل بزرگتر
_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def _model_size(model_id: str) -> float:
    """استخراج تقریبی اندازهی مدل؛ مثلاً qwen2.5-14b-instruct → 14"""
    m = _MODEL_SIZE_RE.search(model_id)
    return float(m.group(1)) if m else 0.0


def _is_embedding_model(model_id: str) -> bool:
    """آیا model id یک مدل embedding است؟"""
    lowered = model_id.lower()
    return any(kw in lowered for kw in _EMBEDDING_KEYWORDS)


class AIBrain:
    def __init__(self):
        """
        راه‌اندازی اتصال به سرور محلی LM Studio
        """
        base_url = getattr(Config, "LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        try:
            self.client = OpenAI(
                base_url=base_url,
                api_key="lm-studio",
                timeout=240.0,
                max_retries=1
            )
            logger.info(f"AI Client successfully connected to {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise e

    def _get_active_models(self) -> list:
        """
        دریافت خودکار لیست مدل‌های فعال و لود شده در LM Studio
        """
        try:
            models_data = self.client.models.list()
            # فقط مدل‌های LLM — رد کردن مدل‌های embedding تا به‌جای یک مدل چت کوئری نشوند
            active_models = [
                model.id for model in models_data.data
                if not _is_embedding_model(model.id)
            ]
            logger.info(f"Detected active models in LM Studio: {active_models}")
            return active_models
        except Exception as e:
            logger.error(f"Could not fetch active models from LM Studio: {str(e)}")
            return []

    def _get_model_opinion(self, model_name: str, symbol: str, technical_summary: dict, market_profile: dict,
                            active_patterns: list = None, chart_patterns: list = None,
                            orderbook_context: dict = None) -> str:
        """
        ارسال متنی درخواست به مدل و دریافت پاسخ با فرمت ایمن و سازگار با همه مدل‌ها (مانند Mistral)
        """
        try:
            # تعریف دستورالعمل نقش سیستم
            system_instruction = (
                "IMPORTANT: Write your ENTIRE response in Persian (Farsi). All prose must be Persian. "
                "Keep only the final VERDICT_LINE keywords (bias=, confidence=, watch=) in English; the "
                "watch value itself should be in Persian. "
                "You are a technical-analysis assistant. Your job is to INTERPRET a multi-dimensional "
                "market profile for the reader, NOT to issue a buy/sell verdict. The profile below already "
                "avoids collapsing everything into a single score, on purpose — after extensive backtesting, "
                "a single blended score was shown to have no reliable statistical edge. Respect that design: "
                "describe what each dimension suggests, and — most importantly — if dimensions disagree with "
                "each other, say so explicitly and explain what that disagreement usually means, rather than "
                "picking a side. Below you also get a 'Technical digest': normalized, stationary, "
                "position-relative values of the latest candle (distances to EMAs, volatility regime, "
                "volume/funding flows, MACD/RSI slopes, pattern net). Use it to QUALIFY the dimensions — "
                "e.g. is the trend strong or fading, is volatility expanding, does flow support the move — "
                "but never as a standalone signal."
            )

            patterns_text = ""
            if active_patterns:
                patterns_text += f"\n            - Active candle patterns on latest candle: {', '.join(active_patterns)}"
            if chart_patterns:
                cp_str = ", ".join(f"{p['pattern']} near {p['price']}" for p in chart_patterns)
                patterns_text += f"\n            - Recent chart patterns: {cp_str}"

            dims = market_profile.get("dimensions", {})
            dims_text = "\n".join(
                f"            - {name.capitalize()}: {info['label']} (raw score: {info['score']})"
                for name, info in dims.items()
            )
            agreement = market_profile.get("agreement", {})
            uncertainty = market_profile.get("uncertainty_note")

            bullish_names = ", ".join(agreement.get("bullish_dimension_names", [])) or "none"
            bearish_names = ", ".join(agreement.get("bearish_dimension_names", [])) or "none"
            neutral_names = ", ".join(agreement.get("neutral_dimension_names", [])) or "none"

            ob_text = ""
            if orderbook_context:
                imb20 = orderbook_context.get("imbalance_20")
                if imb20 is None:
                    pressure = "نامشخص"
                elif imb20 > 0.3:
                    pressure = "فشار خرید"
                elif imb20 < -0.3:
                    pressure = "فشار فروش"
                else:
                    pressure = "متوازن"
                ob_text = (
                    f"\n            Live order book context (real-time depth, descriptive only):\n"
                    f"            - imbalance_20 (+1 = buy pressure, -1 = sell pressure): "
                    f"{orderbook_context.get('imbalance_20')} ({pressure})\n"
                    f"            - depth_ratio_20 (bid depth / ask depth): "
                    f"{orderbook_context.get('depth_ratio_20')}\n"
                    f"            - spread_pct: {orderbook_context.get('spread_pct')}"
                )

            # دیجست نرمال‌شده (اختیاری) — اگر ساخته نشد، پرامپت بدون آن هم کار می‌کند
            digest = technical_summary.get("digest") or {}
            if digest:
                digest_lines = []
                for key, label in _DIGEST_LABELS:
                    val = digest.get(key)
                    if val is None:
                        digest_lines.append(f"            - {label}: n/a")
                    else:
                        digest_lines.append(f"            - {label}: {val}")
                digest_text = (
                    "\n            Technical digest (normalized, stationary values of the latest candle, "
                    "position-relative ONLY — no absolute price levels; descriptive, not a signal):\n"
                    + "\n".join(digest_lines)
                )
            else:
                digest_text = ""

            prompt = f"""
            Here is the current market profile for {symbol}:

            Regime: {market_profile.get('regime')}

            Dimension readings (each is descriptive, not a verdict):
{dims_text}

            IMPORTANT — dimension classification (already computed for you, do NOT recompute or guess):
            - Bullish dimensions: {bullish_names}
            - Bearish dimensions: {bearish_names}
            - Neutral dimensions: {neutral_names}
            Only ever refer to dimensions using these exact classifications and names (trend, momentum,
            volatility, volume) — do not invent other dimension names.

            Dimension agreement: {agreement.get('label')}
            {f"Note on uncertainty: {uncertainty}" if uncertainty else ""}

            Supporting numeric context:
            - Last Close Price: {technical_summary.get('close')}
            - RSI (14): {technical_summary.get('rsi')}
            - CMF (Chaikin Money Flow): {technical_summary.get('cmf')}
            - ADX: {technical_summary.get('adx')}{digest_text}{patterns_text}{ob_text}

            Task (max 5 sentences):
            1. Briefly explain what this specific combination of dimensions typically suggests about
               near-term market behavior. If live order book context is provided, explicitly note
               whether the book pressure (imbalance) supports or contradicts the technical picture.
               Use the technical digest values (e.g. distances to EMAs, volatility regime, flow, slopes)
               to make this explanation concrete — e.g. "trend is still up but slowing", "volatility is
               expanding", "sell flow is rising" — only when the digest supports it.
            2. Use the dimension agreement label above (already computed) to explain what it means in
               practice — do NOT judge for yourself whether the dimensions agree or disagree; the label
               above is authoritative. Do NOT resolve any disagreement by picking a side.
            3. Mention ONE concrete thing a careful observer would watch for next (e.g. a level breaking,
               an indicator crossing) as a way to reduce the uncertainty — not as a trade trigger.
            4. End by reminding the reader this is descriptive context, not financial advice.
            5. Finally, append EXACTLY one final line — nothing after it — in this exact format:
               {VERDICT_TEMPLATE}
               where bias is your descriptive reading of the combined profile (bullish = bias to the
               upside, bearish = bias to the downside, neutral = no clear bias, mixed = conflicting
               strong signals), confidence is how confident you are in that reading (0.0-1.0; lower when
               dimensions conflict), and watch is the SAME concrete thing from task 3, written SHORT
               (max 6 words, preferably a price level like '64779.4'). This line is a
               descriptive summary for an ensemble step, NOT a trade signal.
            """

            # ادغام دستورالعمل سیستم با درخواست کاربر برای جلوگیری از ارور 400 در مدل Mistral v0.3
            combined_content = f"{system_instruction}\n\n{prompt}"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": combined_content}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error communicating with model {model_name}: {str(e)}")
            clean_name = model_name.split('/')[-1] if '/' in model_name else model_name
            return f"Error: Could not get response from {clean_name}."

    # ------------------------------------------------------------------
    # Ensemble Consensus
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_verdict(text: str):
        """
        استخراج خط جمع‌بندی (bias/confidence/watch) از پاسخ مدل.
        اگر قالب پیدا نشود None برمی‌گرداند (صادقانه: عدم توافق قابل استخراج نیست).
        """
        if not text:
            return None
        m = _VERDICT_RE.search(text)
        if not m:
            return None
        bias = m.group(1).strip().lower()
        if bias not in ("bullish", "neutral", "bearish", "mixed"):
            bias = "unknown"
        try:
            conf = float(m.group(2)) if m.group(2) is not None else None
            if conf is not None:
                conf = max(0.0, min(1.0, conf))
        except ValueError:
            conf = None
        watch = m.group(3).strip() or None
        return {"bias": bias, "confidence": conf, "watch": watch}

    @staticmethod
    def compute_consensus(v1: dict, v2: dict) -> dict:
        """
        ترکیب دو «رأی توصیفی» مدل‌ها به یک جمع‌بندی.
        تعیینی و قابل تست — بدون مدل سوم.
        """
        if not v1 or not v2:
            return {
                "level": "unavailable",
                "bias": "unknown",
                "avg_confidence": None,
                "summary": "امکان ترکیب نظرات وجود نداشت (قالب پاسخ مدل‌ها قابل استخراج نبود).",
            }

        b1, b2 = v1["bias"], v2["bias"]
        c1, c2 = v1.get("confidence"), v2.get("confidence")
        confs = [c for c in (c1, c2) if c is not None]
        avg_conf = (sum(confs) / len(confs)) if confs else None

        bias_fa = {
            "bullish": "صعودی", "bearish": "نزولی",
            "neutral": "خنثی", "mixed": "مختلط", "unknown": "نامشخص",
        }

        # ۱. هر دو یک‌جهت
        if b1 == b2:
            if b1 == "neutral":
                return {"level": "neutral_agreement", "bias": "neutral",
                        "avg_confidence": avg_conf,
                        "summary": "🤝 هر دو مدل بی‌طرف‌اند — بازار را بدون جهت روشن می‌بینند."}
            if b1 == "mixed":
                return {"level": "mixed_agreement", "bias": "mixed",
                        "avg_confidence": avg_conf,
                        "summary": ("⚡ هر دو مدل سیگنال‌های متضاد دیده‌اند — عدم قطعیت بالاست."
                                + (f" (اطمینان میانگین: {avg_conf:.2f})" if avg_conf is not None else ""))}
            strong = avg_conf is not None and avg_conf >= 0.65
            level = "strong_agreement" if strong else "agreement"
            tag = "توافق قوی" if strong else "توافق"
            conf_note = f" (اطمینان میانگین: {avg_conf:.2f})" if avg_conf is not None else ""
            return {"level": level, "bias": b1, "avg_confidence": avg_conf,
                    "summary": f"🤝 {tag}: هر دو مدل {bias_fa[b1]} دیده‌اند.{conf_note}"}

        # ۲. یکی بی‌طرف، دیگری جهت‌دار
        if "neutral" in (b1, b2):
            other = b2 if b1 == "neutral" else b1
            return {"level": "one_neutral", "bias": other, "avg_confidence": avg_conf,
                    "summary": f"⚠️ یک مدل بی‌طرف است و دیگری {bias_fa[other]} — جهت ضعیف."}

        # ۳. جهت‌های مخالف (یا ترکیب با mixed)
        return {"level": "disagreement", "bias": "mixed", "avg_confidence": avg_conf,
                "summary": "⚡ اختلاف نظر آشکار بین دو مدل — خودِ این نبودِ اطمینان، اطلاعات مهمی است؛ "
                           "بازار معمولاً در چنین حالتی در نقطه‌ی تصمیم است."}

    def analyze_with_consensus(self, symbol: str, technical_summary: dict, market_profile: dict,
                               active_patterns: list = None, chart_patterns: list = None,
                               orderbook_context: dict = None) -> dict:
        """
        تحلیل با هر دو مدل + محاسبه‌ی Ensemble Consensus.
        خروجی شامل دو نظر و یک جمع‌بندی (`consensus`) است.
        """
        opinions = self.analyze_with_both_models(
            symbol, technical_summary, market_profile,
            active_patterns, chart_patterns, orderbook_context
        )
        v1 = self._parse_verdict(opinions.get("model_1_opinion", ""))
        v2 = self._parse_verdict(opinions.get("model_2_opinion", ""))
        opinions["consensus"] = self.compute_consensus(v1, v2)
        opinions["watch_items"] = [
            v["watch"] for v in (v1, v2) if v and v.get("watch")
        ]
        return opinions

    def analyze_with_both_models(self, symbol: str, technical_summary: dict, market_profile: dict,
                                  active_patterns: list = None, chart_patterns: list = None,
                                  orderbook_context: dict = None) -> dict:
        """
        تحلیل با استفاده از مدل‌های فعال لود شده در LM Studio به صورت همزمان
        """
        active_models = self._get_active_models()
        opinions = {}

        if not active_models:
            return {
                "model_1_name": "Error",
                "model_1_opinion": "No active models detected. Ensure models are loaded in LM Studio.",
                "model_2_name": "None",
                "model_2_opinion": "Only one model is required but none found."
            }

        # ۱. تحلیل با بزرگ‌ترین مدل LLM (مثلاً Qwen 14B)
        # انتخاب دو مدل LLM بزرگ‌تر (پیش‌فرض واقع‌بینانه برای کیفیت) — مستقل از ترتیب لود در LM Studio
        consult_models = sorted(active_models, key=_model_size, reverse=True)[:2]

        model_1 = consult_models[0]
        logger.info(f"Querying Model 1 ({model_1})...")
        opinions["model_1_name"] = model_1.split('/')[-1]
        opinions["model_1_opinion"] = self._get_model_opinion(
            model_1, symbol, technical_summary, market_profile, active_patterns, chart_patterns,
            orderbook_context
        )

        # ۲. تحلیل با دومین مدل بزرگ‌تر
        if len(consult_models) > 1:
            model_2 = consult_models[1]
            logger.info(f"Querying Model 2 ({model_2})...")
            opinions["model_2_name"] = model_2.split('/')[-1]
            opinions["model_2_opinion"] = self._get_model_opinion(
                model_2, symbol, technical_summary, market_profile, active_patterns, chart_patterns,
            orderbook_context
            )
        else:
            opinions["model_2_name"] = "None"
            opinions["model_2_opinion"] = "Only one model is currently loaded in LM Studio."

        return opinions