import logging
from openai import OpenAI
from config.settings import Config

logger = logging.getLogger(__name__)

class AIBrain:
    def __init__(self):
        """
        راه‌اندازی اتصال به سرور محلی LM Studio
        """
        base_url = getattr(Config, "LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        try:
            self.client = OpenAI(
                base_url=base_url,
                api_key="lm-studio"
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
            active_models = [model.id for model in models_data.data]
            logger.info(f"Detected active models in LM Studio: {active_models}")
            return active_models
        except Exception as e:
            logger.error(f"Could not fetch active models from LM Studio: {str(e)}")
            return []

    def _get_model_opinion(self, model_name: str, symbol: str, technical_summary: dict, score_details: dict,
                            active_patterns: list = None, chart_patterns: list = None) -> str:
        """
        ارسال متنی درخواست به مدل و دریافت پاسخ با فرمت ایمن و سازگار با همه مدل‌ها (مانند Mistral)
        """
        try:
            # تعریف دستورالعمل نقش سیستم
            system_instruction = (
                "You are a technical-analysis assistant. Your job is to describe what the current "
                "indicators show, NOT to issue confident trading advice. This scoring formula has been "
                "backtested (Walk-Forward Validation) and shown NO reliable statistical edge — treat it "
                "as descriptive context only, never as a proven signal."
            )

            patterns_text = ""
            if active_patterns:
                patterns_text += f"\n            - Active candle patterns on latest candle: {', '.join(active_patterns)}"
            if chart_patterns:
                cp_str = ", ".join(f"{p['pattern']} near {p['price']}" for p in chart_patterns)
                patterns_text += f"\n            - Recent chart patterns: {cp_str}"

            prompt = f"""
            Here is the current technical read for {symbol}:

            Technical Indicators Summary:
            - Last Close Price: {technical_summary.get('close')}
            - RSI (14): {technical_summary.get('rsi')}
            - CMF (Chaikin Money Flow): {technical_summary.get('cmf')}
            - ADX: {technical_summary.get('adx')}{patterns_text}

            Scoring Engine Output (descriptive only, no proven edge):
            - Technical Score: {score_details.get('score')}/100
            - Indicator Bias: {score_details.get('direction')}

            Task:
            In max 3 sentences, describe what this combination of indicators/patterns typically suggests
            about current market conditions. Mention one bullish consideration AND one bearish/risk
            consideration — do not give a one-sided directive, and do not state a stop-loss/take-profit
            as if it were a confirmed trade setup. End by reminding the reader this is not financial advice.
            """

            # ادغام دستورالعمل سیستم با درخواست کاربر برای جلوگیری از ارور 400 در مدل Mistral v0.3
            combined_content = f"{system_instruction}\n\n{prompt}"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": combined_content}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error communicating with model {model_name}: {str(e)}")
            clean_name = model_name.split('/')[-1] if '/' in model_name else model_name
            return f"Error: Could not get response from {clean_name}."

    def analyze_with_both_models(self, symbol: str, technical_summary: dict, score_details: dict,
                                  active_patterns: list = None, chart_patterns: list = None) -> dict:
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

        # ۱. تحلیل با مدل اول لود شده (مانند Llama 3)
        model_1 = active_models[0]
        logger.info(f"Querying Model 1 ({model_1})...")
        opinions["model_1_name"] = model_1.split('/')[-1]
        opinions["model_1_opinion"] = self._get_model_opinion(
            model_1, symbol, technical_summary, score_details, active_patterns, chart_patterns
        )

        # ۲. تحلیل با مدل دوم لود شده (مانند Mistral)
        if len(active_models) > 1:
            model_2 = active_models[1]
            logger.info(f"Querying Model 2 ({model_2})...")
            opinions["model_2_name"] = model_2.split('/')[-1]
            opinions["model_2_opinion"] = self._get_model_opinion(
                model_2, symbol, technical_summary, score_details, active_patterns, chart_patterns
            )
        else:
            opinions["model_2_name"] = "None"
            opinions["model_2_opinion"] = "Only one model is currently loaded in LM Studio."

        return opinions