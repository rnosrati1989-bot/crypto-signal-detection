import os
import sys
# غیرفعال کردن پراکسی برای ارتباطات محلی (به شدت مهم برای حل ارور 503)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"
import logging
from config.settings import Config
from strategies.ai_brain import AIBrain
from core_pipeline import analyze_symbol, AnalysisError

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("MainSystem")

def run_analysis_pipeline(symbol: str, timeframe: str):
    logger.info("=" * 60)
    logger.info(f"STARTING ANALYSIS FOR {symbol} ON {timeframe} TIMEFRAME")
    logger.info("=" * 60)

    # ۱ تا ۴. دریافت داده + محاسبه اندیکاتورها + الگوها + اسکورینگ (پایپ‌لاین مشترک)
    try:
        result = analyze_symbol(symbol, timeframe, klines_limit=250)
    except AnalysisError as e:
        logger.error(f"Pipeline aborted for {symbol}: {e}")
        return

    technical_summary = result["technical_summary"]
    score_result = result["score_result"]
    supports = result["supports"]
    resistances = result["resistances"]
    active_candle_patterns = result["active_candle_patterns"]
    recent_chart_patterns = result["recent_chart_patterns"]

    logger.info(f">>> Technical Score: {score_result['score']}/100 | Bias: {score_result['direction']} <<<")
    logger.info(f"Detected Supports: {supports}")
    logger.info(f"Detected Resistances: {resistances}")
    if active_candle_patterns:
        logger.info(f"Active candle patterns on last candle: {active_candle_patterns}")
    if recent_chart_patterns:
        logger.info(f"Recent chart patterns: {[p['pattern'] for p in recent_chart_patterns]}")

    # ۵. مشورت با هوش مصنوعی (شناسایی و تحلیل خودکار مدل‌های فعال)
    ai = AIBrain()
    ai_opinions = ai.analyze_with_both_models(
        symbol, technical_summary, score_result,
        active_patterns=active_candle_patterns, chart_patterns=recent_chart_patterns
    )

    # ۶. چاپ گزارش نهایی و یکپارچه بدون کدهای دابل و اضافه
    print("\n" + "="*80)
    print(f"📊 TECHNICAL READ FOR {symbol} ({timeframe})")
    print("="*80)
    print(f"• Current Price: {technical_summary['close']}")
    print(f"• Technical Score: {score_result['score']}/100")
    print(f"• Indicator Bias: {score_result['direction']}  (⚠️ صرفاً خلاصه‌ی وضعیت فعلی اندیکاتورهاست، توصیه‌ی معاملاتی نیست)")
    if active_candle_patterns:
        print(f"• Candle Patterns (آخرین کندل): {', '.join(active_candle_patterns)}")
    if recent_chart_patterns:
        pattern_strs = [f"{p['pattern']} @ {p['price']}" for p in recent_chart_patterns]
        print(f"• Recent Chart Patterns: {', '.join(pattern_strs)}")
    print(f"• Supports: {supports}")
    print(f"• Resistances: {resistances}")
    print("-" * 80)
    
    model_1_title = ai_opinions.get("model_1_name", "MODEL 1")
    print(f"🧠 {model_1_title.upper()} ANALYSIS:")
    print(ai_opinions.get("model_1_opinion"))
    print("-" * 80)
    
    model_2_title = ai_opinions.get("model_2_name", "MODEL 2")
    if model_2_title != "None":
        print(f"🧠 {model_2_title.upper()} ANALYSIS:")
        print(ai_opinions.get("model_2_opinion"))
        print("-" * 80)
        
    print("="*80 + "\n")

if __name__ == "__main__":
    target_symbol = "BTCUSDT"
    target_timeframe = "4h"
    run_analysis_pipeline(target_symbol, target_timeframe)