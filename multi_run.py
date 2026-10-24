import os
import sys
import time
import logging

# غیرفعال کردن پراکسی برای جلوگیری از ارورهای ۵۰۳ در ویندوز
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("MultiAnalyzer")

try:
    from strategies.ai_brain import AIBrain
    from core_pipeline import analyze_symbol, AnalysisError
except ImportError as e:
    logger.error(f"Import Error: {str(e)}. مطمئن شوید در پوشه اصلی پروژه هستید.")
    sys.exit(1)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]

MAX_RETRIES = 3


def run_analysis():
    logger.info("============================================================")
    logger.info("STARTING MULTI-SYMBOL & MULTI-TIMEFRAME ANALYSIS")
    logger.info("============================================================")

    ai_brain = AIBrain()

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            logger.info(f"\n🔄 Analyzing {symbol} on {timeframe} timeframe...")

            result = None
            for attempt in range(MAX_RETRIES):
                try:
                    # پایپ‌لاین واقعی: دریافت داده + محاسبه‌ی همه‌ی اندیکاتورها + اسکورینگ واقعی
                    # (دیگه هیچ عدد ساختگی/hardcoded اینجا وجود نداره)
                    result = analyze_symbol(symbol, timeframe, klines_limit=250)
                    break
                except AnalysisError as e:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = (attempt + 1) * 3
                        logger.warning(
                            f"⚠️ عدم موفقیت در تحلیل: {str(e)[:120]}... "
                            f"تلاش مجدد {attempt + 1} از {MAX_RETRIES} پس از {wait_time} ثانیه..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"❌ خطای قطعی برای {symbol} ({timeframe}) پس از {MAX_RETRIES} بار تلاش: {str(e)}"
                        )
                except Exception as e:
                    # خطاهای غیرمنتظره (شبکه، پارس داده و...) هم با همون منطق retry مدیریت میشن
                    if attempt < MAX_RETRIES - 1:
                        wait_time = (attempt + 1) * 3
                        logger.warning(
                            f"⚠️ خطای غیرمنتظره: {str(e)[:120]}... "
                            f"تلاش مجدد {attempt + 1} از {MAX_RETRIES} پس از {wait_time} ثانیه..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"❌ خطای قطعی برای {symbol} ({timeframe}) پس از {MAX_RETRIES} بار تلاش: {str(e)}"
                        )

            if result is None:
                logger.warning(f"❌ Skipped analysis for {symbol} - {timeframe} due to lack of data.")
                continue

            technical_summary = result["technical_summary"]
            market_profile = result["market_profile"]
            score_result = result["score_result"]
            active_candle_patterns = result.get("active_candle_patterns", [])
            recent_chart_patterns = result.get("recent_chart_patterns", [])

            try:
                # ارسال داده‌ی واقعی (نه ساختگی) به LM Studio
                opinions = ai_brain.analyze_with_both_models(
                    symbol=symbol,
                    technical_summary=technical_summary,
                    market_profile=market_profile,
                    active_patterns=active_candle_patterns,
                    chart_patterns=recent_chart_patterns
                )

                dim_label_fa = {"trend": "روند", "momentum": "مومنتوم", "volatility": "نوسان", "volume": "حجم/نقدینگی"}

                print("\n" + "=" * 80)
                print(f"📊 MARKET PROFILE FOR {symbol} ({timeframe})")
                print("=" * 80)
                print(f"• Current Price: {technical_summary['close']}")
                print(f"• Regime: {market_profile['regime']}")
                print("-" * 80)
                for dim_name, dim_data in market_profile["dimensions"].items():
                    print(f"  • {dim_label_fa.get(dim_name, dim_name):12s}: {dim_data['label']}  (امتیاز خام: {dim_data['score']})")
                print("-" * 80)
                print(f"🔎 هم‌جهتی ابعاد: {market_profile['agreement']['label']}")
                if market_profile["uncertainty_note"]:
                    print(f"   ⚠️ {market_profile['uncertainty_note']}")
                if active_candle_patterns:
                    print(f"• Candle Patterns: {', '.join(active_candle_patterns)}")
                if recent_chart_patterns:
                    pattern_strs = [f"{p['pattern']} @ {p['price']}" for p in recent_chart_patterns]
                    print(f"• Recent Chart Patterns: {', '.join(pattern_strs)}")
                print(f"(مرجع تاریخی: امتیاز فرمول قدیمی = {score_result['score']}/100 — edge اثبات‌شده ندارد)")
                print("-" * 80)
                print(f"🧠 {opinions.get('model_1_name', 'MODEL 1')} ANALYSIS:")
                print(opinions.get('model_1_opinion', 'None'))
                print("-" * 80)
                print(f"🧠 {opinions.get('model_2_name', 'MODEL 2')} ANALYSIS:")
                print(opinions.get('model_2_opinion', 'None'))
                print("=" * 80 + "\n")

                # تاخیر کوچک برای عدم فشار روی شبکه و LM Studio
                time.sleep(1.5)

            except Exception as e:
                logger.error(f"Error during AI analysis of {symbol} on {timeframe}: {str(e)}")


if __name__ == "__main__":
    run_analysis()
