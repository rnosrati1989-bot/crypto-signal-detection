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
    market_profile = result["market_profile"]
    score_result = result["score_result"]
    supports = result["supports"]
    resistances = result["resistances"]
    active_candle_patterns = result["active_candle_patterns"]
    recent_chart_patterns = result["recent_chart_patterns"]

    logger.info(f">>> Regime: {market_profile['regime']} | Agreement: {market_profile['agreement']['label']} <<<")
    logger.info(f"Detected Supports: {supports}")
    logger.info(f"Detected Resistances: {resistances}")
    if active_candle_patterns:
        logger.info(f"Active candle patterns on last candle: {active_candle_patterns}")
    if recent_chart_patterns:
        logger.info(f"Recent chart patterns: {[p['pattern'] for p in recent_chart_patterns]}")

    # ۵. مشورت با هوش مصنوعی (شناسایی و تحلیل خودکار مدل‌های فعال)
    # دریافت آخرین وضعیت Order Book (اگر تازه باشد) — برای تفسیر AI
    from orderbook import load_latest_orderbook_features
    ob_context = load_latest_orderbook_features(symbol, max_age_seconds=300)
    if ob_context:
        logger.info(f"Order Book context: imbalance_20={ob_context['imbalance_20']:.2f} "
                    f"| depth_ratio_20={ob_context['depth_ratio_20']:.2f}")
    else:
        logger.info("Order Book context: در دسترس نیست (فایل/داده‌ی تازه موجود نیست)")

    ai = AIBrain()
    ai_opinions = ai.analyze_with_consensus(
        symbol, technical_summary, market_profile,
        active_patterns=active_candle_patterns, chart_patterns=recent_chart_patterns,
        orderbook_context=ob_context
    )

    # ۶. چاپ گزارش نهایی: پروفایل چندبعدی، نه یک تصمیم قطعی
    print("\n" + "="*80)
    print(f"📊 MARKET PROFILE FOR {symbol} ({timeframe})")
    print("="*80)
    print(f"• Current Price: {technical_summary['close']}")
    print(f"• Regime: {market_profile['regime']}")
    print("-" * 80)
    print("این‌ها توصیفِ وضعیت فعلی هر بُعد بازارن، نه توصیه‌ی معامله:")
    for dim_name, dim_data in market_profile["dimensions"].items():
        dim_label_fa = {"trend": "روند", "momentum": "مومنتوم", "volatility": "نوسان", "volume": "حجم/نقدینگی"}
        print(f"  • {dim_label_fa.get(dim_name, dim_name):12s}: {dim_data['label']}  (امتیاز خام: {dim_data['score']})")
    print("-" * 80)
    print(f"🔎 هم‌جهتی ابعاد: {market_profile['agreement']['label']}")
    print(f"   (صعودی: {market_profile['agreement']['bullish_dimensions']}/4 | "
          f"نزولی: {market_profile['agreement']['bearish_dimensions']}/4 | "
          f"خنثی: {market_profile['agreement']['neutral_dimensions']}/4)")
    if market_profile["uncertainty_note"]:
        print(f"   ⚠️ {market_profile['uncertainty_note']}")
    print("-" * 80)
    if active_candle_patterns:
        print(f"• Candle Patterns (آخرین کندل): {', '.join(active_candle_patterns)}")
    if recent_chart_patterns:
        pattern_strs = [f"{p['pattern']} @ {p['price']}" for p in recent_chart_patterns]
        print(f"• Recent Chart Patterns: {', '.join(pattern_strs)}")
    print(f"• Supports: {supports}")
    print(f"• Resistances: {resistances}")
    print("-" * 80)
    print(f"(برای مرجع تاریخی/بک‌تست: امتیاز فرمول قدیمی = {score_result['score']}/100 — "
          f"این عدد طبق Walk-Forward Validation edge اثبات‌شده‌ای نداره، فقط برای شفافیت نمایش داده میشه)")
    print("-" * 80)

    # Technical Digest — مقادیر نرمال/ایستای آخرین کندل (همونی که به LLM هم فرستاده می‌شود)
    print("📊 Technical Digest (مقادیر نرمال/ایستا — توصیفی، نه سیگنال):")
    _digest = technical_summary.get("digest") or {}
    _digest_fa = {
        "ema9_dist": "فاصله از EMA9", "ema21_dist": "فاصله از EMA21",
        "ema50_dist": "فاصله از EMA50", "ema200_dist": "فاصله از EMA200",
        "ema9_21_cross": "کراس EMA9/21", "atr_pct": "ATR٪",
        "vol_regime": "رژیم نوسان", "range_pos": "جایگاه در رنج",
        "vol_z": "حجم Z", "obv_z": "جریان پول Z",
        "macdh_norm": "MACDh/ATR", "macdh_slope": "شیب MACDh",
        "rsi_slope": "شیب RSI", "vwap_dist": "فاصله از VWAP",
        "kc_pos": "جایگاه کلتنر", "dc_pos": "جایگاه دونچیان",
        "psar_dist_atr": "فاصله از PSAR", "st_dir": "جهت سوپرترند",
        "pat_net_3": "الگو (۳ کندل)", "pat_net_12": "الگو (۱۲ کندل)",
    }
    for _k, _v in _digest.items():
        _label = _digest_fa.get(_k, _k)
        print(f"   • {_label:16s}: {_v if _v is not None else 'n/a'}")
    print("-" * 80)

    # جایگاه ابزار — توصیفی/آموزشی، نه معامله
    print("⚠️ جایگاه این ابزار: تحلیلگر توصیفی/آموزشی است. در دو اجرای Walk-Forward "
          "(مجموعاً ~۶۸۵ معامله، بعد از کارمزد و فاندینگ) edge معاملاتی اثبات نشد؛ "
          "هیچ‌جز از این خروجی توصیه‌ی معامله با پول واقعی نیست.")
    print("-" * 80)

    if ob_context:
        print(f"📗 Order Book (زنده): imbalance_20={ob_context['imbalance_20']:.2f} | "
              f"depth_ratio_20={ob_context['depth_ratio_20']:.2f} | "
              f"spread_pct={ob_context['spread_pct']:.2e}")
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
        
    consensus = ai_opinions.get("consensus")
    if consensus:
        print("\n🧩 ENSEMBLE CONSENSUS:")
        print(consensus.get("summary", ""))
        watch_items = ai_opinions.get("watch_items", [])
        if watch_items:
            print("   👀 نکات قابل مشاهده برای کاهش عدم قطعیت:")
            for item in watch_items:
                print(f"      • {item}")
        print("-" * 80)

    # ۷. سیگنال زنده‌ی مدل LGBM — مستقل از نظر LLM (فقط آخرین کندل بسته‌شده)
    # حلقه‌ی گم‌شده‌ی قبلی: مدل‌ها آموزش دیده بودند ولی در مسیر زنده هرگز
    # اجرا نمی‌شدند. این بخش مدل prod_15m را به داده‌ی فیوچرز زنده وصل می‌کند.
    try:
        from live_signal import build_live_signal, format_signal
        _sig = build_live_signal(symbol, interval="15m")
        print("-" * 80)
        print(format_signal(_sig))
        # ثبت خودکار در paper-trade log (بی‌صدا؛ خطای ثبت نباید تحلیل را بشکند)
        try:
            from signal_logger import append_signal
            append_signal(_sig)
        except Exception as log_e:
            logger.warning(f"ثبت سیگنال در signal_log ناموفق بود: {log_e}")
        print("⚠️ (مدل‌های تولیدی روی تایم‌فریم 15m آموزش دیده‌اند؛ این سیگنال "
              "مکمل تحلیل توصیفی بالاست و طبق Walk-Forward edge اثبات‌شده ندارد — "
              "فقط ورودی کمکی برای تصمیم انسانی.)")
    except Exception as e:
        print(f"🔔 سیگنال مدل در دسترس نیست: {e}")

    print("="*80 + "\n")

if __name__ == "__main__":
    target_symbol = "BTCUSDT"
    target_timeframe = "4h"
    run_analysis_pipeline(target_symbol, target_timeframe)