"""
compare_filters.py
------------------
مقایسه‌ی عملکرد سیستم با و بدون فیلترهای رژیم بازار

این اسکریپت:
1. بهترین 4 سناریوی STABLE رو از Walk-Forward می‌گیره
2. هر کدوم رو دوبار بک‌تست می‌کنه:
   - بدون فیلتر (enable_regime_filter=False)
   - با فیلتر (enable_regime_filter=True)
3. نتایج رو در یک جدول مقایسه‌ای نشون می‌ده

هدف: اثبات این‌که فیلترها سیگنال‌های بی‌کیفیت رو حذف می‌کنن
و در نتیجه win rate و avg R بهتر میشه (با تعداد معاملات کمتر).

اجرا:
    python compare_filters.py
    python compare_filters.py --symbols BTCUSDT ETHUSDT --timeframes 15m 1h
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

from backtest import run_backtest
from data.binance_client import BinanceDataClient
from core_pipeline import build_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CompareFilters")


def run_single_comparison(symbol: str, timeframe: str, rr: float, limit: int = 1000) -> dict:
    """
    اجرای بک‌تست روی یک سناریو با دو حالت (با فیلتر و بدون فیلتر)
    
    Returns:
        dict با کلیدها:
        - without_filter: نتیجه بدون فیلتر
        - with_filter: نتیجه با فیلتر
    """
    logger.info(f"\n{'='*100}")
    logger.info(f"📊 Comparing {symbol}/{timeframe} (RR={rr}) — With vs Without Regime Filters")
    logger.info(f"{'='*100}")
    
    # دریافت داده‌ها (یکبار برای هر دو حالت)
    client = BinanceDataClient()
    df_raw = client.get_historical_klines(symbol=symbol, interval=timeframe, limit=limit)
    
    if df_raw.empty:
        logger.error(f"خطا: داده برای {symbol}/{timeframe} دریافت نشد.")
        return None
    
    # محاسبه اندیکاتورها (یکبار برای هر دو حالت)
    df_full = build_full_pipeline(df_raw)
    
    # ۱. بک‌تست بدون فیلتر
    logger.info("\n🔵 Running backtest WITHOUT regime filters...")
    results_no_filter = run_backtest(
        df=df_full,
        atr_multiplier=1.5,
        risk_reward_ratio=rr,
        enable_regime_filter=False,  # فیلترها خاموش
        buy_threshold=70.0,
        sell_threshold=30.0,
        min_regime_adx=25.0
    )
    
    # ۲. بک‌تست با فیلتر
    logger.info("\n🟢 Running backtest WITH regime filters...")
    results_with_filter = run_backtest(
        df=df_full,
        atr_multiplier=1.5,
        risk_reward_ratio=rr,
        enable_regime_filter=True,  # فیلترها روشن
        buy_threshold=70.0,
        sell_threshold=30.0,
        min_regime_adx=25.0
    )
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "rr": rr,
        "without_filter": results_no_filter,
        "with_filter": results_with_filter
    }


def calculate_improvement(without: dict, with_f: dict) -> dict:
    """
    محاسبه‌ی درصد بهبود بین دو حالت
    
    Returns:
        dict با کلیدها:
        - trades_change_pct: تغییر درصدی تعداد معاملات
        - win_rate_change_pct: تغییر درصدی win rate
        - avg_r_change_pct: تغییر درصدی avg R
    """
    if not without or not with_f:
        return {"trades_change_pct": 0, "win_rate_change_pct": 0, "avg_r_change_pct": 0}
    
    trades_no = without.get("total_trades", 0)
    trades_yes = with_f.get("total_trades", 0)
    trades_change = ((trades_yes - trades_no) / trades_no * 100) if trades_no > 0 else 0
    
    wr_no = without.get("win_rate", 0)
    wr_yes = with_f.get("win_rate", 0)
    wr_change = (wr_yes - wr_no) if wr_no > 0 else 0
    
    avgr_no = without.get("avg_r", 0)
    avgr_yes = with_f.get("avg_r", 0)
    avgr_change = ((avgr_yes - avgr_no) / abs(avgr_no) * 100) if avgr_no != 0 else 0
    
    return {
        "trades_change_pct": round(trades_change, 1),
        "win_rate_change_pp": round(wr_change, 1),  # percentage points
        "avg_r_change_pct": round(avgr_change, 1)
    }


def print_comparison_table(results: list):
    """
    چاپ جدول مقایسه‌ای نتایج
    """
    print("\n" + "="*150)
    print("🔬 REGIME FILTER COMPARISON — Impact on Trade Quality")
    print("="*150)
    
    rows = []
    for res in results:
        if not res:
            continue
        
        symbol = res["symbol"]
        tf = res["timeframe"]
        rr = res["rr"]
        
        no_f = res["without_filter"]
        yes_f = res["with_filter"]
        
        if not no_f or not yes_f:
            continue
        
        improvement = calculate_improvement(no_f, yes_f)
        
        rows.append({
            "Scenario": f"{symbol}/{tf} RR={rr}",
            
            # بدون فیلتر
            "NoFilter_Trades": no_f.get("total_trades", 0),
            "NoFilter_WinRate": round(no_f.get("win_rate", 0), 1),
            "NoFilter_AvgR": round(no_f.get("avg_r", 0), 3),
            "NoFilter_PF": round(no_f.get("profit_factor", 0), 2),
            
            # با فیلتر
            "WithFilter_Trades": yes_f.get("total_trades", 0),
            "WithFilter_WinRate": round(yes_f.get("win_rate", 0), 1),
            "WithFilter_AvgR": round(yes_f.get("avg_r", 0), 3),
            "WithFilter_PF": round(yes_f.get("profit_factor", 0), 2),
            
            # بهبود
            "Improvement_Trades": improvement["trades_change_pct"],
            "Improvement_WinRate": improvement["win_rate_change_pp"],
            "Improvement_AvgR": improvement["avg_r_change_pct"]
        })
    
    df_results = pd.DataFrame(rows)
    
    if df_results.empty:
        print("هیچ نتیجه‌ای برای نمایش وجود نداره.")
        print("="*150 + "\n")
        return
    
    # چاپ جدول اصلی
    print("\n📊 Detailed Results:")
    print(df_results.to_string(index=False))
    print("-"*150)
    
    # خلاصه‌ی بهبودها
    avg_trades_change = df_results["Improvement_Trades"].mean()
    avg_wr_change = df_results["Improvement_WinRate"].mean()
    avg_avgr_change = df_results["Improvement_AvgR"].mean()
    
    positive_avgr = (df_results["Improvement_AvgR"] > 0).sum()
    total_scenarios = len(df_results)
    
    print(f"\n📈 Summary of Improvements (Average across all scenarios):")
    print(f"  • Trades:    {avg_trades_change:+.1f}% (کاهش = خوب، چون سیگنال‌های بی‌کیفیت حذف شدن)")
    print(f"  • Win Rate:  {avg_wr_change:+.1f} percentage points")
    print(f"  • Avg R:     {avg_avgr_change:+.1f}%")
    print(f"  • Scenarios with improved Avg R: {positive_avgr}/{total_scenarios} ({positive_avgr/total_scenarios*100:.0f}%)")
    
    # تفسیر نتایج
    print(f"\n💡 Interpretation:")
    if avg_avgr_change > 10 and avg_wr_change > 5:
        print("  ✅ فیلترها تاثیر مثبت قابل‌توجهی داشتن! win rate و avg R هر دو بهبود یافتن.")
        print("     توصیه می‌شه فیلترها رو روشن نگه داری.")
    elif avg_avgr_change > 0:
        print("  🟡 فیلترها تاثیر مثبت جزئی داشتن، ولی نه خیلی قوی.")
        print("     ممکنه نیاز به تنظیم بیشتر آستانه‌ها (مثلاً کاهش min_adx) باشه.")
    else:
        print("  ❌ فیلترها در این داده‌ها بهبود قابل‌توجهی نداشتن.")
        print("     احتمالاً باید فیلترهای دیگه‌ای رو امتحان کنیم یا پارامترها رو تنظیم کنیم.")
    
    print("="*150 + "\n")
    
    # ذخیره نتایج
    output_path = "logs/filter_comparison.csv"
    Path("logs").mkdir(exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"نتایج کامل در {output_path} ذخیره شد.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="مقایسه‌ی عملکرد با و بدون فیلتر رژیم بازار")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="لیست نمادها برای تست")
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h"],
                        help="لیست تایم‌فریم‌ها برای تست")
    parser.add_argument("--rrs", nargs="+", type=float, default=[1.5, 2.0, 3.0],
                        help="لیست Risk:Reward ratios برای تست")
    parser.add_argument("--limit", type=int, default=1000,
                        help="تعداد کندل‌های تاریخی")
    args = parser.parse_args()
    
    # لیست سناریوهای ثابت (بر اساس نتایج Walk-Forward)
    stable_scenarios = [
        {"symbol": "BTCUSDT", "timeframe": "1h", "rr": 3.0},
        {"symbol": "ETHUSDT", "timeframe": "15m", "rr": 1.5},
        {"symbol": "ETHUSDT", "timeframe": "15m", "rr": 3.0},
        {"symbol": "BTCUSDT", "timeframe": "15m", "rr": 2.0},
    ]
    
    # اجرای مقایس�� روی همه‌ی سناریوها
    all_results = []
    
    for scenario in stable_scenarios:
        try:
            result = run_single_comparison(
                symbol=scenario["symbol"],
                timeframe=scenario["timeframe"],
                rr=scenario["rr"],
                limit=args.limit
            )
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"خطا در پردازش {scenario}: {str(e)}")
    
    # چاپ گزارش نهایی
    if all_results:
        print_comparison_table(all_results)
    else:
        logger.error("هیچ نتیجه‌ای برای نمایش وجود نداره.")
