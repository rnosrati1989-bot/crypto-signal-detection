"""
auto_walk_forward.py
---------------------
اجرای خودکار Walk-Forward Validation روی بهترین سناریوهای شناسایی‌شده
از backtest_sweep.py تا ببینیم آیا edge پایداره یا فقط روی یک بازه خاص خوب بوده.

این اسکریپت:
1. بهترین سناریوها رو از sweep_results.csv می‌خونه
2. برای هرکدوم Walk-Forward می‌زنه (6 پنجره)
3. نتایج رو در یک جدول مقایسه‌ای نشون می‌ده
4. سناریوهایی که واقعاً پایدار بودن رو شناسایی می‌کنه

اجرا:
    python auto_walk_forward.py
    python auto_walk_forward.py --min-edge 5.0 --min-trades 15
"""

import argparse
import logging
import pandas as pd
import os
from pathlib import Path

from walk_forward import run_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AutoWalkForward")


def load_sweep_results(csv_path: str = "logs/sweep_results.csv") -> pd.DataFrame:
    """بارگذاری نتایج sweep از CSV"""
    if not os.path.exists(csv_path):
        logger.error(f"فایل {csv_path} پیدا نشد. ابتدا backtest_sweep.py رو اجرا کن.")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} scenarios from {csv_path}")
    return df


def filter_best_scenarios(df: pd.DataFrame, min_edge: float = 5.0, min_trades: int = 15) -> pd.DataFrame:
    """
    فیلتر سناریوهای برتر بر اساس:
    - edge_vs_breakeven > min_edge
    - trades >= min_trades
    - avg_r > 0
    """
    filtered = df[
        (df['edge_vs_breakeven'] > min_edge) &
        (df['trades'] >= min_trades) &
        (df['avg_r'] > 0)
    ].copy()
    
    # مرتب‌سازی بر اساس avg_r (بهترین‌ها اول)
    filtered = filtered.sort_values('avg_r', ascending=False)
    
    logger.info(f"Filtered to {len(filtered)} promising scenarios (edge > {min_edge}%, trades >= {min_trades})")
    return filtered


def run_walk_forward_batch(scenarios: pd.DataFrame, n_windows: int = 6) -> pd.DataFrame:
    """
    اجرای Walk-Forward روی لیست سناریوها و جمع‌آوری نتایج
    """
    results = []
    
    for idx, row in scenarios.iterrows():
        symbol = row['symbol']
        timeframe = row['timeframe']
        rr = row['rr']
        sweep_avg_r = row['avg_r']
        sweep_edge = row['edge_vs_breakeven']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Running Walk-Forward for {symbol}/{timeframe} (RR={rr})")
        logger.info(f"Sweep Results: Avg R = {sweep_avg_r:.3f}, Edge = {sweep_edge:.2f}%")
        logger.info(f"{'='*80}\n")
        
        try:
            # تعیین اندازه پنجره بر اساس تایم‌فریم
            if timeframe == "15m":
                train_size, test_size, step = 500, 200, 200
            elif timeframe == "1h":
                train_size, test_size, step = 400, 150, 150
            else:  # 4h
                train_size, test_size, step = 300, 100, 100

            # ⚠️ حالت «ترکیب ثابت»: دقیقاً همون پارامترهایی که در backtest_sweep.py
            # (atr_mult=1.5 پیش‌فرض، buy=70/sell=30 پیش‌فرض، بدون فیلتر رژیم) برای این
            # ردیف امتیاز خوب گرفته بودن رو، بدون هیچ بازبهینه‌سازی مخفی‌ای، مستقیم
            # روی هر پنجره تست می‌کنیم. اگر این کار رو نکنیم و بذاریم هر پنجره خودش
            # RR/thresholds رو از نو انتخاب کنه، دیگه داریم سوال متفاوتی رو جواب
            # می‌دیم و نتیجه دیگه به سناریوی مشخص‌شده در sweep ربطی نداره.
            fixed_params = {
                "atr_mult": 1.5,
                "rr": rr,
                "buy_threshold": 70,
                "sell_threshold": 30,
                "regime_adx": 0,
            }

            wf_results = run_walk_forward(
                symbol=symbol,
                timeframe=timeframe,
                train_size=train_size,
                test_size=test_size,
                step=step,
                n_windows=n_windows,
                min_trades=10,  # کمتر از sweep برای Walk-Forward (چون پنجره‌ها کوچیک‌ترن)
                fixed_params=fixed_params,
            )

            if wf_results and len(wf_results) > 0:
                df_wf = pd.DataFrame(wf_results)
                valid = df_wf.dropna(subset=["test_avg_r"])
                
                if len(valid) > 0:
                    positive_windows = (valid["test_avg_r"] > 0).sum()
                    total_windows = len(valid)
                    positive_ratio = positive_windows / total_windows * 100
                    
                    total_test_trades = valid["test_trades"].sum()
                    weighted_test_r = (valid["test_avg_r"] * valid["test_trades"]).sum() / total_test_trades if total_test_trades > 0 else 0
                    
                    results.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "rr": rr,
                        "sweep_avg_r": sweep_avg_r,
                        "sweep_edge": sweep_edge,
                        "wf_windows": total_windows,
                        "wf_positive_windows": positive_windows,
                        "wf_positive_ratio": round(positive_ratio, 1),
                        "wf_weighted_avg_r": round(weighted_test_r, 3),
                        "wf_total_trades": total_test_trades,
                        "verdict": "✅ STABLE" if positive_ratio >= 60 and weighted_test_r > 0.05 else 
                                   "🟡 MIXED" if positive_ratio >= 40 else "❌ UNSTABLE"
                    })
                else:
                    results.append({
                        "symbol": symbol, "timeframe": timeframe, "rr": rr,
                        "sweep_avg_r": sweep_avg_r, "sweep_edge": sweep_edge,
                        "wf_windows": 0, "wf_positive_windows": 0, "wf_positive_ratio": 0,
                        "wf_weighted_avg_r": None, "wf_total_trades": 0,
                        "verdict": "⚠️ NO_DATA"
                    })
        except Exception as e:
            logger.error(f"خطا در Walk-Forward برای {symbol}/{timeframe}: {str(e)}")
            results.append({
                "symbol": symbol, "timeframe": timeframe, "rr": rr,
                "sweep_avg_r": sweep_avg_r, "sweep_edge": sweep_edge,
                "wf_windows": 0, "wf_positive_windows": 0, "wf_positive_ratio": 0,
                "wf_weighted_avg_r": None, "wf_total_trades": 0,
                "verdict": f"❌ ERROR"
            })
    
    return pd.DataFrame(results)


def print_final_report(results: pd.DataFrame):
    """چاپ گزارش نهایی مقایسه‌ای"""
    print("\n" + "="*120)
    print("🚶 AUTO WALK-FORWARD VALIDATION REPORT — Stability Test of Best Scenarios")
    print("="*120)
    
    if results.empty:
        print("هیچ نتیجه‌ای برای نمایش وجود نداره.")
        print("="*120 + "\n")
        return
    
    print(results.to_string(index=False))
    print("-"*120)
    
    stable = results[results['verdict'] == "✅ STABLE"]
    mixed = results[results['verdict'] == "🟡 MIXED"]
    unstable = results[results['verdict'] == "❌ UNSTABLE"]
    
    print(f"\n📊 خلاصه:")
    print(f"  • ✅ STABLE (پایدار روی 60%+ پنجره‌ها): {len(stable)} سناریو")
    print(f"  • 🟡 MIXED (40-60% پنجره‌های مثبت):     {len(mixed)} سناریو")
    print(f"  • ❌ UNSTABLE (کمتر از 40% مثبت):       {len(unstable)} سناریو")
    
    if not stable.empty:
        print(f"\n🏆 بهترین سناریوهای پایدار (توصیه می‌شه برای استفاده واقعی):")
        best_stable = stable.sort_values('wf_weighted_avg_r', ascending=False).head(5)
        print(best_stable[['symbol', 'timeframe', 'rr', 'wf_positive_ratio', 'wf_weighted_avg_r', 'wf_total_trades']].to_string(index=False))
    else:
        print("\n⚠️ هیچ سناریوی کاملاً پایداری پیدا نشد.")
        print("   احتمالاً باید:")
        print("   1. پارامترهای اسکورینگ رو بیشتر تنظیم کنیم")
        print("   2. فیلتر رژیم بازار قوی‌تر اضافه کنیم")
        print("   3. استراتژی‌های مجزا (به‌جای یک فرمول واحد) بسازیم")
    
    print("="*120 + "\n")
    
    # ذخیره نتایج
    output_path = "logs/walk_forward_results.csv"
    results.to_csv(output_path, index=False)
    print(f"نتایج کامل در {output_path} ذخیره شد.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="اجرای خودکار Walk-Forward روی بهترین سناریوهای sweep")
    parser.add_argument("--sweep-csv", default="logs/sweep_results.csv", help="مسیر فایل نتایج sweep")
    parser.add_argument("--min-edge", type=float, default=5.0, help="حداقل edge_vs_breakeven برای انتخاب سناریو")
    parser.add_argument("--min-trades", type=int, default=15, help="حداقل تعداد معاملات برای انتخاب سناریو")
    parser.add_argument("--n-windows", type=int, default=6, help="تعداد پنجره‌های Walk-Forward")
    parser.add_argument("--max-scenarios", type=int, default=10, help="حداکثر تعداد سناریو برای تست (برای صرفه‌جویی وقت)")
    args = parser.parse_args()
    
    # ۱. بارگذاری نتایج sweep
    sweep_df = load_sweep_results(args.sweep_csv)
    if sweep_df.empty:
        exit(1)
    
    # ۲. فیلتر بهترین سناریوها
    best_scenarios = filter_best_scenarios(sweep_df, min_edge=args.min_edge, min_trades=args.min_trades)
    if best_scenarios.empty:
        logger.warning("هیچ سناریویی با معیارهای مشخص‌شده پیدا نشد. معیارها رو کمتر کن.")
        exit(1)
    
    # محدود کردن تعداد برای صرفه‌جویی وقت
    if len(best_scenarios) > args.max_scenarios:
        logger.info(f"محدود کردن به {args.max_scenarios} سناریوی برتر (از {len(best_scenarios)} سناریو)")
        best_scenarios = best_scenarios.head(args.max_scenarios)
    
    # ۳. اجرای Walk-Forward روی همه
    wf_results = run_walk_forward_batch(best_scenarios, n_windows=args.n_windows)
    
    # ۴. چاپ گزارش نهایی
    print_final_report(wf_results)
