"""
backtest.py
------------
موتور بک‌تست ساده اما واقعی برای سنجش عملکرد SignalScorer روی داده‌ی تاریخی.

ایده:
1. داده‌ی تاریخی رو می‌گیریم (تا ۱۰۰۰ کندل، محدودیت API بایننس).
2. تمام اندیکاتورها رو یک‌بار روی کل دیتافریم محاسبه می‌کنیم (برداری/سریع).
3. برای تک‌تک کندل‌ها (از یک نقطه‌ی warm-up به بعد) اسکور همون لحظه رو با
   SignalScorer.score_row حساب می‌کنیم -> این دقیقاً همون منطقیه که در
   حالت زنده (main.py/multi_run.py) استفاده میشه، بدون هیچ تفاوتی.
4. وقتی سیگنال BUY/SELL ظاهر بشه و پوزیشنی باز نباشه، در کندلِ بعدی وارد
   معامله می‌شیم (برای جلوگیری از look-ahead bias). خروج بر اساس Stop-Loss/
   Take-Profit مبتنی بر ATR انجام میشه، ی�� با ظاهر شدن سیگنال مخالف.
5. آمار استاندارد: تعداد معاملات، Win Rate، R میانگین، Expectancy،
   Profit Factor و Max Drawdown محاسبه و چاپ میشه.

⚠️ این ابزار فقط برای سنجش «آیا منطق اسکورینگ فعلی از نظر آماری روی داده‌ی
گذشته معنی‌دار بوده یا نه» است. نتیجه‌ی خوب در بک‌تست هیچ تضمینی برای آینده
نیست (overfitting، تغییر رژیم بازار، کارمزد/اسلیپیج واقعی و ...در نظر گرفته
نشده مگر صریحاً ذکر بشه).
"""

import argparse
import logging
import pandas as pd

from data.binance_client import BinanceDataClient
from indicators.trend import TrendIndicators
from indicators.momentum import MomentumIndicators
from indicators.volatility import VolatilityIndicators
from indicators.volume import VolumeIndicators
from strategies.scoring import SignalScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Backtest")


def prepare_dataframe(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """دریافت داده و محاسبه‌ی یک‌بارِ همه‌ی اندیکاتورها روی کل تاریخچه."""
    try:
        client = BinanceDataClient()
        df = client.get_historical_klines(symbol, timeframe, limit=limit)
    except Exception as e:
        raise RuntimeError(f"خطا در دریافت داده برای {symbol}/{timeframe}: {e}")

    if df.empty:
        raise RuntimeError(f"داده‌ای برای {symbol}/{timeframe} دریافت نشد.")

    df = TrendIndicators.calculate_all_trend(df)
    df = MomentumIndicators.calculate_all_momentum(df)
    df = VolatilityIndicators.calculate_all_volatility(df)
    df = VolumeIndicators.calculate_all_volume(df)
    return df


def run_backtest(
    df: pd.DataFrame,
    atr_sl_mult: float = 1.5,
    risk_reward: float = 2.0,
    warmup: int = 210,
    buy_threshold: float = 70,
    sell_threshold: float = 30,
    regime_adx_min: float = 0,
    enable_regime_filter: bool = None,
) -> dict:
    """
    اجرای شبیه‌سازی معاملات بر اساس سیگنال‌های SignalScorer.

    :param atr_sl_mult: فاصله‌ی حد ضرر بر حسب ضریبی از ATR
    :param risk_reward: نسبت ریسک به ریوارد برای تعیین حد سود (TP = SL_distance * risk_reward)
    :param warmup: تعداد کندل ابتدایی که به اندیکاتورهای کندتر (مثل EMA200) زمان میدیم پایدار بشن
    :param buy_threshold: آستانه‌ی امتیاز برای صدور سیگنال خرید
    :param sell_threshold: آستانه‌ی امتیاز برای صدور سیگنال فروش
    :param regime_adx_min: اگر > 0 باشه، فقط وقتی ADX_14 >= این مقدار (یعنی بازار روند‌دار
                            تشخیص داده بشه) اجازه‌ی ورود به معامله صادر میشه. ۰ = بدون فیلتر رژیم.
    :param enable_regime_filter: فعال/غیرفعال کردن فیلترهای پیشرفته RegimeFilter در scoring.py
                                 اگر None باشه، خودکار بر اساس regime_adx_min تنظیم میشه.
    """
    if len(df) <= warmup + 2:
        raise RuntimeError("داده‌ی کافی برای بک‌تست (بعد از warm-up) وجود ندارد.")

    # اگر enable_regime_filter مشخص نشده، خودکار تنظیم می‌کنیم
    if enable_regime_filter is None:
        enable_regime_filter = (regime_adx_min > 0)

    trades = []
    position = None  # dict: direction, entry_price, sl, tp, entry_index, entry_time

    rows = df.reset_index()  # ستون timestamp رو به عنوان ستون عادی برمی‌گردونیم برای دسترسی راحت

    for i in range(warmup, len(rows) - 1):
        row = rows.iloc[i]
        next_row = rows.iloc[i + 1]  # ورود/خروج در کندل بعدی انجام میشه تا look-ahead نداشته باشیم

        atr = row.get("ATR", None)
        if atr is None or pd.isna(atr) or atr <= 0:
            continue

        # --- اگر پوزیشنی باز است، بررسی خروج ---
        if position is not None:
            exit_price = None
            exit_reason = None

            if position["direction"] == "BUY":
                if next_row["low"] <= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"
                elif next_row["high"] >= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"
            else:  # SELL (short)
                if next_row["high"] >= position["sl"]:
                    exit_price, exit_reason = position["sl"], "SL"
                elif next_row["low"] <= position["tp"]:
                    exit_price, exit_reason = position["tp"], "TP"

            if exit_price is not None:
                risk = abs(position["entry_price"] - position["sl"])
                if position["direction"] == "BUY":
                    pnl = exit_price - position["entry_price"]
                else:
                    pnl = position["entry_price"] - exit_price
                r_multiple = pnl / risk if risk > 0 else 0.0

                trades.append({
                    "direction": position["direction"],
                    "entry_time": position["entry_time"],
                    "exit_time": next_row["timestamp"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "r_multiple": r_multiple,
                })
                position = None
                continue  # همون کندل بعدی نمی‌تونه هم‌زمان معامله‌ی جدید باز کنه

        # --- اگر پوزیشنی باز نیست، بررسی سیگنال ورود ---
        if position is None:
            # فیلتر رژیم بازار (اختیاری - legacy): فقط وقتی روند به‌اندازه‌ی کافی قویه وارد شو
            if regime_adx_min > 0:
                adx_val = row.get("ADX_14", 0)
                if pd.isna(adx_val) or adx_val < regime_adx_min:
                    continue

            # محاسبه اسکور با فیلترهای پیشرفته (اگر فعال باشه)
            score_result = SignalScorer.score_row(
                row, 
                buy_threshold=buy_threshold, 
                sell_threshold=sell_threshold,
                enable_regime_filter=enable_regime_filter,
                min_adx=regime_adx_min if regime_adx_min > 0 else 25.0
            )
            direction = score_result["direction"]

            if direction in ("BUY", "SELL"):
                entry_price = next_row["open"]
                sl_distance = atr_sl_mult * atr
                if direction == "BUY":
                    sl = entry_price - sl_distance
                    tp = entry_price + sl_distance * risk_reward
                else:
                    sl = entry_price + sl_distance
                    tp = entry_price - sl_distance * risk_reward

                position = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "entry_time": next_row["timestamp"],
                }

    return compute_stats(trades)


def compute_stats(trades: list) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "message": "هیچ معامله‌ای طی بازه‌ی بک‌تست رخ نداد (احتمالاً سیگنال‌ها خیلی محدودن یا warm-up زیاده)."
        }

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades["r_multiple"] > 0]
    losses = df_trades[df_trades["r_multiple"] <= 0]

    total = len(df_trades)
    win_rate = len(wins) / total * 100
    avg_r = df_trades["r_multiple"].mean()
    expectancy = avg_r  # چون بر حسب R نرمالایز شده، میانگین R = expectancy به ازای هر معامله

    gross_profit = wins["r_multiple"].sum()
    gross_loss = abs(losses["r_multiple"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # منحنی سرمایه فرضی با ریسک ثابت ۱٪ به ازای هر معامله، برای محاسبه‌ی Max Drawdown
    equity = [1.0]
    for r in df_trades["r_multiple"]:
        equity.append(equity[-1] * (1 + 0.01 * r))
    equity_series = pd.Series(equity)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100

    return {
        "total_trades": total,
        "win_rate_pct": round(win_rate, 2),
        "avg_r_multiple": round(avg_r, 3),
        "expectancy_per_trade_R": round(expectancy, 3),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "max_drawdown_pct_at_1pct_risk": round(max_drawdown_pct, 2),
        "wins": len(wins),
        "losses": len(losses),
        "trades_detail": df_trades,
    }


def print_report(symbol: str, timeframe: str, stats: dict):
    print("\n" + "=" * 80)
    print(f"📈 BACKTEST REPORT — {symbol} ({timeframe})")
    print("=" * 80)

    if stats.get("total_trades", 0) == 0:
        print(stats.get("message"))
        print("=" * 80 + "\n")
        return

    print(f"• تعداد معاملات:            {stats['total_trades']}")
    print(f"• درصد برد (Win Rate):      {stats['win_rate_pct']}%  ({stats['wins']} برد / {stats['losses']} باخت)")
    print(f"• میانگین R هر معامله:      {stats['avg_r_multiple']}")
    print(f"• Expectancy (به ازای R):   {stats['expectancy_per_trade_R']}")
    print(f"• Profit Factor:            {stats['profit_factor']}")
    print(f"• حداکثر افت سرمایه (۱٪ ریسک ثابت): {stats['max_drawdown_pct_at_1pct_risk']}%")
    print("-" * 80)
    print("⚠️ توجه: این نتیجه کارمزد صرافی، اسلیپیج، و تغییر رژیم بازار در آینده رو")
    print("   لحاظ نکرده. عدد مثبت در بک‌تست تضمینی برای سود آینده نیست.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest SignalScorer روی داده‌ی تاریخی Binance")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=1000, help="تعداد کندل تاریخی (حداکثر معمولاً ۱۰۰۰)")
    parser.add_argument("--atr-mult", type=float, default=1.5, help="ضریب ATR برای فاصله‌ی حد ضرر")
    parser.add_argument("--rr", type=float, default=2.0, help="نسبت ریسک به ریوارد (TP نسبت به SL)")
    args = parser.parse_args()

    logger.info(f"Fetching & preparing data for {args.symbol} ({args.timeframe})...")
    df = prepare_dataframe(args.symbol, args.timeframe, limit=args.limit)

    logger.info("Running backtest simulation...")
    stats = run_backtest(df, atr_sl_mult=args.atr_mult, risk_reward=args.rr)

    print_report(args.symbol, args.timeframe, stats)
