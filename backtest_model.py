"""
backtest_model.py
-----------------
شبیه‌سازی واقع‌گرایانه‌ی معاملات با مدل — با در نظر گرفتن هزینه‌ها.

سؤال اصلی: «آیا بعد از کسر کارمزد صرافی و فاندینگ، سیگنال‌های مدل هنوز
ارزش دارند؟»

قوانین شبیه‌سازی:
1. مدل روی بخش Train (همه‌ی نمادها) آموزش می‌بیند و روی بخش Validation اجرا می‌شود.
2. ⚠️ شبیه‌سازی **به تفکیک هر نماد** انجام می‌شود (هر نماد سری زمانی خودش را دارد).
3. ورود: وقتی احتمال مدل >= آستانه (و فیلتر رژیم بازار اگر فعال باشد).
   ورود با قیمت بسته‌شدن همان کندل (بدون look-ahead).
4. خروج: دقیقاً horizon کندل بعد (افق پیش‌بینی مدل).
5. هزینه‌ها: کارمزد تیکر (ورود+خروج) + فاندینگ (رویدادهای ۸ ساعته).

نحوه‌ی اجرا (از داخل پوشه‌ی پروژه):
    python backtest_model.py --horizon 4 --threshold 0.25 --sweep-adx
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from features import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BacktestModel")

CACHE_DIR = "data_cache"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_INTERVAL = "15m"

DEFAULT_FEE = 0.0005  # 0.05% تیکر فیوچرز بایننس


def load_combined_dataset(symbols, interval):
    frames = []
    for symbol in symbols:
        path = os.path.join(CACHE_DIR, f"{symbol}_{interval}_features.csv")
        if not os.path.exists(path):
            logger.warning(f"فایل دیتاست پیدا نشد: {path} — رد شد.")
            continue
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        # ⚠️ خواندن CSV پهن + افزودن ستون نماد → pandas.DataFrame fragmentation
        # (PerformanceWarning). با concat(axis=1) یک فریمِ یک‌تکه ساخته می‌شود.
        df = pd.concat(
            [df, pd.Series(symbol, index=df.index, name="symbol")], axis=1)
        frames.append(df)

    if not frames:
        raise RuntimeError("هیچ دیتاست ویژگی‌ای پیدا نشد. اول build_dataset.py را اجرا کنید.")

    df = pd.concat(frames).sort_index()
    logger.info(f"کل داده: {len(df)} ردیف")
    return df


def simulate(df_symbol: pd.DataFrame, probs: np.ndarray, horizon: int,
             threshold: float, fee: float, min_adx: float,
             use_funding: bool) -> list:
    """
    شبیه‌سازی معاملات LONG روی داده‌ی یک نماد (سری زمانی خالص).
    :return: لیست معاملات با pnl_pct (درصد سود/ضرر خالص)
    """
    closes = df_symbol["close"].values
    adx_col = df_symbol["ADX_14"] if "ADX_14" in df_symbol.columns \
        else pd.Series(0.0, index=df_symbol.index)
    adx = adx_col.fillna(0).values
    funding = df_symbol["funding_rate"].fillna(0).values

    trades = []
    pos = None  # (entry_index, entry_price)

    n = len(df_symbol)
    for i in range(n - horizon):
        # --- خروج پوزیشن باز ---
        if pos is not None and i == pos[0] + horizon:
            exit_price = closes[i]
            raw_ret = exit_price / pos[1] - 1.0
            pnl = raw_ret - 2 * fee  # کارمزد ورود + خروج

            if use_funding:
                fcost = 0.0
                for j in range(pos[0] + 1, i + 1):
                    if funding[j] != funding[j - 1]:
                        fcost += funding[j]  # لانگ: فاندینگ مثبت = پرداخت
                pnl -= fcost

            trades.append({
                "symbol": df_symbol["symbol"].iloc[0],
                "entry_time": df_symbol.index[pos[0]],
                "exit_time": df_symbol.index[i],
                "raw_ret_pct": raw_ret * 100,
                "pnl_pct": pnl * 100,
            })
            pos = None

        # --- ورود پوزیشن جدید ---
        if pos is None:
            if probs[i] >= threshold and adx[i] >= min_adx:
                pos = (i, closes[i])

    return trades


def simulate_all(val_df: pd.DataFrame, val_probs: np.ndarray, horizon: int,
                 threshold: float, fee: float, min_adx: float,
                 use_funding: bool) -> list:
    """اجرای شبیه‌سازی به تفکیک نماد و ترکیب معاملات."""
    all_trades = []
    for symbol in val_df["symbol"].unique():
        mask = (val_df["symbol"] == symbol).values
        grp = val_df[mask]
        grp_probs = val_probs[mask]
        all_trades.extend(
            simulate(grp, grp_probs, horizon, threshold, fee, min_adx, use_funding)
        )
    return all_trades


def compute_stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0, "expectancy_pct": 0, "profit_factor": 0}

    pnls = np.array([t["pnl_pct"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "n": len(trades),
        "win_rate": (pnls > 0).mean() * 100,
        "expectancy_pct": pnls.mean(),
        "profit_factor": pf,
    }


def main():
    parser = argparse.ArgumentParser(description="بک‌تست واقع‌گرایانه‌ی مدل")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--min-adx", type=float, default=0.0,
                        help="فیلتر رژیم: حداقل ADX برای ورود (0 = خاموش)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--sweep-adx", action="store_true",
                        help="اجرای خودکار با چند مقدار ADX و مقایسه")
    parser.add_argument("--no-funding", action="store_true",
                        help="فاندینگ را در محاسبه لحاظ نکن")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("\n❌ lightgbm نصب نیست: pip install lightgbm scikit-learn\n")
        return

    df = load_combined_dataset([s.upper() for s in args.symbols], args.interval)

    target_col = f"target_up_{args.horizon}"
    if target_col not in df.columns:
        print(f"❌ ستون {target_col} در دیتاست نیست.")
        return

    feature_cols = get_feature_columns(df)
    if "symbol" not in feature_cols:
        feature_cols = feature_cols + ["symbol"]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    X["symbol"] = X["symbol"].astype("category")

    # تقسیم زمانی
    split_idx = int(len(X) * (1 - args.val_fraction))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    val_df = df.iloc[split_idx:].copy()

    # آموزش مدل (یک بار)
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_X=X_val, eval_y=y_val,
              callbacks=[lgb.early_stopping(40, verbose=False)])
    probs = model.predict_proba(X_val)[:, 1]

    # مرجع Buy & Hold به تفکیک نماد
    bh_lines = []
    for symbol in val_df["symbol"].unique():
        grp = val_df[val_df["symbol"] == symbol]
        bh = (grp["close"].iloc[-1] / grp["close"].iloc[0] - 1) * 100
        bh_lines.append(f"{symbol} {bh:+.2f}٪")
    bh_str = " | ".join(bh_lines)

    use_funding = not args.no_funding
    print("\n" + "=" * 80)
    print("💼 بک‌تست واقع‌گرایانه — مدل LightGBM (LONG)")
    print("=" * 80)
    print(f"• بازه‌ی Validation:      {val_df.index.min()} تا {val_df.index.max()}")
    print(f"• آستانه‌ی احتمال:       {args.threshold}")
    print(f"• افق نگه‌داری:          {args.horizon} کندل ({args.horizon*15} دقیقه)")
    print(f"• کارمزد (هر طرف):       {args.fee*100:.3f}٪")
    print(f"• فاندینگ:               {'لحاظ می‌شود' if use_funding else 'نادیده گرفته می‌شود'}")
    print(f"• Buy & Hold در بازه:    {bh_str}")

    if args.sweep_adx:
        print("\n" + "-" * 80)
        print("🔎 مقایسه‌ی فیلتر رژیم بازار (ADX) — به تفکیک هر نماد شبیه‌سازی شد:")
        print(f"    {'ADX≥':<6} {'تعداد':>6} {'WinRate':>8} {'Expectancy':>12} {'P.Factor':>9}")
        print("    " + "-" * 55)
        for madx in [0, 15, 20, 25, 30]:
            trades = simulate_all(val_df, probs, args.horizon, args.threshold,
                                  args.fee, float(madx), use_funding)
            st = compute_stats(trades)
            pf_str = f"{st['profit_factor']:.2f}" if st["profit_factor"] != float("inf") else "inf"
            print(f"    {madx:<6} {st['n']:>6} {st['win_rate']:>7.1f}% "
                  f"{st['expectancy_pct']:>+10.3f}٪ {pf_str:>9}")
        print("    " + "-" * 55)
        print("    (Expectancy مثبت بعد از همه‌ی هزینه‌ها = سیگنال‌ها ارزش واقعی دارند)")

    # اجرای نهایی با min_adx انتخابی
    trades = simulate_all(val_df, probs, args.horizon, args.threshold,
                          args.fee, args.min_adx, use_funding)
    st = compute_stats(trades)

    print("\n" + "=" * 80)
    print(f"📊 نتیجه‌ی نهایی (min_adx={args.min_adx}):")
    print("=" * 80)
    print(f"• تعداد معاملات:          {st['n']}")
    if st["n"] == 0:
        print("• هیچ معامله‌ای انجام نشد (آستانه/فیلتر خیلی سخت‌گیرانه).")
        return
    print(f"• Win Rate:               {st['win_rate']:.1f}٪")
    print(f"• Expectancy (هر معامله): {st['expectancy_pct']:+.3f}٪")
    print(f"• Profit Factor:          {st['profit_factor']:.2f}")
    print("-" * 80)
    if st["expectancy_pct"] > 0:
        print("✅ Expectancy مثبت بعد از هزینه‌ها — سیگنال‌ها ارزش واقعی دارند")
        print("   (هنوز نیاز به تأیید با Walk-Forward و داده‌ی بیشتر دارد)")
    else:
        print("🔴 Expectancy منفی بعد از هزینه‌ها — در این پیکربندی edge کافی نیست")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
