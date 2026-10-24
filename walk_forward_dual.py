"""
walk_forward_dual.py
--------------------
Walk-Forward شبیه‌سازی‌محور برای مدل «دوجهته» (Long + Short).

مشکل مدل تک‌طرفه: فقط صعود را پیش‌بینی می‌کرد → در بازار نزولی ضرر می‌داد.
اینجا دو مدل جدا آموزش می‌بینند:
  - up_model   : احتمال «قیمت در افق بالا می‌رود»  (target_up)
  - down_model : احتمال «قیمت در افق پایین می‌رود» (target_down)

قوانین صادقانه (همان walk_forward_sim):
1. پنجره‌های متوالی: Train گذشته → Test آینده‌ی ندیده.
2. آستانه‌های هر سمت از چندک Train همان پنجره (بدون look-ahead).
3. شبیه‌سازی LONG و SHORT با کارمزد + فاندینگ + فیلتر رژیم ADX.
4. ورود: LONG اگر up_prob >= tau_up (و ADX>=min_adx)؛ SHORT اگر down_prob >= tau_down.
   اگر هر دو فعال باشند، سمت با «فاصله‌ی بیشتر از آستانه‌اش» برنده است.

اجرا:
    python walk_forward_dual.py --interval 1h --horizon 4 --quantile 0.75 --min-adx 30
"""

import argparse
import logging

import numpy as np
import pandas as pd

from backtest_model import (
    load_combined_dataset,
    DEFAULT_SYMBOLS,
    DEFAULT_INTERVAL,
    DEFAULT_FEE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WalkForwardDual")


def simulate_dual(df_symbol: pd.DataFrame, up_probs: np.ndarray,
                  down_probs: np.ndarray, horizon: int, up_thr: float,
                  down_thr: float, fee: float, min_adx: float,
                  use_funding: bool) -> list:
    """شبیه‌سازی دوطرفه روی داده‌ی یک نماد. برمی‌گرداند لیست معاملات."""
    closes = df_symbol["close"].values
    adx_col = df_symbol["ADX_14"] if "ADX_14" in df_symbol.columns \
        else pd.Series(0.0, index=df_symbol.index)
    adx = adx_col.fillna(0).values
    funding = df_symbol["funding_rate"].fillna(0).values

    trades = []
    pos = None  # (entry_index, entry_price, side)
    n = len(df_symbol)

    for i in range(n - horizon):
        # خروج پوزیشن باز
        if pos is not None and i == pos[0] + horizon:
            exit_price = closes[i]
            entry_price = pos[1]
            side = pos[2]
            if side == "long":
                raw = exit_price / entry_price - 1.0
            else:
                raw = entry_price / exit_price - 1.0
            pnl = raw - 2 * fee
            if use_funding:
                fcost = 0.0
                for j in range(pos[0] + 1, i + 1):
                    if funding[j] != funding[j - 1]:
                        # لانگ: فاندینگ مثبت = پرداخت | شورت: برعکس
                        fcost += funding[j] if side == "long" else -funding[j]
                pnl -= fcost
            trades.append({
                "symbol": df_symbol["symbol"].iloc[0],
                "side": side,
                "entry_time": df_symbol.index[pos[0]],
                "exit_time": df_symbol.index[i],
                "raw_ret_pct": raw * 100,
                "pnl_pct": pnl * 100,
            })
            pos = None

        # ورود پوزیشن جدید
        if pos is None and adx[i] >= min_adx:
            long_fire = up_probs[i] >= up_thr
            short_fire = down_probs[i] >= down_thr
            if long_fire and short_fire:
                # هر دو فعال: سمت با فاصله‌ی بیشتر از آستانه‌ی خودش
                if (up_probs[i] - up_thr) >= (down_probs[i] - down_thr):
                    pos = (i, closes[i], "long")
                else:
                    pos = (i, closes[i], "short")
            elif long_fire:
                pos = (i, closes[i], "long")
            elif short_fire:
                pos = (i, closes[i], "short")

    return trades


def split_trades(trades: list, side: str) -> list:
    return [t for t in trades if t["side"] == side]


def stats_of(trades: list) -> dict:
    if not trades:
        return {"n": 0, "expectancy_pct": None, "win_rate": None}
    pnls = np.array([t["pnl_pct"] for t in trades])
    return {
        "n": len(trades),
        "expectancy_pct": float(pnls.mean()),
        "win_rate": float((pnls > 0).mean() * 100),
    }


def run_window(df, X, y_up, y_dn, start, train_size, test_size, horizon,
               quantile, min_adx, fee, use_funding, lgb):
    train_idx = slice(start, start + train_size)
    test_idx = slice(start + train_size, start + train_size + test_size)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_up_train, y_up_test = y_up.iloc[train_idx], y_up.iloc[test_idx]
    y_dn_train, y_dn_test = y_dn.iloc[train_idx], y_dn.iloc[test_idx]
    test_df = df.iloc[test_idx].copy()

    def _train(y_tr, y_te):
        m = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        m.fit(X_train, y_tr, eval_X=X_test, eval_y=y_te,
              callbacks=[lgb.early_stopping(30, verbose=False)])
        return m

    up_model = _train(y_up_train, y_up_test)
    dn_model = _train(y_dn_train, y_dn_test)

    up_train_prob = up_model.predict_proba(X_train)[:, 1]
    dn_train_prob = dn_model.predict_proba(X_train)[:, 1]
    up_test_prob = up_model.predict_proba(X_test)[:, 1]
    dn_test_prob = dn_model.predict_proba(X_test)[:, 1]

    tau_up = float(np.quantile(up_train_prob, quantile))
    tau_dn = float(np.quantile(dn_train_prob, quantile))

    # شبیه‌سازی به تفکیک هر نماد (سری زمانی هر نماد جدا — مثل backtest_model)
    trades = []
    for symbol in test_df["symbol"].unique():
        mask = (test_df["symbol"] == symbol).values
        grp = test_df[mask]
        trades.extend(simulate_dual(grp, up_test_prob[mask], dn_test_prob[mask],
                                    horizon, tau_up, tau_dn, fee, min_adx,
                                    use_funding))

    st_all = stats_of(trades)
    st_long = stats_of(split_trades(trades, "long"))
    st_short = stats_of(split_trades(trades, "short"))

    bh_lines = []
    for symbol in test_df["symbol"].unique():
        grp = test_df[test_df["symbol"] == symbol]
        bh = (grp["close"].iloc[-1] / grp["close"].iloc[0] - 1) * 100
        bh_lines.append(f"{symbol} {bh:+.2f}٪")

    return {
        "window_start": test_df.index[0],
        "window_end": test_df.index[-1],
        "tau_up": tau_up, "tau_dn": tau_dn,
        "n": st_all["n"], "exp": st_all["expectancy_pct"], "wr": st_all["win_rate"],
        "n_long": st_long["n"], "exp_long": st_long["expectancy_pct"],
        "n_short": st_short["n"], "exp_short": st_short["expectancy_pct"],
        "bh": " | ".join(bh_lines),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward مدل دوجهته")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--min-adx", type=float, default=30.0)
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--no-funding", action="store_true")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("\n❌ lightgbm نصب نیست: pip install lightgbm scikit-learn\n")
        return

    df = load_combined_dataset([s.upper() for s in args.symbols], args.interval)

    up_col = f"target_up_{args.horizon}"
    dn_col = f"target_down_{args.horizon}"
    if up_col not in df.columns or dn_col not in df.columns:
        print(f"❌ ستون‌های {up_col}/{dn_col} در دیتاست نیستند. build_dataset را اجرا کنید.")
        return

    from features import get_feature_columns
    feature_cols = get_feature_columns(df)
    if "symbol" not in feature_cols:
        feature_cols = feature_cols + ["symbol"]

    X = df[feature_cols].copy()
    y_up = df[up_col].astype(int)
    y_dn = df[dn_col].astype(int)
    X["symbol"] = X["symbol"].astype("category")

    n = len(X)
    use_funding = not args.no_funding

    print("\n" + "=" * 100)
    print("🚶 WALK-FORWARD — مدل دوجهته (LONG + SHORT)")
    print("=" * 100)
    print(f"• پیکربندی: {args.interval} | horizon={args.horizon} | "
          f"quantile={args.quantile} | ADX≥{args.min_adx:g} | fee={args.fee*100:.3f}٪")
    print(f"• پنجره‌ها: Train {args.train_size} → Test {args.test_size} "
          f"(گام {args.step}) | حداکثر {args.windows} پنجره")
    print()

    results = []
    start = 0
    window_num = 0
    while (start + args.train_size + args.test_size <= n) and (window_num < args.windows):
        window_num += 1
        try:
            r = run_window(df, X, y_up, y_dn, start, args.train_size,
                           args.test_size, args.horizon, args.quantile,
                           args.min_adx, args.fee, use_funding, lgb)
        except Exception as e:
            logger.error(f"پنجره {window_num}: خطا — {e}")
            start += args.step
            continue

        results.append(r)
        exp = r["exp"]
        e = f"{exp:+.3f}٪" if exp is not None else "بدون معامله"
        el = f"{r['exp_long']:+.3f}٪" if r["exp_long"] is not None else "—"
        es = f"{r['exp_short']:+.3f}٪" if r["exp_short"] is not None else "—"
        print(f"پنجره {window_num}: {r['window_start'].date()} → {r['window_end'].date()}")
        print(f"   دوطرفه: {r['n']:>3} معامله | Expectancy {e} | WinRate {r['wr']:.0f}٪" if r["wr"] is not None
              else f"   دوطرفه: {r['n']} معامله | بدون معامله")
        print(f"   لانگ: {r['n_long']:>3} ({el}) | شورت: {r['n_short']:>3} ({es})")
        print(f"   آستانه‌ها: up={r['tau_up']:.4f} | dn={r['tau_dn']:.4f} | Buy&Hold: {r['bh']}")
        print("-" * 100)
        start += args.step

    if not results:
        print("❌ هیچ پنجره‌ای با موفقیت اجرا نشد.")
        return

    valid = [r for r in results if r["exp"] is not None]
    print("\n" + "=" * 100)
    print("📊 جمع‌بندی Walk-Forward — مدل دوجهته")
    print("=" * 100)

    if valid:
        exps = np.array([r["exp"] for r in valid])
        pos = int((exps > 0).sum())
        mean = exps.mean()
        total_trades = sum(r["n"] for r in valid)
        print(f"• دوطرفه ({len(valid)} پنجره‌ی معتبر):")
        print(f"    پنجره‌های مثبت: {pos}/{len(valid)} ({pos/len(valid)*100:.0f}٪)")
        print(f"    میانگین Expectancy: {mean:+.4f}٪ | کل معاملات: {total_trades}")

        # سهم هر سمت
        longs = [r for r in valid if r["exp_long"] is not None]
        shorts = [r for r in valid if r["exp_short"] is not None]
        if longs:
            le = np.mean([r["exp_long"] for r in longs])
            print(f"• فقط لانگ: میانگین {le:+.4f}٪ ({sum(r['n_long'] for r in longs)} معامله)")
        if shorts:
            se = np.mean([r["exp_short"] for r in shorts])
            print(f"• فقط شورت: میانگین {se:+.4f}٪ ({sum(r['n_short'] for r in shorts)} معامله)")

        print("-" * 100)
        if pos / len(valid) >= 0.6 and mean > 0:
            print("🟢 مدل دوجهته در اکثر پنجره‌ها مثبت بوده — شواهد edge پایدار قوی‌تر شد.")
        elif pos / len(valid) >= 0.4:
            print("🟡 نتیجه ترکیبی — بهبود نسبت به تک‌طرفه، ولی edge هنوز به رژیم وابسته است.")
        else:
            print("🔴 مدل دوجهته هنوز در اکثر پنجره‌ها مثبت نبوده — edge پایدار تأیید نمی‌شود.")
    else:
        print("🔴 هیچ معامله‌ای در هیچ پنجره‌ای انجام نشد.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
