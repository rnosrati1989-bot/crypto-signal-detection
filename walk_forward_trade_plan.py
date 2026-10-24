"""
walk_forward_trade_plan.py
--------------------------
اعتبارسنجی Walk-Forward برای «پلن معاملاتی دقیق» (SL/TP) روی تایم‌فریم کوتاه.

تفاوت با walk_forward_dual.py (که خروج را پایانِ افق می‌گرفت):
اینجا خروج دقیقاً با قوانین strategies/trade_plan.py انجام می‌شود:

  - ورود   : قیمت بسته‌شدن کندلِ سیگنال (قرارداد بک‌تست‌های قبلی؛ بدون look-ahead)
  - SL     : ورود ∓ ATR × atr_sl_mult
  - TP     : ورود ± فاصله‌ی SL × risk_reward
  - اولویت : اگر در یک کندل هم SL و هم TP برخورد شود، SL مقدم (محافظه‌کارانه)
  - زمان‌سنج : اگر تا max_hold کندل به هدف نرسید، در بسته‌شدنِ همان کندل خروج
  - ریسک   : ثابت (هر معامله risk_pct از حساب — مثل پلن زنده)

سؤال این اعتبارسنجی: «آیا پلن دقیقِ ورود/SL/TP روی 15m بعد از کارمزد و
فاندینگ edge دارد؟» — نه فقط «مدل جهت را درست می‌گوید یا نه».

اجرا (از داخل پوشه‌ی پروژه، بعد از fetch + build_dataset):
    python walk_forward_trade_plan.py --interval 15m --horizon 4 \
           --quantile 0.75 --min-adx 30
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


# پیش‌فرض‌هایی که با strategies/trade_plan.py هماهنگ‌اند
DEFAULT_ATR_SL_MULT = 1.5
DEFAULT_RISK_REWARD = 2.0
DEFAULT_RISK_PCT = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WalkForwardTradePlan")


def simulate_plan(df_symbol: pd.DataFrame, up_probs: np.ndarray,
                  down_probs: np.ndarray, up_thr: float, dn_thr: float,
                  min_adx: float, atr_sl_mult: float, risk_reward: float,
                  max_hold: int, fee: float, use_funding: bool) -> list:
    """
    شبیه‌سازی معاملات با قوانین دقیق trade_plan روی داده‌ی یک نماد.
    :return: لیست معاملات (هر معامله شامل r_mult و net_pct).
    """
    n = len(df_symbol)
    closes = df_symbol["close"].values
    highs = df_symbol["high"].values
    lows = df_symbol["low"].values
    opens = (df_symbol["open"].values if "open" in df_symbol.columns
             else closes)

    atr = (df_symbol["ATR"].fillna(0).values
           if "ATR" in df_symbol.columns else np.zeros(n))
    adx = (df_symbol["ADX_14"].fillna(0).values
           if "ADX_14" in df_symbol.columns else np.zeros(n))
    funding = df_symbol["funding_rate"].fillna(0).values

    trades = []
    pos = None  # dict: entry_idx, side, entry, sl, tp
    pending = None  # سیگنالِ کندل قبل: (side, atr_سیگنال) — ورود در open کندل فعلی

    for i in range(n):
        # ---------- 🔒 ورود واقع‌بینانه: open کندلِ «بعد از» سیگنال ----------
        # (قبلاً با close همان کندل سیگنال وارد می‌شد که در عمل ممکن نیست)
        if pos is None and pending is not None:
            side, sig_atr = pending
            pending = None
            entry = opens[i]
            sd = sig_atr * atr_sl_mult
            if entry > 0 and sd > 0:
                sl = entry - sd if side == "long" else entry + sd
                tp = entry + sd * risk_reward if side == "long" else entry - sd * risk_reward
                pos = {"entry_idx": i, "side": side,
                       "entry": entry, "sl": sl, "tp": tp}

        # ---------- خروج پوزیشن باز ----------
        if pos is not None:
            side = pos["side"]
            hit_sl = (lows[i] <= pos["sl"]) if side == "long" else (highs[i] >= pos["sl"])
            hit_tp = (highs[i] >= pos["tp"]) if side == "long" else (lows[i] <= pos["tp"])
            time_up = (i - pos["entry_idx"]) >= max_hold

            if hit_sl and hit_tp:
                exit_price, exit_reason = pos["sl"], "SL"
            elif hit_sl:
                exit_price, exit_reason = pos["sl"], "SL"
            elif hit_tp:
                exit_price, exit_reason = pos["tp"], "TP"
            elif time_up:
                exit_price, exit_reason = closes[i], "TIME"
            else:
                exit_price = None

            if exit_price is not None:
                entry = pos["entry"]
                sl = pos["sl"]
                if side == "long":
                    raw_pct = (exit_price / entry - 1.0) * 100.0
                else:
                    raw_pct = (entry / exit_price - 1.0) * 100.0

                net_pct = raw_pct - 2 * fee * 100.0  # کارمزد ورود + خروج

                if use_funding:
                    fcost = 0.0
                    for j in range(pos["entry_idx"] + 1, i + 1):
                        if funding[j] != funding[j - 1]:
                            fcost += funding[j] if side == "long" else -funding[j]
                    net_pct -= fcost * 100.0

                # نسبت R: نسبت سود/زیان به ریسک (SL = -1)
                sl_dist_pct = abs(entry - sl) / entry * 100.0
                r_mult = net_pct / sl_dist_pct if sl_dist_pct > 0 else 0.0

                trades.append({
                    "symbol": df_symbol["symbol"].iloc[0],
                    "side": side,
                    "entry_time": df_symbol.index[pos["entry_idx"]],
                    "exit_time": df_symbol.index[i],
                    "entry": entry,
                    "sl": sl,
                    "tp": pos["tp"],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "net_pct": net_pct,
                    "r_mult": r_mult,
                })
                pos = None

        # ---------- ورود پوزیشن جدید ----------
        if pos is None and atr[i] > 0 and adx[i] >= min_adx:
            long_fire = up_probs[i] >= up_thr
            short_fire = down_probs[i] >= dn_thr

            if long_fire and short_fire:
                side = ("long" if (up_probs[i] - up_thr) >= (down_probs[i] - dn_thr)
                        else "short")
            elif long_fire:
                side = "long"
            elif short_fire:
                side = "short"
            else:
                side = None

            if side is not None and i + 1 < n:
                pending = (side, atr[i])

    return trades


def split_trades(trades: list, side: str) -> list:
    return [t for t in trades if t["side"] == side]


EXIT_REASONS = ("TP", "SL", "TIME")


def exit_breakdown(trades: list) -> dict:
    """تجمیع دلایل خروج (TP/SL/TIME): تعداد و میانگین R هر دلیل.

    این تفکیک نشان می‌دهد زیان از «انتخاب غلط جهت» می‌آید یا از «هندسه‌ی خروج»
    (مثلاً بسته‌شدن زودهنگام با تایمر max_hold قبل از رسیدن به TP).
    """
    out = {}
    for reason in EXIT_REASONS:
        sub = [t for t in trades if t.get("exit_reason") == reason]
        if not sub:
            continue
        rs = np.array([t["r_mult"] for t in sub])
        out[reason] = {"n": len(sub), "avg_r": float(rs.mean())}
    return out


def stats_of(trades: list) -> dict:
    if not trades:
        return {"n": 0, "win_rate": None, "avg_r": None, "exp_pct": None}
    rs = np.array([t["r_mult"] for t in trades])
    pnls = np.array([t["net_pct"] for t in trades])
    return {
        "n": len(trades),
        "win_rate": float((rs > 0).mean() * 100),
        "avg_r": float(rs.mean()),
        "exp_pct": float(pnls.mean()),
    }


def run_window(df, X, y_up, y_dn, start, train_size, test_size, horizon,
               quantile, min_adx, atr_sl_mult, risk_reward, fee,
               use_funding, lgb, embargo=0):
    # 🔒 embargo: فاصله‌ی بین آخرین ردیف Train و اولین ردیف Test تا برچسب‌های
    # Train (که به horizon کندل آینده نگاه می‌کنند) با دوره‌ی Test هم‌پوشانی نکنند.
    train_idx = slice(start, start + train_size)
    test_start = start + train_size + embargo
    test_idx = slice(test_start, test_start + test_size)

    # 🔒 Train داخلی (۸۵٪) + Validation داخلی (۱۵٪ آخر Train، با embargo)
    # برای early stopping و آستانه — دیگر هیچ چیز از Test دیده نمی‌شود.
    fit_end = start + int(train_size * 0.85)
    val_start = min(fit_end + embargo, start + train_size - 1)
    fit_idx = slice(start, fit_end)
    val_idx = slice(val_start, start + train_size)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    test_df = df.iloc[test_idx].copy()

    X_fit, X_val = X.iloc[fit_idx], X.iloc[val_idx]

    def _train(y_all):
        m = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        m.fit(X_fit, y_all.iloc[fit_idx], eval_X=X_val, eval_y=y_all.iloc[val_idx],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        return m

    up_model = _train(y_up)
    dn_model = _train(y_dn)

    up_val_prob = up_model.predict_proba(X_val)[:, 1]
    dn_val_prob = dn_model.predict_proba(X_val)[:, 1]
    up_test_prob = up_model.predict_proba(X_test)[:, 1]
    dn_test_prob = dn_model.predict_proba(X_test)[:, 1]

    # 🔒 آستانه از احتمال‌های «خارج از نمونه‌ی Validation» (نه Train که overfit است)
    tau_up = float(np.quantile(up_val_prob, quantile))
    tau_dn = float(np.quantile(dn_val_prob, quantile))

    trades = []
    for symbol in test_df["symbol"].unique():
        mask = (test_df["symbol"] == symbol).values
        grp = test_df[mask]
        trades.extend(simulate_plan(
            grp, up_test_prob[mask], dn_test_prob[mask],
            tau_up, tau_dn, min_adx, atr_sl_mult, risk_reward,
            max_hold=horizon, fee=fee, use_funding=use_funding,
        ))

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
        "n": st_all["n"], "avg_r": st_all["avg_r"], "wr": st_all["win_rate"],
        "exits": exit_breakdown(trades),
        "exp_pct": st_all["exp_pct"],
        "n_long": st_long["n"], "avg_r_long": st_long["avg_r"],
        "n_short": st_short["n"], "avg_r_short": st_short["avg_r"],
        "bh": " | ".join(bh_lines),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward پلن معاملاتی دقیق (SL/TP) روی تایم‌فریم کوتاه")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--min-adx", type=float, default=30.0)
    parser.add_argument("--atr-sl-mult", type=float, default=DEFAULT_ATR_SL_MULT)
    parser.add_argument("--risk-reward", type=float, default=DEFAULT_RISK_REWARD)
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT,
                        help="ریسک هر معامله از حساب (درصد) — برای گزارش ٪ حساب")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--embargo", type=int, default=None,
                        help="فاصله‌ی Train→Test به ردیف (پیش‌فرض: horizon × تعداد نماد)")
    parser.add_argument("--no-funding", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="فقط جمع‌بندی + تفکیک سال‌به‌سال (بدون جزئیات هر پنجره)")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="کنترل تصادفی: شافل برچسب‌ها (حذف سیگنال) برای رد مصنوع انتخاب")
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
    if args.shuffle_labels:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(y_up))
        y_up = y_up.iloc[perm].reset_index(drop=True)
        y_dn = y_dn.iloc[perm].reset_index(drop=True)
        print("🔀 کنترل تصادفی فعال: برچسب‌های هدف به‌هم ریختند (سیگنال باید از بین برود)")

    X["symbol"] = X["symbol"].astype("category")

    n = len(X)
    use_funding = not args.no_funding

    print("\n" + "=" * 100)
    print("🚶 WALK-FORWARD — پلن معاملاتی دقیق (SL/TP) — تایم‌فریم کوتاه")
    print("=" * 100)
    print(f"• پیکربندی: {args.interval} | horizon={args.horizon} | "
          f"quantile={args.quantile} | ADX≥{args.min_adx:g}")
    print(f"• قوانین: SL=ATR×{args.atr_sl_mult:g} | TP={args.risk_reward:g}R | "
          f"ریسک {args.risk_pct:g}٪ | fee={args.fee*100:.3f}٪/طرف")
    print(f"• پنجره‌ها: Train {args.train_size} → Test {args.test_size} "
          f"(گام {args.step}) | حداکثر {args.windows} پنجره")
    print()

    results = []
    if args.summary_only:
        def _pprint(*a, **k):  # pragma: no cover
            pass
    else:
        def _pprint(*a, **k):
            print(*a, **k)
    n_sym = max(1, df["symbol"].nunique())
    # ردیف‌ها بین نمادها درهم‌اند؛ embargo = بزرگ‌ترین افق × تعداد نماد
    embargo = args.embargo if args.embargo is not None else args.horizon * n_sym
    print(f"• embargo بین Train و Test: {embargo} ردیف | ورود: open کندل بعد از سیگنال")
    start = 0
    window_num = 0
    while (start + args.train_size + embargo + args.test_size <= n) and (window_num < args.windows):
        window_num += 1
        try:
            r = run_window(df, X, y_up, y_dn, start, args.train_size,
                           args.test_size, args.horizon, args.quantile,
                           args.min_adx, args.atr_sl_mult, args.risk_reward,
                           args.fee, use_funding, lgb, embargo=embargo)
        except Exception as e:
            logger.error(f"پنجره {window_num}: خطا — {e}")
            start += args.step
            continue

        results.append(r)
        r_str = f"{r['avg_r']:+.3f}R" if r["avg_r"] is not None else "بدون معامله"
        rl = f"{r['avg_r_long']:+.3f}R" if r["avg_r_long"] is not None else "—"
        rs = f"{r['avg_r_short']:+.3f}R" if r["avg_r_short"] is not None else "—"
        if not args.summary_only:
            _pprint(f"پنجره {window_num}: {r['window_start'].date()} → {r['window_end'].date()}")
        elif window_num % 20 == 0:
            print(f"  … پنجره {window_num} اجرا شد …")
        _pprint(f"   دوطرفه: {r['n']:>3} معامله | میانگین {r_str} | WinRate "
              f"{r['wr']:.0f}٪" if r["wr"] is not None
              else f"   دوطرفه: {r['n']} معامله | بدون معامله")
        _pprint(f"   لانگ: {r['n_long']:>3} ({rl}) | شورت: {r['n_short']:>3} ({rs})")
        _pprint(f"   آستانه‌ها: up={r['tau_up']:.4f} | dn={r['tau_dn']:.4f} | Buy&Hold: {r['bh']}")
        _pprint("-" * 100)
        start += args.step

    if not results:
        print("❌ هیچ پنجره‌ای با موفقیت اجرا نشد.")
        return

    valid = [r for r in results if r["avg_r"] is not None]
    print("\n" + "=" * 100)
    print("📊 جمع‌بندی Walk-Forward — پلن معاملاتی دقیق")
    print("=" * 100)

    if valid:
        avg_rs = np.array([r["avg_r"] for r in valid])
        pos = int((avg_rs > 0).sum())
        mean_r = avg_rs.mean()
        total_trades = sum(r["n"] for r in valid)

        print(f"• دوطرفه ({len(valid)} پنجره‌ی معتبر):")
        print(f"    پنجره‌های مثبت: {pos}/{len(valid)} ({pos/len(valid)*100:.0f}٪)")
        print(f"    میانگین Expectancy: {mean_r:+.4f}R | کل معاملات: {total_trades}")
        print(f"    معادل ٪ حساب (ریسک {args.risk_pct:g}٪): {mean_r*args.risk_pct:+.4f}٪ حساب/معامله")

        longs = [r for r in valid if r["avg_r_long"] is not None]
        shorts = [r for r in valid if r["avg_r_short"] is not None]
        if longs:
            le = np.mean([r["avg_r_long"] for r in longs])
            print(f"• فقط لانگ: میانگین {le:+.4f}R ({sum(r['n_long'] for r in longs)} معامله)")
        if shorts:
            se = np.mean([r["avg_r_short"] for r in shorts])
            print(f"• فقط شورت: میانگین {se:+.4f}R ({sum(r['n_short'] for r in shorts)} معامله)")

        # --- تست دوام: تفکیک پنجره‌ها بر اساس سال (پایداری edge در بازه‌های متفاوت) ---
        by_year = {}
        for r in valid:
            by_year.setdefault(r["window_start"].year, []).append(r)
        print("-" * 100)
        print("تفکیک سال‌به‌سال (پایداری edge در رژیم‌های متفاوت بازار):")
        for y in sorted(by_year):
            yr = by_year[y]
            a = np.array([r["avg_r"] for r in yr])
            posy = int((a > 0).sum())
            ntr = sum(r["n"] for r in yr)
            print(f"  {y}: {len(yr):>3} پنجره | میانگین {a.mean():+.4f}R | "
                  f"مثبت {posy}/{len(yr)} | {ntr} معامله")

        # (جزئیات دلایل خروج در نسخه‌های بعدی به گزارش اضافه می‌شود)
        # (متغیر بی‌استفاده حذف شد)
        # (حلقه‌ی بی‌اثر حذف شد)
            # (بلوک مرده حذف شد)

        print("-" * 100)
        # --- ترکیب دلایل خروج: TP / SL / TIME (چرا معامله‌ها بسته می‌شوند؟) ---
        exit_tot = {}
        for r in valid:
            for reason, agg in r.get("exits", {}).items():
                e = exit_tot.setdefault(reason, {"n": 0, "r_sum": 0.0})
                e["n"] += agg["n"]
                e["r_sum"] += agg["avg_r"] * agg["n"]
        if exit_tot:
            grand = sum(e["n"] for e in exit_tot.values())
            print("-" * 100)
            print("دلایل خروج (TP = هدف، SL = حد ضرر، TIME = پایان تایمر max_hold):")
            for reason in EXIT_REASONS:
                e = exit_tot.get(reason)
                if not e or not e["n"]:
                    continue
                share = e["n"] / grand * 100 if grand else 0.0
                print(f"  {reason:<5}: {e['n']:>5} معامله ({share:4.1f}٪ از کل) | "
                      f"میانگین {e['r_sum'] / e['n']:+.3f}R")

        if pos / len(valid) >= 0.6 and mean_r > 0:
            print("🟢 پلن دقیق در اکثر پنجره‌ها مثبت است — شواهد edge برای ورود/خروج دقیق وجود دارد.")
        elif pos / len(valid) >= 0.4:
            print("🟡 نتیجه ترکیبی — پلن هنوز به رژیم بازار وابسته است؛ نیاز به داده‌ی بیشتر.")
        else:
            print("🔴 پلن دقیق در اکثر پنجره‌ها مثبت نبوده — edge پایدار هنوز تأیید نمی‌شود.")
    else:
        print("🔴 هیچ معامله‌ای در هیچ پنجره‌ای انجام نشد (آستانه‌ها/فیلترها خیلی سخت‌گیرانه).")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()