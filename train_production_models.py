"""
train_production_models.py
--------------------------
آموزش و ذخیره‌ی «مدل‌های تولیدی» دوجهته (UP + DOWN) برای سیستم آلارم زنده.

قبلاً همه‌ی مدل‌ها داخل حلقه‌های Walk-Forward آموزش دیده و دور ریخته می‌شدند.
این اسکریپت برای اولین بار مدل‌های قابل‌استفاده‌ی زنده می‌سازد:

  1. بارگذاری دیتاست ویژگی‌های 15m (باید اول fetch_futures_data و build_dataset اجرا شود)
  2. آموزش دو مدل جدا: UP (احتمال صعود) و DOWN (احتمال نزول)
  3. آستانه‌های چندک از احتمال‌های Train همان دوره (بدون look-ahead)
  4. sanity check روی یک بلوک جدا (Holdout) با قوانین دقیق trade_plan
  5. ذخیره‌ی مدل‌ها + متادیتا (JSON)

اجرا (از داخل پوشه‌ی پروژه):
    python train_production_models.py --interval 15m --horizon 4 --quantile 0.75

خروجی:
    models/prod_{interval}_up.txt
    models/prod_{interval}_dn.txt
    models/prod_{interval}_meta.json
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest_model import (
    load_combined_dataset,
    DEFAULT_SYMBOLS,
    DEFAULT_INTERVAL,
    DEFAULT_FEE,
)
from features import get_feature_columns

# قوانین پلن (هماهنگ با strategies/trade_plan.py و walk_forward_trade_plan.py)
from walk_forward_trade_plan import simulate_plan, stats_of

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TrainProduction")


def _predict_proba(booster, X_df: pd.DataFrame) -> np.ndarray:
    """پیش‌بینی احتمال با مدل ذخیره‌شده — ترتیب ستون‌ها با feature_name تنظیم می‌شود."""
    fnames = list(booster.feature_name())
    Xr = X_df[fnames]
    return booster.predict(Xr)  # برای objective باینری، خروجی همان احتمال است


def _dataset_age_hours(df: pd.DataFrame) -> float:
    last_ts = df.index[-1]
    now = datetime.now(timezone.utc)
    if last_ts.tzinfo is None:
        now = now.replace(tzinfo=None)
    return (now - last_ts).total_seconds() / 3600.0


def _load_feature_flags(symbols, interval, cache_dir="data_cache"):
    """
    خواندن فلگ‌های ساخت ویژگی از متادیتای دیتاست (sidecar) که build_dataset.py
    می‌نویسد — تا بدانیم آموزش با کدام بلوک‌های ویژگی انجام شده و اجرای زنده
    دقیقاً همان‌ها را بسازد (ضد train/serve skew).
    """
    flags = None
    for s in symbols:
        p = os.path.join(cache_dir, f"{s}_{interval}_features_meta.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            logger.warning(f"خواندن متادیتای دیتاست ناموفق بود ({p}): {e}")
            continue
        fl = d.get("feature_flags")
        if not fl:
            continue
        if flags is None:
            flags = fl
        elif flags != fl:
            logger.warning(f"فلگ‌های دیتاست {s} با بقیه یکسان نیست: {fl} != {flags}")
    return flags


def main():
    parser = argparse.ArgumentParser(description="آموزش مدل‌های تولیدی دوجهته")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--min-adx", type=float, default=30.0)
    parser.add_argument("--atr-sl-mult", type=float, default=1.5)
    parser.add_argument("--risk-reward", type=float, default=2.0)
    parser.add_argument("--holdout", type=int, default=500,
                        help="ردیف آخر به عنوان بلوک جدا برای sanity check")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE)
    parser.add_argument("--no-funding", action="store_true")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("\n❌ lightgbm نصب نیست: pip install lightgbm scikit-learn\n")
        return

    symbols = [s.upper() for s in args.symbols]
    df = load_combined_dataset(symbols, args.interval)

    up_col = f"target_up_{args.horizon}"
    dn_col = f"target_down_{args.horizon}"
    if up_col not in df.columns or dn_col not in df.columns:
        print(f"❌ ستون‌های {up_col}/{dn_col} در دیتاست نیستند. build_dataset را اجرا کنید.")
        return

    # هشدار صادقانه اگر داده قدیمی باشد
    age_h = _dataset_age_hours(df)
    if age_h > 24:
        logger.warning(
            f"دیتاست حدود {age_h:.0f} ساعت قدمت دارد ({df.index[-1]}). "
            "برای داده‌ی تازه: fetch_futures_data.py سپس build_dataset.py را دوباره اجرا کنید."
        )
    else:
        logger.info(f"قدمت دیتاست: {age_h:.1f} ساعت — تازه است.")

    feature_cols = get_feature_columns(df)
    if "symbol" not in feature_cols:
        feature_cols = feature_cols + ["symbol"]

    X = df[feature_cols].copy()
    y_up = df[up_col].astype(int)
    y_dn = df[dn_col].astype(int)
    X["symbol"] = X["symbol"].astype("category")

    n = len(X)
    split_idx = n - args.holdout
    if split_idx <= 0:
        print("❌ اندازه‌ی بلوک holdout از کل داده بزرگ‌تر است.")
        return

    # 🔒 embargo: ردیف‌های آخر Train که برچسبشان به دوره‌ی Holdout نگاه می‌کند حذف می‌شوند
    #    (ردیف‌ها بین نمادها درهم‌اند → horizon × تعداد نماد)
    n_sym = max(1, df["symbol"].nunique())
    embargo = args.horizon * n_sym
    train_end = split_idx - embargo
    if train_end <= 0:
        print("❌ داده برای embargo + holdout کافی نیست.")
        return
    # 🔒 Train داخلی (۸۵٪) / Validation داخلی (۱۵٪ آخر) فقط برای early stopping و آستانه
    fit_end = int(train_end * 0.85)
    val_start = fit_end + embargo

    X_train, X_hold = X.iloc[:fit_end], X.iloc[split_idx:]
    X_val = X.iloc[val_start:train_end]
    y_up_train, y_up_val = y_up.iloc[:fit_end], y_up.iloc[val_start:train_end]
    y_dn_train, y_dn_val = y_dn.iloc[:fit_end], y_dn.iloc[val_start:train_end]

    use_funding = not args.no_funding
    print("\n" + "=" * 100)
    print("🧠 آموزش مدل‌های تولیدی دوجهته")
    print("=" * 100)
    print(f"• نمادها: {', '.join(symbols)} | تایم‌فریم: {args.interval}")
    print(f"• Train: {split_idx} ردیف | Holdout (sanity): {args.holdout} ردیف")
    print(f"• ویژگی‌ها: {len(feature_cols)} | horizon={args.horizon} | quantile={args.quantile}")

    # ---------- آموزش UP و DOWN ----------
    def _train(y_tr, y_te):
        m = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        m.fit(X_train, y_tr, eval_X=X_val, eval_y=y_te,
              callbacks=[lgb.early_stopping(30, verbose=False)])
        return m

    print("\nآموزش مدل UP ...")
    up_model = _train(y_up_train, y_up_val)
    print("آموزش مدل DOWN ...")
    dn_model = _train(y_dn_train, y_dn_val)

    # 🔒 آستانه از احتمال‌های Validation (خارج از نمونه) — نه Train که overfit است
    up_val_prob = up_model.predict_proba(X_val)[:, 1]
    dn_val_prob = dn_model.predict_proba(X_val)[:, 1]
    tau_up = float(np.quantile(up_val_prob, args.quantile))
    tau_dn = float(np.quantile(dn_val_prob, args.quantile))
    print(f"\nآستانه‌های سیگنال (چندک {args.quantile:g} Validation):")
    print(f"  tau_up  = {tau_up:.4f}  (فقط اگر up_prob  >= این مقدار → LONG)")
    print(f"  tau_dn  = {tau_dn:.4f}  (فقط اگر dn_prob  >= این مقدار → SHORT)")

    # ---------- ذخیره‌سازی ----------
    os.makedirs(args.out, exist_ok=True)
    up_path = os.path.join(args.out, f"prod_{args.interval}_up.txt")
    dn_path = os.path.join(args.out, f"prod_{args.interval}_dn.txt")
    up_model.booster_.save_model(up_path)
    dn_model.booster_.save_model(dn_path)
    logger.info(f"ذخیره شد: {up_path}")
    logger.info(f"ذخیره شد: {dn_path}")

    # ---------- sanity check روی بلوک جدا با قوانین دقیق پلن ----------
    up_booster = lgb.Booster(model_file=up_path)
    dn_booster = lgb.Booster(model_file=dn_path)
    up_hold_prob = _predict_proba(up_booster, X_hold)
    dn_hold_prob = _predict_proba(dn_booster, X_hold)
    hold_df = df.iloc[split_idx:].copy()

    trades = []
    for symbol in hold_df["symbol"].unique():
        mask = (hold_df["symbol"] == symbol).values
        grp = hold_df[mask]
        trades.extend(simulate_plan(
            grp, up_hold_prob[mask], dn_hold_prob[mask],
            tau_up, tau_dn, args.min_adx, args.atr_sl_mult,
            args.risk_reward, max_hold=args.horizon,
            fee=args.fee, use_funding=use_funding,
        ))
    st = stats_of(trades)

    print("\n" + "-" * 100)
    print(f"🔎 Sanity Check روی بلوک جدا ({hold_df.index[0]} → {hold_df.index[-1]}):")
    if st["n"] == 0:
        print("   هیچ سیگنالی در بلوک holdout فعال نشد (آستانه‌ها سخت‌گیرانه‌اند).")
    else:
        print(f"   معاملات: {st['n']} | WinRate: {st['win_rate']:.0f}٪ | "
              f"میانگین: {st['avg_r']:+.3f}R | سود خالص: {st['exp_pct']:+.3f}٪")
        print("   ⚠️ این فقط sanity است؛ اعتبارسنجی کامل = walk_forward_trade_plan.py")
    print("-" * 100)

    # ---------- متادیتا ----------
    feature_flags = _load_feature_flags(symbols, args.interval)
    if feature_flags is None:
        logger.warning("متادیتای دیتاست (sidecar) پیدا نشد — فلگ‌های ویژگی در متادیتای "
                       "مدل ثبت نمی‌شوند و اجرای زنده به پیش‌فرض‌ها تکیه می‌کند.")
    meta = {
        "interval": args.interval,
        "horizon": args.horizon,
        "quantile": args.quantile,
        "min_adx": args.min_adx,
        "atr_sl_mult": args.atr_sl_mult,
        "risk_reward": args.risk_reward,
        "fee": args.fee,
        "use_funding": use_funding,
        "symbols": symbols,
        "n_train": int(fit_end),
        "n_val": int(len(X_val)),
        "embargo_rows": int(embargo),
        "leak_fixed": True,  # ICS_26 حذف + تاخیر تأیید swing + ورود open بعدی
        "n_holdout": int(args.holdout),
        "feature_cols": list(feature_cols),
        "feature_flags": feature_flags,
        "tau_up": tau_up,
        "tau_dn": tau_dn,
        "dataset_start": str(df.index[0]),
        "dataset_end": str(df.index[-1]),
        "holdout_start": str(hold_df.index[0]),
        "holdout_end": str(hold_df.index[-1]),
        "holdout_summary": {
            "n_trades": int(st["n"]),
            "win_rate": st["win_rate"],
            "avg_r": st["avg_r"],
            "exp_pct": st["exp_pct"],
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {
            "up": os.path.basename(up_path),
            "down": os.path.basename(dn_path),
        },
    }
    meta_path = os.path.join(args.out, f"prod_{args.interval}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"متادیتا ذخیره شد: {meta_path}")

    print("\n✅ مدل‌های تولیدی آماده‌اند. قدم بعدی:")
    print("   اعتبارسنجی کامل:")
    print(f"   python walk_forward_trade_plan.py --interval {args.interval} "
          f"--horizon {args.horizon} --quantile {args.quantile} --min-adx {args.min_adx:g}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()