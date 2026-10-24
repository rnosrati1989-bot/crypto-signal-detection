"""
live_signal.py
--------------
موتور «سیگنال زنده» — وصل‌کردن مدل‌های LGBM تولیدی به مسیر زنده.

مشکل قبلی: مدل‌ها (models/prod_15m_up.txt و ...) آموزش دیده و ذخیره می‌شدند
ولی هیچ‌وقت در اجرای زنده بارگذاری نمی‌شدند؛ main.py فقط اندیکاتور + دیجست +
نظر دو LLM را نشان می‌داد. این ماژول حلقه‌ی گم‌شده است:

  1. دریافت داده‌ی فیوچرز زنده (شامل taker_buy_volume / funding_rate / OI)
     — همان ستون‌هایی که دیتاست آموزشی داشت (اسپات این‌ها را ندارد).
  2. ساخت ویژگی‌ها با features.build_features — دقیقاً همان ۶۸ ویژگی آموزش.
  3. بارگذاری مدل‌های prod_{interval}_up/dn + متادیتا (آستانه‌های tau).
  4. پیش‌بینی احتمال صعود/نزول روی آخرین کندلِ بسته‌شده.
  5. اعمال قوانین ورودِ پلن معاملاتی (همان walk_forward_trade_plan):
        atr>0 و ADX>=min_adx
        up_prob>=tau_up  → LONG   |   dn_prob>=tau_dn → SHORT
     (اگر هر دو: طرفی که حاشیه‌ی بیشتری بالای آستانه دارد)
  6. محاسبه‌ی سطوح: ورود=close آخرین کندل، SL=ورود∓ATR×1.5، TP=±2R،
     حداکثر نگهداشت=horizon کندل.

⚠️ صداقت: این قوانین در دو اجرای Walk-Forward (~۶۸۵ معامله بعد از کارمزد
و فاندینگ) edge اثبات‌شده‌ای نشان ندادند (میانگین ~0.12- تا 0.18- R).
خروجی این ماژول یک «ورودی کمکی برای تصمیم انسان» است، نه توصیه‌ی معامله
با پول واقعی. قبل از هر استفاده‌ی واقعی: paper-trade و ثبت سیگنال‌ها.

اجرا (از داخل پوشه‌ی پروژه):
    venv\\Scripts\\python.exe live_signal.py --symbol BTCUSDT
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from data.binance_futures_client import BinanceFuturesDataClient
from features import (
    build_features, attach_btc_context, build_btc_context,
    liquidations_to_candles, BTC_CONTEXT_SYMBOL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LiveSignal")

MODELS_DIR = "models"

# فید عمومی لیکوییدیشن بایننس (/fapi/v1/allForceOrders) بازنشسته شده و
# HTTP 404 می‌دهد. بعد از اولین شکست، برای همین اجرا غیرفعال می‌شود تا
# در هر بررسی برای هر نماد، درخواست و لاگ تکراری تولید نشود.
_LIQ_DISABLED = False


# ----------------------------------------------------------------------
# بارگذاری
# ----------------------------------------------------------------------
def load_models(interval: str = "15m", models_dir: str = MODELS_DIR):
    """بارگذاری متادیتا + دو مدل Booster. اگر وجود نداشتند خطای واضح می‌دهد."""
    meta_path = os.path.join(models_dir, f"prod_{interval}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"مدل‌های {interval} پیدا نشدند ({meta_path}). "
            "اول اجرا کنید: fetch_futures_data.py ← build_dataset.py ← train_production_models.py"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("lightgbm نصب نیست: pip install lightgbm scikit-learn")

    up = lgb.Booster(model_file=os.path.join(models_dir, f"prod_{interval}_up.txt"))
    dn = lgb.Booster(model_file=os.path.join(models_dir, f"prod_{interval}_dn.txt"))

    # 🔒 مدل‌های قدیمی با ویژگی نشتی (قیمت آینده) آموزش دیده‌اند و در حالت زنده
    # روی کندل‌های اخیر NaN می‌گیرند → سیگنال بی‌اعتبار. باید دوباره آموزش ببینند.
    bad = [f for f in list(up.feature_name()) + list(dn.feature_name())
           if str(f).startswith("ICS_")]
    if bad:
        raise RuntimeError(
            "این مدل با ویژگی نشتی ICS_26 (قیمت آینده) آموزش دیده و نامعتبر است. "
            "دوباره اجرا کنید: build_dataset.py → train_production_models.py")
    return meta, up, dn


def _predict(booster, row_df: pd.DataFrame) -> float:
    """پیش‌بینی احتمال با مدل ذخیره‌شده — ترتیب ستون‌ها با feature_name تنظیم می‌شود."""
    fnames = list(booster.feature_name())
    Xr = row_df[fnames]
    return float(booster.predict(Xr)[0])


# ----------------------------------------------------------------------
# مومنتوم / ریز
# ----------------------------------------------------------------------
def detect_momentum(raw, atr=None, window=4, atr_mult=1.0, body_min=0.55):
    """
    هشدار روند ریز/ریخت روی کندل‌های بسته‌شده — مکمل مدل LGBM (شفاف، بدون ML).
    معیار: حرکت close در `window` کندل آخر ≥ atr_mult × ATR و بسته‌شدن کندل آخر
    در جهت حرکت (بدنه در سمت هدف). خروجی dict یا None.
    """
    if raw is None or len(raw) < window + 1:
        return None
    closes = raw["close"].to_numpy(dtype=float)
    last = raw.iloc[-1]
    close_last = float(last["close"])
    close_prev = float(closes[-1 - window])
    move = close_last - close_prev
    if atr is None or atr <= 0.0:
        highs = raw["high"].to_numpy(dtype=float)
        lows = raw["low"].to_numpy(dtype=float)
        atr = float(np.mean(highs[-window:] - lows[-window:])) or 1.0
    mom_r = move / atr
    if abs(mom_r) < atr_mult:
        return None
    side = "LONG" if mom_r > 0.0 else "SHORT"
    rng = float(last["high"]) - float(last["low"])
    body = (close_last - float(last["low"])) / rng if rng > 0 else 0.5
    if (side == "LONG" and body < body_min) or (side == "SHORT" and body > 1.0 - body_min):
        return None
    return {
        "fired": True,
        "side": side,
        "window": window,
        "mom_r": float(mom_r),
        "move_pct": float(move / close_prev * 100.0),
        "close": close_last,
        "entry": close_last,
        "atr": float(atr),
        "bar_time": str(raw.index[-1]),
    }


# ----------------------------------------------------------------------
# داده و ویژگی
# ----------------------------------------------------------------------
def _interval_delta(interval: str) -> pd.Timedelta:
    return pd.to_timedelta(interval)


# ----------------------------------------------------------------------
# لیکوییدیشن زنده + context بیت‌کوین
# ----------------------------------------------------------------------
def load_local_liquidations(symbol: str, hours: int = 24,
                            liq_dir: str = "data_cache/liquidations") -> pd.DataFrame:
    """
    خواندن لیکوییدیشن‌های اخیر از **مخزن محلی** (خروجی liquidations_collector.py).

    همان منبعی که در آموزش استفاده می‌شود → بدون train/serve skew.
    ستون‌ها: timestamp, venue, side, price, qty, notional
    side: SELL = لیکویید شدن لانگ | BUY = لیکویید شدن شورت
    """
    path = os.path.join(liq_dir, f"{symbol.upper()}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as e:
        logger.debug(f"خواندن مخزن لیکوییدیشن ناموفق بود: {e}")
        return pd.DataFrame()
    if df.empty or not {"timestamp", "side", "notional"}.issubset(df.columns):
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df = df.dropna(subset=["timestamp", "notional"])
    cutoff = (pd.Timestamp.now(tz="UTC").tz_localize(None)
              - pd.Timedelta(hours=int(hours)))
    return df[df["timestamp"] >= cutoff]


def live_liquidation_pressure(symbol: str, interval: str = "15m",
                              limit: int = 1000, hours: int = 24):
    """
    فشار لیکوییدیشن اخیر از **مخزن محلی** (خروجی liquidations_collector.py) و در
    صورت خالی‌بودن آن، تلاش برای واکشی زنده (اندپوینت عمومی بایننس بازنشسته است).

    ⚠️ ۲۰۲۶-۱۰: اندپوینت عمومی لیکوییدیشن بایننس بازنشسته شده است (HTTP 404)؛
    پس منبع اصلی، مخزن محلی است (لیکوییدیشن تاریخچه‌ی رایگان ندارد).
    (یادداشت طراحی اولیه: تاریخچه‌ی این اندپوینت کوتاه بود؛ بنابراین برای
    *آموزش* بلندمدت مناسب نیست و این تابع برای «اطلاع لحظه‌ای» است.
    خروجی: (dict اطلاعات یا None، DataFrame سفارش‌ها)
    """
    # ۱) منبع اصلی: مخزن محلیِ collector (همان داده‌ی آموزش → بدون skew)
    orders = load_local_liquidations(symbol, hours=int(hours))
    # ۲) اگر مخزن محلی خالی بود، واکشی زنده تلاش می‌شود (بایننس ۴۰۴ می‌دهد → خالی)
    client = BinanceFuturesDataClient()
    global _LIQ_DISABLED
    if _LIQ_DISABLED and (orders is None or orders.empty):
        # فید عمومی لیکوییدیشن بازنشسته شده (۴۰۴) — برای این اجرا خاموش می‌ماند
        _LIQ_DISABLED = True
        logger.debug("فید عمومی لیکوییدیشن غیرفعال است (۴۰۴) — رد شد.")
        return None, pd.DataFrame()

    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms = end_ms - int(hours) * 3600 * 1000
    orders = orders if (orders is not None and not orders.empty) else client.get_liquidation_orders(symbol, limit=limit,
                                           start_time_ms=start_ms,
                                           end_time_ms=end_ms)
    if orders is None or orders.empty:
        logger.info("لیکوییدیشنی در دسترس نیست — نه در مخزن محلی و نه از API (۴۰۴).")
        _LIQ_DISABLED = True
        return None, pd.DataFrame()

    long_amt = float(orders.loc[orders["side"] == "SELL", "notional"].sum())
    short_amt = float(orders.loc[orders["side"] == "BUY", "notional"].sum())
    total = long_amt + short_amt
    info = {
        "window_hours": int(hours),
        "n_orders": int(len(orders)),
        "long_notional": long_amt,    # لیکویید شدن پوزیشن‌های لانگ
        "short_notional": short_amt,  # لیکویید شدن پوزیشن‌های شورت
        "imbalance": ((short_amt - long_amt) / total) if total > 0 else 0.0,
        "max_notional": float(orders["notional"].max()),
    }
    return info, orders


def _live_btc_context(symbol: str, interval: str, bars: int,
                      own_raw: pd.DataFrame = None) -> pd.DataFrame:
    """context بیت‌کوین برای نمادها — با همان تایم‌فریم و تعداد کندل."""
    if symbol.upper() == BTC_CONTEXT_SYMBOL:
        # برای خودِ بیت‌کوین، context از داده‌ی خودش ساخته می‌شود (مثل زمان آموزش)
        return build_btc_context(own_raw) if own_raw is not None else pd.DataFrame()
    try:
        btc_raw = BinanceFuturesDataClient().get_futures_dataset(
            BTC_CONTEXT_SYMBOL, interval, limit=bars)
    except Exception as e:
        logger.warning(f"دریافت داده‌ی بیت‌کوین برای context ناموفق بود: {e}")
        return pd.DataFrame()
    return build_btc_context(btc_raw)


def build_live_feature_frame(symbol: str, interval: str = "15m",
                             bars: int = 500,
                             feature_flags: dict = None) -> pd.DataFrame:
    """
    دریافت داده‌ی فیوچرز زنده + ساخت ویژگی‌ها (بدون target).
    کندلِ در حال شکل‌گیری حذف می‌شود تا سیگنال روی «آخرین کندل بسته‌شده» باشد
    (همان قرارداد بک‌تست: ورود با close کندل سیگنال — بدون repaint).

    خروجی: DataFrame ویژگی‌ها؛ ردیف آخر = آخرین کندل بسته‌شده.
    """
    client = BinanceFuturesDataClient()
    raw = client.get_futures_dataset(symbol, interval, limit=bars)
    if raw is None or raw.empty:
        raise RuntimeError(f"داده‌ی فیوچرز زنده برای {symbol}/{interval} دریافت نشد.")

    # حذف کندل ناتمام (هنوز بسته نشده — سیگنال‌های ناقص می‌دهد)
    bar_end = raw.index + _interval_delta(interval)
    now_utc = pd.Timestamp.now(tz="UTC")
    if raw.index.tz is None:
        now_utc = now_utc.tz_localize(None)
    completed = raw[bar_end <= now_utc]
    if len(completed) < 200:
        raise RuntimeError(
            f"تاریخچه‌ی تکمیل‌شده ناکافی ({len(completed)} کندل) برای ساخت ویژگی‌های پایدار."
        )
    raw = completed

    # فلگ‌های ساخت ویژگی — از متادیتای مدل می‌آید تا با آموزش یکسان بماند
    flags = dict(feature_flags or {})

    # لیکوییدیشن اخیر: هم برای اطلاع، هم (اگر در آموزش فعال بوده) به‌عنوان ویژگی
    liq_info, liq_orders = None, pd.DataFrame()
    try:
        liq_info, liq_orders = live_liquidation_pressure(symbol, interval)
    except Exception as liq_e:
        logger.debug(f"لیکوییدیشن زنده در دسترس نبود: {liq_e}")
    if (bool(flags.get("include_liquidations", False))
            and liq_orders is not None and not liq_orders.empty):
        raw = raw.join(liquidations_to_candles(liq_orders, raw.index, interval))

    feat = build_features(raw,
                           include_smart_money=bool(flags.get("include_smart_money", True)),
                           include_chart_patterns=bool(flags.get("include_chart_patterns", True)),
                           include_scores=bool(flags.get("include_scores", True)),
                           include_absolute_levels=bool(flags.get("include_absolute_levels", True)),
                           include_liquidations=bool(flags.get("include_liquidations", False)),
                           include_elliott=bool(flags.get("include_elliott", False)))
    if feat.empty:
        raise RuntimeError("ساخت ویژگی‌ها با داده‌ی زنده شکست خورد.")
    # context بیت‌کوین (ویژگی‌های متقاطع) — دقیقاً مثل زمان آموزش
    if bool(flags.get("include_btc_context", True)):
        feat = attach_btc_context(feat, _live_btc_context(symbol, interval, bars, raw))

    feat.attrs["raw"] = raw  # برای detect_momentum در build_live_signal
    feat.attrs["liquidation"] = liq_info
    feat.attrs["feature_flags"] = flags
    return feat


# ----------------------------------------------------------------------
# سیگنال
# ----------------------------------------------------------------------
def build_live_signal(symbol: str, interval: str = "15m",
                      bars: int = 500, models_dir: str = MODELS_DIR,
                      min_adx: float = None, tau_up: float = None, tau_dn: float = None,
                      tau_scale: float = 1.0, mom_atr_mult: float = 1.0,
                      mom_window: int = None) -> dict:
    """
    سیگنال زنده روی آخرین کندل بسته‌شده با قوانین دقیق پلن معاملاتی.
    """
    symbol = symbol.upper()
    meta, up, dn = load_models(interval, models_dir)

    trained_symbols = [s.upper() for s in meta.get("symbols", [])]
    if symbol not in trained_symbols:
        raise RuntimeError(
            f"مدل فقط روی نمادهای {trained_symbols} آموزش دیده؛ {symbol} پشتیبانی نمی‌شود."
        )

    flags = meta.get("feature_flags") or {}
    if not flags:
        logger.warning("متادیتای مدل فلگ‌های ویژگی ندارد (مدل قدیمی) — اجرای زنده با "
                       "پیش‌فرض‌ها ساخته می‌شود؛ برای رفع قطعی مدل را دوباره آموزش دهید.")
    feat = build_live_feature_frame(symbol, interval, bars=bars, feature_flags=flags)
    raw = feat.attrs.get("raw") if hasattr(feat, "attrs") else None
    liq_info = feat.attrs.get("liquidation") if hasattr(feat, "attrs") else None

    # 🔒 نگهبان train/serve skew: هر ویژگی‌ای که مدل انتظار دارد باید در داده‌ی
    # زنده هم ساخته شده باشد؛ وگرنه سیگنال بی‌اعتبار است (همان باگ HOLD همیشگی).
    expected = list(dict.fromkeys(list(up.feature_name()) + list(dn.feature_name())))
    missing_feats = [f for f in expected if f != "symbol" and f not in feat.columns]
    extra_feats = [c for c in feat.columns if c not in expected and c != "symbol"]
    if missing_feats:
        raise RuntimeError(
            "ناسازگاری ویژگی آموزش/زنده — این ستون‌ها در داده‌ی زنده نیستند: "
            f"{missing_feats}. دیتاست و مدل را با همان فلگ‌ها بازسازی/آموزش دهید."
        )
    if extra_feats:
        logger.info(f"{len(extra_feats)} ستون اضافه در داده‌ی زنده (بی‌اثر روی مدل): "
                    f"{extra_feats[:8]}{' ...' if len(extra_feats) > 8 else ''}")

    # ردیف آخر (آخرین کندل بسته‌شده) → ورودی مدل
    row = feat.iloc[[-1]].copy()
    # کدگذاری symbol باید با ترتیب دسته‌های آموزش (sorted) یکی باشد
    categories = sorted(trained_symbols)
    row["symbol"] = pd.Categorical([symbol], categories=categories)

    up_prob = _predict(up, row)
    dn_prob = _predict(dn, row)

    tau_up = float(tau_up) if tau_up is not None else float(meta["tau_up"])
    tau_dn = float(tau_dn) if tau_dn is not None else float(meta["tau_dn"])
    tau_up = tau_up * float(tau_scale)
    tau_dn = tau_dn * float(tau_scale)
    min_adx = float(min_adx) if min_adx is not None else float(meta["min_adx"])
    atr_sl_mult = float(meta["atr_sl_mult"])
    risk_reward = float(meta["risk_reward"])
    horizon = int(meta["horizon"])

    last = feat.iloc[-1]
    close = float(last["close"])
    atr = float(last["ATR"]) if pd.notna(last.get("ATR")) else 0.0
    adx = float(last["ADX_14"]) if pd.notna(last.get("ADX_14")) else 0.0

    long_fire = up_prob >= tau_up
    short_fire = dn_prob >= tau_dn
    adx_ok = adx >= min_adx
    atr_ok = atr > 0.0

    # قوانین ورود — دقیقاً مطابق simulate_plan در walk_forward_trade_plan.py
    hold_reason = None
    if not (adx_ok and atr_ok):
        side = "HOLD"
        hold_reason = "ADX" if not adx_ok else "ATR"
    elif long_fire and short_fire:
        side = "LONG" if (up_prob - tau_up) >= (dn_prob - tau_dn) else "SHORT"
    elif long_fire:
        side = "LONG"
    elif short_fire:
        side = "SHORT"
    else:
        side = "HOLD"

    # حاشیه (قدرت سیگنال): فاصله از آستانه نرمال‌شده 0..1
    if side == "LONG":
        margin = (up_prob - tau_up) / (1.0 - tau_up)
    elif side == "SHORT":
        margin = (dn_prob - tau_dn) / (1.0 - tau_dn)
    else:
        margin = max((up_prob - tau_up) / (1.0 - tau_up),
                     (dn_prob - tau_dn) / (1.0 - tau_dn), 0.0)

    # سطوح پلن
    entry = close
    sd = atr * atr_sl_mult
    sl = entry - sd if side == "LONG" else entry + sd
    tps = [entry + sd * m if side == "LONG" else entry - sd * m for m in (1.0, 2.0, 3.0)]  # ۳ پله‌ی TP: 1R / 2R / 3R

    # هشدار روند ریز/ریخت — مکمل مدل (روی کندل‌های بسته‌شده، شفاف)
    momentum = detect_momentum(raw, atr=atr,
                               window=int(mom_window) if mom_window else int(horizon),
                               atr_mult=float(mom_atr_mult))

    return {
        "ok": True,
        "symbol": symbol,
        "interval": interval,
        "bar_time": str(feat.index[-1]),
        "close": close,
        "atr": atr,
        "adx": adx,
        "up_prob": up_prob,
        "dn_prob": dn_prob,
        "tau_up": tau_up,
        "tau_dn": tau_dn,
        "min_adx": min_adx,
        "side": side,
        "margin": float(np.clip(margin, 0.0, 1.0)),
        "hold_reason": hold_reason,
        "entry": entry,
        "sl": sl if side != "HOLD" else None,
        "tps": ([round(float(t), 2) for t in tps] if side != "HOLD" else None),
        "atr_sl_mult": atr_sl_mult,
        "risk_reward": risk_reward,
        "max_hold": horizon,
        "model_meta": meta,
        "momentum": momentum,
        "liquidation": liq_info,
    }


# ----------------------------------------------------------------------
# نمایش
# ----------------------------------------------------------------------
def format_signal(sig: dict) -> str:
    lines = []
    lines.append("🔔 سیگنال زنده‌ی مدل (LGBM) — قوانین پلن معاملاتی:")
    lines.append(f"   • کندل سیگنال: {sig['bar_time']} ({sig['symbol']}/{sig['interval']}) | "
                 f"بسته: {sig['close']:.2f}")
    lines.append(f"   • احتمال صعود {sig['up_prob']:.3f} (آستانه {sig['tau_up']:.3f}) | "
                 f"احتمال نزول {sig['dn_prob']:.3f} (آستانه {sig['tau_dn']:.3f})")
    lines.append(f"   • ADX={sig['adx']:.1f} (حداقل {sig['min_adx']:g}) | ATR={sig['atr']:.2f}")

    mom = sig.get("momentum")
    if mom:
        arrow = "📈" if mom["side"] == "LONG" else "📉"
        fa_dir = "صعودی" if mom["side"] == "LONG" else "نزولی"
        lines.append(f"   {arrow} روند {fa_dir} (مومنتوم): {mom['move_pct']:+.2f}٪ "
                     f"در {mom['window']} کندل ({mom['mom_r']:+.2f}R)")

    liq = sig.get("liquidation")
    if liq and liq.get("n_orders"):
        fa_press = ("شورت‌ها" if liq["imbalance"] > 0.05
                    else "لانگ‌ها" if liq["imbalance"] < -0.05 else "متوازن")
        lines.append(f"   💥 لیکوییدیشن {liq['window_hours']} ساعت اخیر: "
                     f"لانگ ${liq['long_notional'] / 1e6:.1f}M | "
                     f"شورت ${liq['short_notional'] / 1e6:.1f}M "
                     f"(فشار: {fa_press}، تراز {liq['imbalance']:+.2f})")

    if sig["side"] == "HOLD":
        reason = sig.get("hold_reason") or "هیچ‌کدام از آستانه‌ها فعال نشد"
        lines.append(f"   • نتیجه: 🤚 HOLD — {reason}")
    else:
        fa_side = "🟢 LONG (خرید)" if sig["side"] == "LONG" else "🔴 SHORT (فروش)"
        lines.append(f"   • نتیجه: {fa_side} — قدرت {sig['margin']:.2f} (حاشیه از آستانه)")
        lines.append(f"   • ورود: {sig['entry']:.2f} | SL: {sig['sl']:.2f} | "
                     f"حداکثر نگهداشت: {sig['max_hold']} کندل")
        for i, (t, f) in enumerate(zip(sig["tps"], (0.33, 0.33, 0.34)), 1):
            lines.append(f"       TP{i}: {t:.2f}  ({int(f * 100)}٪ حجم)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="سیگنال زنده‌ی مدل LGBM (پلن معاملاتی)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m",
                        help="باید با مدل آموزش‌دیده هماهنگ باشد (15m)")
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--tau-up", type=float, default=None,
                        help="نادیده‌گرفتن آستانه‌ی صعود مدل")
    parser.add_argument("--tau-dn", type=float, default=None,
                        help="نادیده‌گرفتن آستانه‌ی نزول مدل")
    
                        
                        
                        
    parser.add_argument("--tau-scale", type=float, default=1.0,
                        help="حساسیت‌زدن آستانه‌ها (۱=بدون تغییر؛ <۱ حساس‌تر)")
    parser.add_argument("--mom-atr-mult", type=float, default=1.0,
                        help="آستانه‌ی هشدار روند ریز بر حسب ATR")
    parser.add_argument("--mom-window", type=int, default=None,
                        help="پنجره‌ی مومنتوم به کندل (پیش‌فرض: horizon مدل)")
    parser.add_argument("--min-adx", type=float, default=None,
                        help="نادیده‌گرفتن حداقل ADX مدل (پیش‌فرض: از متادیتا)")
    parser.add_argument("--models-dir", default=MODELS_DIR)
    args = parser.parse_args()

    try:
        sig = build_live_signal(args.symbol, args.interval, args.bars, args.models_dir,
                                    min_adx=args.min_adx, tau_up=args.tau_up, tau_dn=args.tau_dn,
                                    tau_scale=args.tau_scale,
                                    mom_atr_mult=args.mom_atr_mult,
                                    mom_window=args.mom_window)
    except Exception as e:
        print(f"🔴 سیگنال زنده در دسترس نیست: {e}")
        return

    print("\n" + "=" * 80)
    print("🔔 LIVE MODEL SIGNAL")
    print("=" * 80)
    print(format_signal(sig))
    print("-" * 80)
    print("⚠️ یادآوری: همین قوانین در دو اجرای Walk-Forward (~۶۸۵ معامله، بعد از "
          "کارمزد و فاندینگ) edge اثبات‌شده نشان ندادند. خروجی، ورودی کمکی برای "
          "تصمیم انسانی است — نه توصیه‌ی معامله با پول واقعی.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()