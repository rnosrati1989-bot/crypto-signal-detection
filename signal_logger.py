"""
signal_logger.py
----------------
ثبت خودکار سیگنال‌های زنده (paper-trade log) + ارزیابی واقعی نتیجه‌ی آن‌ها.

چرا؟ سیگنال زنده را فقط با ثبت و ارزیابیِ واقعیِ بعدی می‌توان صادقانه سنجید.
هر بار که main.py اجرا می‌شود، سیگنال مدل در logs/signals.csv ذخیره می‌شود؛
بعداً این ماژول با داده‌ی واقعی فیوچرز، نتیجه‌ی همان سیگنال‌ها (SL/TP/زمان)
را دقیقاً با قوانین پلن (همان walk_forward_trade_plan) محاسبه می‌کند و جدول
امتیاز زنده (WinRate / میانگین R / Expectancy) را به‌روز نگه می‌دارد.

اجرا:
    ثبت خودکار : از داخل main.py (بخش سیگنال) انجام می‌شود.
    ارزیابی    : venv\\Scripts\\python.exe signal_logger.py --review
    بازمحاسبه  : venv\\Scripts\\python.exe signal_logger.py --recalc
                (نتیجه‌ی همه‌ی سیگنال‌ها پاک و با تنظیمات فعلی — مثلاً بدون کارمزد — دوباره محاسبه می‌شود)
    گزارش      : venv\\Scripts\\python.exe signal_logger.py --stats
    خلاصه‌ی روز : venv\\Scripts\\python.exe signal_logger.py --daily
    بازه‌ی دلخواه: venv\\Scripts\\python.exe signal_logger.py --daily --hours 24
                (یا --days 3 | --day 2026-10-15)

لاگ‌ها:
    logs\\signals.csv = فقط سیگنال‌های LONG/SHORT (معاملاتِ قابل‌ارزیابی)
    logs\\alerts.csv  = هر پیام ارسالی به تلگرام (TRADE/MOMENTUM/DIGEST/DAILY)
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd

from data.binance_futures_client import BinanceFuturesDataClient

SIGNALS_CSV = os.path.join("logs", "signals.csv")
# لاگ پیام‌های ارسالی به تلگرام: شامل مومنتوم/HOLD و دیجست وضعیت که «سیگنال
# معاملاتی» نیستند و در signals.csv ثبت نمی‌شوند. با این لاگ، تعداد پیام‌هایی که
# کاربر می‌بیند با تعداد معاملاتِ ثبت‌شده قابل‌مقایسه می‌شود.
ALERTS_CSV = os.path.join("logs", "alerts.csv")
ALERT_COLUMNS = ["time", "kind", "symbol", "side", "bar_time", "momentum", "detail"]
ALERT_KINDS = ("CONNECT", "TRADE", "MOMENTUM", "DIGEST", "DAILY")

COLUMNS = [
    "eval_time", "symbol", "interval", "bar_time", "close",
    "up_prob", "dn_prob", "tau_up", "tau_dn", "adx", "atr",
    "side", "margin", "entry", "sl", "tp", "tp1", "tp2", "tp3", "max_hold",
    "exit_time", "exit_reason", "tp_level", "net_pct", "r_mult", "resolved_at",
]


def _ensure_file():
    os.makedirs(os.path.dirname(SIGNALS_CSV), exist_ok=True)
    if not os.path.exists(SIGNALS_CSV):
        pd.DataFrame(columns=COLUMNS).to_csv(SIGNALS_CSV, index=False)
        return
    # مهاجرت: ستون‌های جدید (tp1/tp2/tp3/tp_level) را بدون از دست دادن ردیف‌ها اضافه کن
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        return
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = pd.NA
        if "tp1" in missing and "tp" in df.columns:
            df["tp1"] = df["tp"]
        df.reindex(columns=COLUMNS).to_csv(SIGNALS_CSV, index=False)


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------
# بازه‌ی گزارش
# ----------------------------------------------------------------------
def _window_mask(time_series, day: str = None, hours: int = None,
                 days: int = None):
    """
    ماسک ردیف‌های داخل بازه + برچسب توصیفی بازه. مبنای همه‌ی گزارش‌ها «زمان
    ثبت» (ساعت محلی) است.

    چرا؟ چون ربات شبانه‌روز می‌چرخد؛ گزارشِ فقط-روزِ-تقویمی باعث می‌شود
    سیگنال‌های بعد از نصف‌شب (که به روز قبل می‌افتند) در شمارش نیایند.
    """
    ts = pd.to_datetime(time_series, errors="coerce")
    now = pd.Timestamp(datetime.datetime.now())
    if hours:
        start = now - pd.Timedelta(hours=int(hours))
        label = (f"آخرین {int(hours)} ساعت ({start.strftime('%Y-%m-%d %H:%M')}"
                 f" → {now.strftime('%Y-%m-%d %H:%M')})")
        return ((ts >= start) & (ts <= now)).fillna(False), label, now
    if days:
        start = now.normalize() - pd.Timedelta(days=int(days) - 1)
        label = (f"آخرین {int(days)} روز ({start.strftime('%Y-%m-%d')}"
                 f" → {now.strftime('%Y-%m-%d')})")
        return ((ts >= start) & (ts <= now)).fillna(False), label, now
    d = str(day or now.strftime("%Y-%m-%d"))
    try:
        start = pd.Timestamp(d)
    except Exception:
        d = now.strftime("%Y-%m-%d")
        start = pd.Timestamp(d)
    end = start + pd.Timedelta(days=1)
    label = f"روز {d} (۰۰:۰۰ تا ۲۴:۰۰ به وقت محلی)"
    return ((ts >= start) & (ts < end)).fillna(False), label, now


# ----------------------------------------------------------------------
# لاگ پیام‌های ارسالی (برای مقایسه با لاگ معاملات)
# ----------------------------------------------------------------------
def _ensure_alerts_file():
    os.makedirs(os.path.dirname(ALERTS_CSV), exist_ok=True)
    if not os.path.exists(ALERTS_CSV):
        pd.DataFrame(columns=ALERT_COLUMNS).to_csv(ALERTS_CSV, index=False)


def append_alert(kind: str, symbol: str = "", side: str = "",
                 bar_time: str = "", momentum: str = "",
                 detail: str = "") -> bool:
    """
    ثبت یک پیامِ ارسال‌شده به تلگرام.
    kind: CONNECT | TRADE | MOMENTUM | DIGEST | DAILY
    (TRADE = سیگنال ورود مدل؛ MOMENTUM = هشدار روند ریز در حالت HOLD؛
     DIGEST = خلاصه‌ی دوره‌ای وضعیت؛ DAILY = خلاصه‌ی روزانه)
    """
    try:
        _ensure_alerts_file()
        row = {
            "time": _now_str(),
            "kind": str(kind),
            "symbol": str(symbol or ""),
            "side": str(side or ""),
            "bar_time": str(bar_time or ""),
            "momentum": str(momentum or ""),
            "detail": str(detail or "")[:120],
        }
        pd.DataFrame([row])[ALERT_COLUMNS].to_csv(ALERTS_CSV, mode="a",
                                                  header=False, index=False)
        return True
    except Exception:
        return False


def _with_alerts(out: dict, day: str = None, hours: int = None,
                 days: int = None) -> dict:
    """اطلاعات لاگ پیام‌های ارسالی را به خروجی گزارش اضافه می‌کند."""
    ac = alert_counts(day=day, hours=hours, days=days)
    out["alerts_total"] = ac["total"]
    out["alerts_by_kind"] = ac["by_kind"]
    if days and int(days) > 1:
        # جدول میانگین هر روز (برای --days N)
        out["day_rows"] = _daily_breakdown(int(days))
    return out
    return out


def alert_counts(day: str = None, hours: int = None, days: int = None) -> dict:
    """شمارش پیام‌های ارسالی در همان بازه‌ی گزارش (کل + به تفکیک kind)."""
    if not os.path.exists(ALERTS_CSV):
        return {"total": 0, "by_kind": {}}
    try:
        adf = pd.read_csv(ALERTS_CSV)
    except Exception:
        return {"total": 0, "by_kind": {}}
    if adf.empty or "time" not in adf.columns or "kind" not in adf.columns:
        return {"total": 0, "by_kind": {}}
    mask, _label, _now = _window_mask(adf["time"], day=day, hours=hours,
                                      days=days)
    win = adf[mask]
    return {
        "total": int(len(win)),
        "by_kind": {str(k): int(v) for k, v in win["kind"].value_counts().items()},
    }


# ----------------------------------------------------------------------
# ثبت
# ----------------------------------------------------------------------
def append_signal(sig: dict) -> bool:
    """
    افزودن یک ردیف سیگنال به log. اگر همان کندل/نماد/طرف قبلاً ثبت شده
    باشد، چیزی اضافه نمی‌کند (بدون تکرار). ردیف‌های HOLD بلافاصله
    NO_TRADE علامت می‌خورند تا در آمار معاملاتی شمرده نشوند.
    """
    _ensure_file()
    df = pd.read_csv(SIGNALS_CSV, dtype={"bar_time": str})
    bar_time = str(sig.get("bar_time", ""))
    symbol = str(sig.get("symbol", ""))
    interval = str(sig.get("interval", ""))
    side = str(sig.get("side", ""))

    if not df.empty:
        dup = df[(df["bar_time"] == bar_time) & (df["symbol"] == symbol) &
                 (df["interval"] == interval) & (df["side"] == side)]
        if not dup.empty:
            return False

    ev = _now_str()
    _tps = list(sig.get("tps") or [])
    _tp1 = sig.get("tp") if sig.get("tp") is not None else (_tps[0] if len(_tps) > 0 else None)
    _tp2 = _tps[1] if len(_tps) > 1 else None
    _tp3 = _tps[2] if len(_tps) > 2 else None
    row = {
        "eval_time": ev,
        "symbol": symbol,
        "interval": interval,
        "bar_time": bar_time,
        "close": sig.get("close"),
        "up_prob": sig.get("up_prob"),
        "dn_prob": sig.get("dn_prob"),
        "tau_up": sig.get("tau_up"),
        "tau_dn": sig.get("tau_dn"),
        "adx": sig.get("adx"),
        "atr": sig.get("atr"),
        "side": side,
        "margin": sig.get("margin"),
        "entry": sig.get("entry"),
        "sl": sig.get("sl"),
        "tp1": _tp1, "tp2": _tp2, "tp3": _tp3, "tp": _tp1,
               # نردبان TP جداگانه در tp1/tp2/tp3 ذخیره می‌شود
        "max_hold": sig.get("max_hold"),
        "exit_time": "",
        "exit_reason": "NO_TRADE" if side not in ("LONG", "SHORT") else "",
        "tp_level": 0 if side not in ("LONG", "SHORT") else "",
        "net_pct": "",
        "r_mult": "",
        "resolved_at": ev if side not in ("LONG", "SHORT") else "",
    }
    pd.DataFrame([row])[COLUMNS].to_csv(SIGNALS_CSV, mode="a", header=False, index=False)
    return True


# ----------------------------------------------------------------------
# ارزیابی واقعی سیگنال‌های ثبت‌شده
# ----------------------------------------------------------------------
def _clear_resolution(interval: str = None) -> int:
    """
    پاک‌کردن نتیجه‌ی ارزیابیِ همه‌ی سیگنال‌های LONG/SHORT تا با تنظیمات جدید
    (مثلاً بدون کارمزد) دوباره از داده‌ی واقعی محاسبه شوند.
    """
    _ensure_file()
    df = pd.read_csv(SIGNALS_CSV)
    if df.empty:
        return 0
    m = df["side"].isin(["LONG", "SHORT"])
    if interval:
        m &= (df["interval"] == interval)
    n = int(m.sum())
    for _c in ("exit_time", "exit_reason", "tp_level", "net_pct", "r_mult", "resolved_at"):
        if _c in df.columns:
            df[_c] = df[_c].astype("object")
        df.loc[m, _c] = ""
    df.to_csv(SIGNALS_CSV, index=False)
    return n


def review_pending(interval: str = "15m", fee: float = 0.0,
                   use_funding: bool = True) -> int:
    """
    برای هر سیگنال LONG/SHORTِ ثبت‌شده که هنوز نتیجه‌اش محاسبه نشده، با
    داده‌ی واقعی فیوچرز، خروج را با قوانین پلن (SL مقدم بر TP، خروج زمان
    در max_hold، فاندینگ؛ کارمزد پیش‌فرض ۰ = بدون کارمزد) تعیین و ردیف را به‌روز می‌کند.
    """
    _ensure_file()
    df = pd.read_csv(SIGNALS_CSV)
    if df.empty:
        return 0

    # ستون‌های رشته‌ای را object کن تا نوشتن مقدار روی ستونِ خالی (float64) خطا ندهد
    for _c in ("exit_time", "exit_reason", "resolved_at", "tp_level", "net_pct", "r_mult"):
        if _c in df.columns:
            df[_c] = df[_c].astype("object")
    unresolved_mask = df["exit_reason"].isna() | (df["exit_reason"] == "")
    pending = df[unresolved_mask]
    if pending.empty:
        return 0

    client = BinanceFuturesDataClient()
    resolved = 0
    for idx in pending.index:
        row = df.loc[idx]
        symbol = str(row["symbol"])
        interval_row = str(row["interval"])
        side = str(row["side"]).lower()
        if side not in ("long", "short"):
            continue
        try:
            raw = client.get_futures_dataset(symbol, interval_row, limit=1500)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        try:
            entry_idx = raw.index.get_loc(pd.Timestamp(row["bar_time"]))
        except KeyError:
            continue  # کندل از تاریخچه خارج شده؟ ردیف دست‌نخورده می‌ماند

        if entry_idx + 1 >= len(raw):
            continue  # هنوز کندل بعدی نیامده

        max_hold = int(row["max_hold"]) if pd.notna(row["max_hold"]) else 4
        entry = float(row["entry"])
        sl = float(row["sl"])
        _lv = []
        for _c in ("tp1", "tp2", "tp3"):
            _v = row.get(_c)
            _lv.append(float(_v) if pd.notna(_v) else None)
        if all(_v is None for _v in _lv):
            _v = row.get("tp")
            _lv = [float(_v) if pd.notna(_v) else None, None, None]
        tps = [x for x in _lv if x is not None]
        if not tps:
            continue
        tp = tps[0]
        closes = raw["close"].values
        highs = raw["high"].values
        lows = raw["low"].values
        funding = (raw["funding_rate"].fillna(0).values
                   if "funding_rate" in raw.columns
                   else np.zeros(len(raw)))

        # بالاترین پله‌ی TP که قیمت در پنجره‌ی نگهداشت لمس کرده (۱=TP1، ۲=TP2، ۳=TP3)
        tp_reached = 0
        for j3 in range(1, max_hold + 1):  # اسکن اولیه (کل پنجره) — مقدار نهایی پایین‌تر بازنویسی می‌شود
            i3 = entry_idx + j3
            if i3 >= len(raw):
                break
            for rung in range(len(tps), 0, -1):
                lvl = tps[rung - 1]
                touched = (highs[i3] >= lvl) if side == "long" else (lows[i3] <= lvl)
                if touched:
                    if rung > tp_reached:
                        tp_reached = rung
                    break

        exit_price = None
        reason = None
        exit_i = None
        for j in range(1, max_hold + 1):
            i = entry_idx + j
            if i >= len(raw):
                break
            if side == "long":
                hit_sl = lows[i] <= sl
                hit_tp = highs[i] >= tp
            else:
                hit_sl = highs[i] >= sl
                hit_tp = lows[i] <= tp
            if hit_sl and hit_tp:
                exit_price, reason = sl, "SL"  # اولویت SL (محافظه‌کارانه)
            elif hit_sl:
                exit_price, reason = sl, "SL"
            elif hit_tp:
                exit_price, reason = tp, "TP"
            elif j >= max_hold:
                exit_price, reason = closes[i], "TIME"
            else:
                continue
            exit_i = i
            break
        if exit_price is None:
            continue

        if side == "long":
            raw_pct = (exit_price / entry - 1.0) * 100.0
        else:
            raw_pct = (entry / exit_price - 1.0) * 100.0
        net_pct = raw_pct - 2 * fee * 100.0  # کارمزد ورود + خروج

        if use_funding:
            fcost = 0.0
            for j2 in range(entry_idx + 1, exit_i + 1):
                if funding[j2] != funding[j2 - 1]:
                    fcost += funding[j2] if side == "long" else -funding[j2]
            net_pct -= fcost * 100.0

        # بازمحاسبه‌ی پله‌های TP — فقط تا کندل خروج:
        # اسکنِ بالایی کل پنجره‌ی نگهداشت را نگاه می‌کرد و ردیفی که اول SL خورده و
        # بعد قیمت برگشته بود را هم «رسیده به TP» می‌شمرد (آمار گمراه‌کننده: هم SL،
        # هم TP در یک ردیف). اینجا مقدار نهایی محدود به کندل‌های قبل از خروج است؛
        # اگر خروج TP بود، خودِ کندل خروج هم شمرده می‌شود.
        tp_reached = 0
        _last_bar = (exit_i - entry_idx) if reason == "TP" else (exit_i - entry_idx - 1)
        for j4 in range(1, _last_bar + 1):
            i4 = entry_idx + j4
            for rung in range(len(tps), 0, -1):
                lvl = tps[rung - 1]
                if (highs[i4] >= lvl) if side == "long" else (lows[i4] <= lvl):
                    if rung > tp_reached:
                        tp_reached = rung
                    break

        sl_dist_pct = abs(entry - sl) / entry * 100.0
        r_mult = net_pct / sl_dist_pct if sl_dist_pct > 0 else 0.0

        df.at[idx, "exit_time"] = str(raw.index[exit_i])
        df.at[idx, "exit_reason"] = reason
        df.at[idx, "tp_level"] = tp_reached
        df.at[idx, "net_pct"] = round(net_pct, 4)
        df.at[idx, "r_mult"] = round(r_mult, 4)
        df.at[idx, "resolved_at"] = _now_str()
        resolved += 1

    df.to_csv(SIGNALS_CSV, index=False)
    return resolved


# ----------------------------------------------------------------------
# آمار
# ----------------------------------------------------------------------
def stats() -> dict:
    """جدول امتیاز زنده از سیگنال‌هایی که نتیجه‌شان واقعاً مشخص شده."""
    _ensure_file()
    df = pd.read_csv(SIGNALS_CSV)
    total = len(df)
    if total == 0:
        return None

    trades = df[df["exit_reason"].isin(["SL", "TP", "TIME"])].copy()
    pending = int((df["exit_reason"].isna() | (df["exit_reason"] == "")).sum())
    no_trade = int((df["exit_reason"] == "NO_TRADE").sum())

    out = {
        "signals_total": total,
        "pending": pending,
        "no_trade": no_trade,
        "trades": 0, "win_rate": None, "avg_r": None,
        "exp_pct": None, "by_side": {},
    }
    if trades.empty:
        return out

    rs = trades["r_mult"].astype(float)
    pn = trades["net_pct"].astype(float)
    out.update({
        "trades": len(trades),
        "win_rate": float((rs > 0).mean() * 100),
        "avg_r": float(rs.mean()),
        "exp_pct": float(pn.mean()),
    })
    for side in ("long", "short"):
        sub = trades[trades["side"].str.lower() == side]
        if not sub.empty:
            rs2 = sub["r_mult"].astype(float)
            out["by_side"][side] = {
                "n": len(sub),
                "win_rate": float((rs2 > 0).mean() * 100),
                "avg_r": float(rs2.mean()),
            }
    return out


def open_signal(symbol: str, interval: str = None) -> dict:
    """آخرین سیگنال LONG/SHORTِ هنوز باز (نتیجه‌اش مشخص نشده) برای یک نماد.
    برای جلوگیری از تکرار سیگنال تا وقتی معامله‌ی قبلی باز است."""
    _ensure_file()
    df = pd.read_csv(SIGNALS_CSV)
    if df.empty:
        return None
    m = (df["symbol"] == symbol) & (df["side"].isin(["LONG", "SHORT"]))
    m &= (df["exit_reason"].isna() | (df["exit_reason"] == ""))
    if interval:
        m &= (df["interval"] == interval)
    sub = df[m]
    return None if sub.empty else sub.iloc[-1].to_dict()


def daily_summary_stats(day: str = None, hours: int = None, days: int = None) -> dict:
    """آمار سیگنال‌های LONG/SHORT در یک بازه: چند درصد TP/SL/زمانی/باز.
    بازه (بر مبنای eval_time، ساعت محلی): پیش‌فرض = امروز؛ با hours/days = بازه‌ی غلتان.
    outside/outside_open = ردیف‌های خارج از بازه؛ alerts_* = لاگ پیام‌های ارسالی."""
    # بازه: پیش‌فرض امروز (به وقت محلی)؛ با hours/days بازه‌ی غلتان می‌شود.
    _ensure_file()
    if day is not None:
        day = str(day)
    # (day=None → امروز؛ فیلترِ واقعیِ بازه پایین‌تر با _window_mask انجام می‌شود)
    df = pd.read_csv(SIGNALS_CSV)
    out = {"day": "", "window": "", "signals": 0, "tp1": 0, "tp2": 0, "tp3": 0, "sl": 0, "time": 0,
           "open": 0, "tp1_pct": 0.0, "tp2_pct": 0.0, "tp3_pct": 0.0, "sl_pct": 0.0, "time_pct": 0.0, "win_pct": None,
           "outside": 0, "outside_open": 0, "rows": [], "n_done": 0, "net_sum": 0.0,
           "net_avg": None, "net_median": None, "net_best": None, "net_worst": None,
           "day_rows": []}
    if df.empty:
        return _with_alerts(out, day, hours, days)
    sig = df[df["side"].isin(["LONG", "SHORT"])].copy()
    if sig.empty:
        return _with_alerts(out, day, hours, days)
    mask, label, now = _window_mask(sig["eval_time"], day=day, hours=hours,
                                    days=days)
    out["window"] = label
    out["day"] = (f"آخرین {int(hours)} ساعت" if hours
                  else f"آخرین {int(days)} روز" if days
                  else str(day or now.strftime("%Y-%m-%d")))
    outside = sig[~mask]
    out["outside"] = int(len(outside))
    out["outside_open"] = int((outside["exit_reason"].isna()
                               | (outside["exit_reason"] == "")).sum())
    sig = sig[mask]
    out["signals"] = len(sig)
    if sig.empty:
        return _with_alerts(out, day, hours, days)
    lvl = pd.to_numeric(sig["tp_level"], errors="coerce").fillna(0)
    n_tp1 = int((lvl >= 1).sum()); n_tp2 = int((lvl >= 2).sum()); n_tp3 = int((lvl >= 3).sum())
    n_sl = int((sig["exit_reason"] == "SL").sum())
    n_time = int((sig["exit_reason"] == "TIME").sum())
    n_open = int((sig["exit_reason"].isna() | (sig["exit_reason"] == "")).sum())
    n_all = max(1, len(sig))
    out.update({
        "tp1": n_tp1, "tp2": n_tp2, "tp3": n_tp3, "sl": n_sl, "time": n_time, "open": n_open,
        "tp1_pct": n_tp1 / n_all * 100.0, "tp2_pct": n_tp2 / n_all * 100.0, "tp3_pct": n_tp3 / n_all * 100.0,
        "sl_pct": n_sl / n_all * 100.0,
        "time_pct": n_time / n_all * 100.0,
    })
    # درصد هر سیگنال (net_pct = خروج نسبت به ورود، منهای کارمزد رفت‌وبرگشت و
    # فاندینگ) + میانگین‌ها. ردیف‌های هنوز-باز net_pct ندارند و در میانگین شمرده نمی‌شوند.
    pn = pd.to_numeric(sig["net_pct"], errors="coerce")
    rr = pd.to_numeric(sig["r_mult"], errors="coerce")
    rows = []
    for _i, _r in sig.sort_values("eval_time").iterrows():
        _ex = _r.get("exit_reason")
        rows.append({
            "eval_time": str(_r.get("eval_time", "")),
            "symbol": str(_r.get("symbol", "")),
            "side": str(_r.get("side", "")),
            "bar_time": str(_r.get("bar_time", "")),
            "exit_reason": "" if pd.isna(_ex) else str(_ex),
            "tp_level": int(lvl.loc[_i]) if _i in lvl.index else 0,
            "net_pct": (float(pn.loc[_i]) if pd.notna(pn.loc[_i]) else None),
            "r_mult": (float(rr.loc[_i]) if pd.notna(rr.loc[_i]) else None),
        })
    out["rows"] = rows
    _done = pn.dropna()
    out["n_done"] = int(len(_done))
    out["net_sum"] = float(_done.sum()) if len(_done) else 0.0
    out["net_avg"] = float(_done.mean()) if len(_done) else None
    out["net_median"] = float(_done.median()) if len(_done) else None
    out["net_best"] = float(_done.max()) if len(_done) else None
    out["net_worst"] = float(_done.min()) if len(_done) else None
    resolved = sig[sig["exit_reason"].isin(["TP", "SL", "TIME"])]
    if not resolved.empty:
        out["win_pct"] = float((resolved["r_mult"].astype(float) > 0).mean() * 100.0)
    return _with_alerts(out, day, hours, days)


def _daily_breakdown(days: int) -> list:
    """میانگین درصد هر روز از آخرین N روزِ موجود در لاگ (برای --days N)."""
    _ensure_file()
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        return []
    if df.empty:
        return []
    sig = df[df["side"].isin(["LONG", "SHORT"])].copy()
    if sig.empty:
        return []
    sig["_d"] = pd.to_datetime(sig["eval_time"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = []
    for d in sorted(sig["_d"].dropna().unique())[-max(1, int(days)):]:
        sub = sig[sig["_d"] == d]
        pn = pd.to_numeric(sub["net_pct"], errors="coerce").dropna()
        rr = pd.to_numeric(sub["r_mult"], errors="coerce").dropna()
        out.append({
            "day": str(d),
            "signals": int(len(sub)),
            "resolved": int(len(pn)),
            "net_sum": float(pn.sum()) if len(pn) else 0.0,
            "net_avg": float(pn.mean()) if len(pn) else None,
            "win_pct": float((rr > 0).mean() * 100.0) if len(rr) else None,
        })
    return out


def _fmt_signal_rows(s: dict, limit: int = 60) -> list:
    """خطوط «درصد هر سیگنال» + جمع/میانگین بازه."""
    rows = s.get("rows") or []
    if not rows:
        return []
    limit = max(1, int(limit))
    lines = ["", "📈 درصد هر سیگنال (بدون کارمزد؛ فاندینگ لحاظ شده | زمان کندل به UTC):"]
    for r in rows[:limit]:
        p = (f"{r['net_pct']:+.2f}٪" if r["net_pct"] is not None else "—")
        rr = (f"{r['r_mult']:+.2f}R" if r["r_mult"] is not None else "—")
        st = r["exit_reason"] or "باز"
        tp = f" TP{r['tp_level']}" if r["tp_level"] else ""
        lines.append(f"  • {str(r['bar_time'])[5:16]} {r['symbol']} {r['side']} → "
                     f"{st}{tp} | {p} ({rr})")
    if len(rows) > limit:
        lines.append(f"  … و {len(rows) - limit} سیگنال دیگر")
    if s.get("n_done"):
        lines.append(f"• ارزیابی‌شده: {s['n_done']} از {s['signals']} | "
                     f"جمع: {s['net_sum']:+.2f}٪ | میانگین: {s['net_avg']:+.3f}٪ | "
                     f"میانه: {s['net_median']:+.3f}٪")
        lines.append(f"• بهترین: {s['net_best']:+.2f}٪ | بدترین: {s['net_worst']:+.2f}٪")
    return lines


def _fmt_alerts_footer(s: dict) -> list:
    """خطوط پایانی گزارش: تعداد پیام‌های ارسالی + هشدار سیگنال‌های خارج از بازه."""
    extra = []
    _drows = s.get("day_rows") or []
    if _drows:
        extra.append("📆 میانگین روزانه:")
        for _d in _drows:
            _avg = (f"{_d['net_avg']:+.3f}٪" if _d["net_avg"] is not None else "—")
            _win = (f"{_d['win_pct']:.0f}٪" if _d["win_pct"] is not None else "—")
            extra.append(f"  {_d['day']}: {_d['signals']} سیگنال | "
                         f"{_d['resolved']} ارزیابی‌شده | میانگین {_avg} | "
                         f"جمع {_d['net_sum']:+.2f}٪ | برد {_win}")
        extra.append("")
    if s.get("outside"):
        op = (f" (از این‌ها {s['outside_open']} هنوز باز)"
              if s.get("outside_open") else "")
        extra.append(f"• در کل لاگ و خارج از این بازه: {s['outside']} سیگنال{op}")
        extra.append("  بازه‌ی بزرگ‌تر: --days 3 | --hours 48 | --day YYYY-MM-DD")
    total = int(s.get("alerts_total") or 0)
    if total:
        fa = {"TRADE": "سیگنال معاملاتی", "MOMENTUM": "مومنتوم/HOLD",
              "DIGEST": "دیجست وضعیت", "DAILY": "گزارش روزانه",
              "CONNECT": "اتصال ربات"}
        bk = s.get("alerts_by_kind") or {}
        parts = [f"{fa.get(k, k)}: {v}"
                 for k, v in sorted(bk.items(), key=lambda x: -x[1])]
        extra.append(f"• پیام‌های ارسالی به تلگرام: {total}  ({' | '.join(parts)})")
    else:
        extra.append("• لاگ پیام‌های ارسالی خالی است "
                     "(از این پس با اجرای ربات ثبت می‌شود).")
    return extra


def _num(x):
    """تبدیل امن به float (None اگر نشد)."""
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _hit_rate_bucket(sig: pd.DataFrame, mask, label: str) -> list:
    """یک بلوک «کارنامه»: چند درصد سیگنال‌ها TP خوردند / SL خوردند."""
    sub = sig[mask]
    n = len(sub)
    if n == 0:
        return [f"▪️ {label}: هیچ سیگنالی ثبت نشد."]
    lvl = pd.to_numeric(sub["tp_level"], errors="coerce").fillna(0)
    n_tp = int((lvl >= 1).sum())
    n_sl = int((sub["exit_reason"] == "SL").sum())
    n_time = int((sub["exit_reason"] == "TIME").sum())
    n_open = int((sub["exit_reason"].isna() | (sub["exit_reason"] == "")).sum())
    denom = max(1, n_tp + n_sl + n_time)  # درصدها نسبت به «ارزیابی‌شده‌ها»
    pn = pd.to_numeric(sub["net_pct"], errors="coerce").dropna()
    avg = f"{pn.mean():+.3f}٪" if len(pn) else "—"
    return [
        f"▪️ {label}: {n} سیگنال",
        f"    حد سود (TP): {n_tp} ({n_tp / denom * 100:.0f}٪) | "
        f"حد ضرر (SL): {n_sl} ({n_sl / denom * 100:.0f}٪) | "
        f"خروج زمانی: {n_time} ({n_time / denom * 100:.0f}٪) | باز: {n_open}",
        f"    میانگین هر سیگنال: {avg}",
    ]


def format_feedback_report(limit: int = 15) -> str:
    """
    کارنامه‌ی سیگنال‌های ارسالی: چند درصد حد سود خورده‌اند و چند درصد حد ضرر.
    سه بازه: امروز | آخرین ۲۴ ساعت | از ابتدا (کل لاگ) + فهرست آخرین سیگنال‌ها.

    مبنا: logs/signals.csv (فقط ردیف‌های LONG/SHORT). TP = رسیدن به اولین پله‌ی
    سود، SL = خوردن حد ضرر، «باز» = هنوز به نتیجه نرسیده.
    """
    _ensure_file()
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:
        df = pd.DataFrame()
    now = pd.Timestamp(datetime.datetime.now())
    lines = [f"📊 کارنامه‌ی سیگنال‌ها — {now.strftime('%Y-%m-%d %H:%M')}", ""]
    if df.empty or not df["side"].isin(["LONG", "SHORT"]).any():
        lines.append("• هنوز هیچ سیگنال LONG/SHORT ثبت نشده است.")
        lines.append("")
        lines.append("⚠️ این گزارش فقط پایش آماری است، نه اثبات سوددهی.")
        return "\n".join(lines)

    sig = df[df["side"].isin(["LONG", "SHORT"])].copy()
    ts = pd.to_datetime(sig["eval_time"], errors="coerce")
    lines += _hit_rate_bucket(sig, (ts >= now.normalize()) & (ts <= now),
                              f"امروز ({now.strftime('%Y-%m-%d')})")
    lines += _hit_rate_bucket(sig, (ts >= now - pd.Timedelta(hours=24)) & (ts <= now),
                              "آخرین ۲۴ ساعت")
    lines += _hit_rate_bucket(sig, ts.notna(), f"از ابتدا (کل لاگ)")

    rows = sig.sort_values("eval_time").tail(max(1, int(limit)))
    lines.append("")
    lines.append(f"📈 {len(rows)} سیگنال آخر (زمان کندل به UTC):")
    for _i, _r in rows.iterrows():
        _ex = _r.get("exit_reason")
        _ex = "" if pd.isna(_ex) else str(_ex)
        _lvl = _num(_r.get("tp_level")) or 0
        _pn = _num(_r.get("net_pct"))
        _rr = _num(_r.get("r_mult"))
        p = f"{_pn:+.2f}٪" if _pn is not None else "—"
        r = f"{_rr:+.2f}R" if _rr is not None else "—"
        tp = f" TP{int(_lvl)}" if _lvl else ""
        lines.append(f"  • {str(_r.get('bar_time', ''))[5:16]} {_r.get('symbol')} "
                     f"{_r.get('side')} → {(_ex or 'باز')}{tp} | {p} ({r})")
    lines.append("")
    lines.append("ℹ️ مبنای درصدها: بدون کارمزد؛ فاندینگ لحاظ شده.")
    lines.append("⚠️ «حد سود خورده» = رسیدن به اولین پله‌ی سود. نمونه‌ی کوچک = آمار "
                 "نوسانی؛ این گزارش پایش است، نه اثبات سوددهی.")
    return "\n".join(lines)


def format_daily_summary(day: str = None, hours: int = None, days: int = None) -> str:
    """متن خلاصه‌ی روزانه برای تلگرام: درصد TP در مقابل SL."""
    s = daily_summary_stats(day=day, hours=hours, days=days)
    lines = [f"📅 تحلیل سیگنال‌های {s['day']}"]
    if s["window"]:
        lines.append(f"• بازه: {s['window']}")
    if s["signals"] == 0:
        lines.append("• در این بازه هیچ سیگنال LONG/SHORT ثبت نشد.")
        lines += _fmt_alerts_footer(s)
        lines.append("")
        lines.append("⚠️ چند سیگنال = نمونه‌ی کوچک؛ این آمار برای پایش است، نه اثبات سوددهی.")
        return "\n".join(lines)
    # درصدها نسبت به کل سیگنال‌های همان روز محاسبه شده‌اند
    lines.append(f"• کل سیگنال LONG/SHORT: {s['signals']}")
    lines.append(f"• رسیده به TP1 (1R): {s['tp1']}  ({s['tp1_pct']:.0f}٪)")
    lines.append(f"• رسیده به TP2 (2R): {s['tp2']}  ({s['tp2_pct']:.0f}٪)")
    lines.append(f"• رسیده به TP3 (3R): {s['tp3']}  ({s['tp3_pct']:.0f}٪)")
    lines.append(f"• خورده به حد ضرر (SL): {s['sl']}  ({s['sl_pct']:.0f}٪)")
    lines.append(f"• خروج زمانی (TIME): {s['time']}  ({s['time_pct']:.0f}٪)")
    if s["win_pct"] is not None:
        lines.append(f"• نرخ برد (R>0): {s['win_pct']:.0f}٪")
    lines.append(f"• هنوز باز: {s['open']}")
    lines += _fmt_signal_rows(s)
    lines += _fmt_alerts_footer(s)
    lines.append("")
    lines.append("⚠️ چند سیگنال = نمونه‌ی کوچک؛ این آمار برای پایش است، نه اثبات سوددهی.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ثبت و ارزیابی سیگنال‌های زنده (paper-trade log)")
    parser.add_argument("--review", action="store_true", help="ارزیابی سیگنال‌های در انتظار")
    parser.add_argument("--stats", action="store_true", help="نمایش جدول امتیاز")
    parser.add_argument("--daily", action="store_true",
                        help="خلاصه‌ی روزانه‌ی سیگنال‌ها (درصد TP/SL)")
    parser.add_argument("--day", default=None,
                        help="روز گزارش به شکل YYYY-MM-DD (پیش‌فرض: امروز)")
    parser.add_argument("--days", type=int, default=None,
                        help="بازه‌ی غلتان: N روز اخیر (مثلاً --days 2)")
    parser.add_argument("--hours", type=int, default=None,
                        help="بازه‌ی غلتان: N ساعت اخیر (مثلاً --hours 24)")
    parser.add_argument("--fee", type=float, default=0.0,
                        help="کارمزد هر سمت (پیش‌فرض ۰ = بدون کارمزد)")
    parser.add_argument("--recalc", action="store_true",
                        help="پاک‌کردن نتیجه‌ی همه‌ی سیگنال‌ها و ارزیابی دوباره با تنظیمات فعلی")
    parser.add_argument("--no-funding", action="store_true")
    args = parser.parse_args()

    if args.daily:
        print(format_daily_summary(day=args.day, hours=args.hours, days=args.days))
        return

    if args.recalc:
        # همه‌ی سیگنال‌ها (حتی بسته‌شده‌ها) با تنظیمات فعلی — مثلاً بدون کارمزد —
        # دوباره از داده‌ی واقعی محاسبه می‌شوند تا آمار یکدست بماند.
        n0 = _clear_resolution()
        n = review_pending(fee=args.fee, use_funding=not args.no_funding)
        print(f"♻️ {n0} ردیف بازنشانی شد و {n} سیگنال دوباره با کارمزد "
              f"{args.fee * 100:.3f}٪ ارزیابی شد.")
        return

    if args.review:
        n = review_pending(fee=args.fee, use_funding=not args.no_funding)
        print(f"📝 {n} سیگنال با داده‌ی واقعی ارزیابی شد.")

    s = stats()
    if s:
        print("\n" + "=" * 70)
        print("📊 PAPER-TRADE SCOREBOARD (ثبت زنده)")
        print("=" * 70)
        print(f"• کل سیگنال‌های ثبت‌شده : {s['signals_total']}")
        print(f"• در انتظار نتیجه       : {s['pending']}")
        print(f"• بدون ورود (HOLD)      : {s['no_trade']}")
        if s["trades"]:
            print(f"• معاملات ارزیابی‌شده : {s['trades']} | WinRate: {s['win_rate']:.0f}٪ | "
                  f"میانگین: {s['avg_r']:+.3f}R | سود خالص: {s['exp_pct']:+.3f}٪")
            for side, sd in s["by_side"].items():
                fa = "لانگ" if side == "long" else "شورت"
                print(f"   • {fa}: {sd['n']} معامله | WR {sd['win_rate']:.0f}٪ | "
                      f"{sd['avg_r']:+.3f}R")
        else:
            print("• هنوز هیچ معامله‌ای به نتیجه نرسیده (کمی صبر و داده‌ی جدید).")
        print("-" * 70)
        print("⚠️ این جدول فقط ثبت زنده است؛ برای edge باید هم‌زمان احتمال "
              "تصادفی (random) هم سنجیده شود.")
        print("=" * 70 + "\n")
    else:
        print("📭 هنوز سیگنالی ثبت نشده. اول main.py را اجرا کنید.")


if __name__ == "__main__":
    main()