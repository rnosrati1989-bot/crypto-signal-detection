"""
strategies/trade_plan.py
------------------------
لایه‌ی «پلن معاملاتی» (Trade Plan): تبدیل خروجی مدل به نقطه‌ی ورود/خروج دقیق.

این ماژول پاسخ به این خواسته است: «سیستم فقط جهت می‌دهد؛ نقطه‌ی ورود و خروج
دقیق برای فیوچرز کجاست؟»

اصول:
1. همه‌ی قیمت‌ها به‌صورت قطعی (deterministic) از داده‌ی واقعی و خروجیِ عددیِ
   مدل محاسبه می‌شوند. هوش مصنوعی متنی (LLM) هرگز عدد نمی‌سازد — فقط تفسیر
   می‌کند و عددها را از این ماژول می‌گیرد.
2. قوانین دقیقاً همان قوانین موتور بک‌تست هستند (ورود -> SL با ATR -> TP با RR)
   تا اعتبارسنجی Walk-Forward دقیقاً رفتاری را بسنجد که در حالت زنده اجرا می‌شود.
3. هیچ‌وقت پلنی بدون حد ضرر تولید نمی‌شود.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# پیش‌فرض‌های منطقی (در محل فراخوانی قابل تغییر هستند)
DEFAULT_ATR_SL_MULT = 1.5   # حد ضرر = ATR * 1.5
DEFAULT_RISK_REWARD = 2.0   # هدف = فاصله‌ی حد ضرر * 2
DEFAULT_RISK_PCT = 1.0      # ریسک هر معامله: ۱٪ حساب
DEFAULT_MIN_ADX = 30.0      # فیلتر رژیم: فقط در بازار رونددار
DEFAULT_BALANCE = 1000.0    # موجودی فرضی حساب (در .env قابل تنظیم است)
DEFAULT_FEE_PCT = 0.05      # کارمزد هر طرف (درصد) — تیکر فیوچرز بایننس
DEFAULT_VALIDITY_CANDLES = 2  # پلن چند کندل معتبر است


@dataclass
class TradePlan:
    """پلن کامل یک معامله: جهت + ورود + حد ضرر + هدف + سایز.

    همه‌ی مقادیر عددی از داده‌ی واقعی و خروجی مدل ساخته شده‌اند؛
    هیچ عدد دستی/ساختگی در کار نیست.
    """
    symbol: str
    timeframe: str
    direction: str                  # "long" یا "short"
    entry: float                    # قیمت ورود (آخرین قیمت / باز شدن کندل بعد)
    sl: float                       # حد ضرر
    tp: float                       # هدف (Take Profit)
    risk_reward: float              # نسبت پاداش به ریسک (همان RR)
    sl_distance_pct: float          # فاصله‌ی SL تا ورود (درصد)
    tp_distance_pct: float          # فاصله‌ی TP تا ورود (درصد)
    up_prob: float                  # احتمال صعود از مدل up
    down_prob: float                # احتمال نزول از مدل down
    tau_up: float                   # آستانه‌ی صعود (چندک Train)
    tau_dn: float                   # آستانه‌ی نزول (چندک Train)
    adx: float                      # ADX فعلی (فیلتر رژیم)
    risk_pct: float                 # ریسک هر معامله از حساب (درصد)
    risk_amount: float              # مقدار ریسک به واحد حساب
    notional: float                 # ارزش نوتینال پوزیشن پیشنهادی
    quantity: float                 # حجم پیشنهادی (به واحد کوین)
    fee_warning: bool               # هشدار: هدف آن‌قدر نزدیک است که کارمزد می‌خورد
    net_tp_pct: float               # سود خالص پس از کارمزد رفت‌وبرگشت (اگر TP بخورد)
    validity_candles: int           # پلن تا چند کندل بعد معتبر است
    created_at: str                 # زمان ساخت (UTC ISO)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "risk_reward": self.risk_reward,
            "sl_distance_pct": self.sl_distance_pct,
            "tp_distance_pct": self.tp_distance_pct,
            "up_prob": self.up_prob,
            "down_prob": self.down_prob,
            "tau_up": self.tau_up,
            "tau_dn": self.tau_dn,
            "adx": self.adx,
            "risk_pct": self.risk_pct,
            "risk_amount": self.risk_amount,
            "notional": self.notional,
            "quantity": self.quantity,
            "fee_warning": self.fee_warning,
            "net_tp_pct": self.net_tp_pct,
            "validity_candles": self.validity_candles,
            "created_at": self.created_at,
        }


def _fmt(x: float, dec: int = 2) -> str:
    """عدد با جداکننده‌ی هزارگان."""
    return f"{x:,.{dec}f}"


def build_trade_plan(
    symbol: str,
    timeframe: str,
    entry: float,
    atr: float,
    up_prob: float,
    down_prob: float,
    tau_up: float,
    tau_dn: float,
    adx: float,
    *,
    min_adx: float = DEFAULT_MIN_ADX,
    atr_sl_mult: float = DEFAULT_ATR_SL_MULT,
    risk_reward: float = DEFAULT_RISK_REWARD,
    risk_pct: float = DEFAULT_RISK_PCT,
    balance: float = DEFAULT_BALANCE,
    fee_pct: float = DEFAULT_FEE_PCT,
    validity_candles: int = DEFAULT_VALIDITY_CANDLES,
) -> Optional[TradePlan]:
    """ساخت پلن معاملاتی از خروجی مدل و وضعیت بازار.

    :return: TradePlan اگر سیگنال معتبر باشد، وگرنه None (بدون سیگنال).

    منطق جهت (دقیقاً مشابه walk_forward_dual):
      - LONG  اگر up_prob   >= tau_up
      - SHORT اگر down_prob >= tau_dn
      - اگر هر دو فعال باشند، سمت با «فاصله‌ی بیشتر از آستانه‌ی خودش» برنده است.
    فیلتر رژیم: اگر ADX < min_adx باشد هیچ سیگنالی صادر نمی‌شود.
    """
    if entry is None or entry <= 0 or atr is None or atr <= 0:
        return None

    if adx < min_adx:
        return None

    long_fire = up_prob >= tau_up
    short_fire = down_prob >= tau_dn

    if long_fire and short_fire:
        direction = "long" if (up_prob - tau_up) >= (down_prob - tau_dn) else "short"
    elif long_fire:
        direction = "long"
    elif short_fire:
        direction = "short"
    else:
        return None

    # ---- حد ضرر و هدف (همان قوانین بک‌تست) ----
    sl_distance = atr * atr_sl_mult
    if direction == "long":
        sl = entry - sl_distance
        tp = entry + sl_distance * risk_reward
    else:
        sl = entry + sl_distance
        tp = entry - sl_distance * risk_reward

    sl_dist_pct = abs(entry - sl) / entry
    tp_dist_pct = abs(tp - entry) / entry

    # ---- سایز پوزیشن بر اساس ریسک ثابت (۱٪) ----
    risk_amount = balance * risk_pct / 100.0
    notional = risk_amount / sl_dist_pct
    quantity = notional / entry

    # ---- بررسی واقع‌بینی هدف نسبت به هزینه‌ها ----
    round_trip_fee = fee_pct * 2 / 100.0  # کارمزد ورود + خروج
    net_tp_pct = (tp_dist_pct - round_trip_fee) * 100.0
    fee_warning = tp_dist_pct <= round_trip_fee * 3  # هدف نزدیک‌تر از ۳ برابر هزینه‌ها

    return TradePlan(
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=tp,
        risk_reward=risk_reward,
        sl_distance_pct=sl_dist_pct * 100.0,
        tp_distance_pct=tp_dist_pct * 100.0,
        up_prob=up_prob,
        down_prob=down_prob,
        tau_up=tau_up,
        tau_dn=tau_dn,
        adx=adx,
        risk_pct=risk_pct,
        risk_amount=risk_amount,
        notional=notional,
        quantity=quantity,
        fee_warning=fee_warning,
        net_tp_pct=net_tp_pct,
        validity_candles=validity_candles,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def plan_to_text(plan: TradePlan, extra_note: str = "") -> str:
    """قالب فارسی پلن برای تلگرام/ترمینال."""
    direction_fa = "LONG (خرید)" if plan.direction == "long" else "SHORT (فروش)"
    arrow = "🟢" if plan.direction == "long" else "🔴"

    lines = [
        f"{arrow} آلارم معاملاتی — {plan.symbol} ({plan.timeframe})",
        f"جهت: {direction_fa}",
        f"قیمت ورود: {_fmt(plan.entry)}",
        f"حد ضرر (SL): {_fmt(plan.sl)} ({plan.sl_distance_pct:+.2f}٪)",
        f"هدف (TP): {_fmt(plan.tp)} ({plan.tp_distance_pct:+.2f}٪)",
        f"نسبت پاداش/ریسک: {plan.risk_reward:.2f}R",
        "",
        f"احتمال مدل: up={plan.up_prob:.3f} | down={plan.down_prob:.3f} "
        f"(آستانه‌ها: up≥{plan.tau_up:.3f}, dn≥{plan.tau_dn:.3f})",
        f"رژیم بازار: ADX={plan.adx:.1f} (رونددار ✓)",
        "",
        f"ریسک: {plan.risk_pct:g}٪ حساب = {_fmt(plan.risk_amount)} USDT",
        f"نوتینال پیشنهادی: {_fmt(plan.notional, 0)} USDT",
        f"حجم: {plan.quantity:.6g} {plan.symbol.replace('USDT', '')}",
        f"سود خالص اگر هدف بخورد (بعد از کارمزد): {plan.net_tp_pct:+.2f}٪",
        f"اعتبار پلن: تا {plan.validity_candles} کندل دیگر — بعداً دوباره محاسبه شود",
    ]

    if plan.fee_warning:
        lines.append("")
        lines.append("⚠️ هشدار: هدف آن‌قدر به قیمت نزدیک است که کارمزد سهم بزرگی"
                     " از سود را می‌خورد؛ این معامله ارزش ریسک ندارد.")

    if extra_note:
        lines.append("")
        lines.append(extra_note)

    lines.append("")
    lines.append("⚠️ آلارم ابزار کمکی است، نه تضمین سود. حد ضرر همیشه ست شود.")
    return "\n".join(lines)