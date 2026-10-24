"""
orderbook.py
------------
محاسبه‌ی ویژگی‌های «دفتر سفارش» (Order Book) از اسنپ‌شات‌های عمق بازار.

این ماژول خالص است (بدون شبکه): ورودی، لیست bid/ask از دفتر سفارش است و
خروجی، ویژگی‌های فشرده‌ای که رفتار نقدینگی و فشار سفارش را توصیف می‌کنند:

  - spread_pct        : فاصله‌ی بهترین bid/ask نسبت به قیمت میانی
  - imbalance_{1,5,10,20} : عدم‌توازن حجم (bid - ask)/(bid + ask) در سطوح مختلف
                          (+1 = فشار خرید، -1 = فشار فروش)
  - bid_vol_1 / ask_vol_1  : حجم بهترین سطح هر طرف
  - bid_vol_20 / ask_vol_20: حجم کل ۲۰ سطح هر طرف
  - depth_ratio_20    : نسبت حجم کل bid به ask

چرا مهم است؟ دفتر سفارش تنها منبع اطلاعاتی است که در داده‌ی کندل/تِیکر
وجود ندارد: «نقدینگی در انتظار» در هر سطح قیمت. دیوارهای بزرگ سفارش،
شکار استاپ، و فشار پنهان خرید/فروش از این‌جا دیده می‌شوند.

⚠️ محدودیت صادقانه: بایننس تاریخچه‌ی دفتر سفارش را با API رایگان نمی‌دهد؛
   داده فقط از لحظه‌ی اجرای کلکتور به بعد جمع می‌شود (orderbook_stream.py).
   بنابراین این ویژگی‌ها را نمی‌توان روی داده‌ی تاریخی بک‌تست کرد — باید
   داده را «از حالا به بعد» جمع کنیم.

نحوه‌ی استفاده:
    from orderbook import compute_orderbook_features
    feats = compute_orderbook_features(bids, asks)   # bids: [[price, qty], ...]
"""

import logging
import math
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ستون‌های خروجی (برای هدر CSV و سازگاری بین کلکتور و مصرف‌کننده)
ORDERBOOK_FEATURE_COLUMNS = [
    "mid_price", "best_bid", "best_ask", "spread_pct",
    "imbalance_1", "imbalance_5", "imbalance_10", "imbalance_20",
    "bid_vol_1", "ask_vol_1", "bid_vol_20", "ask_vol_20", "depth_ratio_20",
]


def _parse_book(book):
    """تبدیل [[price_str, qty_str], ...] به آرایه‌های float جداگانه."""
    prices = np.array([float(r[0]) for r in book], dtype=float)
    qtys = np.array([float(r[1]) for r in book], dtype=float)
    return prices, qtys


def compute_orderbook_features(bids, asks, max_levels: int = 20) -> dict:
    """
    محاسبه‌ی ویژگی‌های دفتر سفارش از اسنپ‌شات.

    :param bids: لیست سطوح خرید مرتب‌شده از بهترین به پایین: [[price, qty], ...]
    :param asks: لیست سطوح فروش مرتب‌شده از بهترین به پایین: [[price, qty], ...]
    :param max_levels: حداکثر سطوحی که در محاسبات دیده می‌شوند

    :return: دیکشنری ویژگی‌ها (کلیدها = ORDERBOOK_FEATURE_COLUMNS)
    """
    if not bids or not asks:
        logger.warning("دفتر سفارش خالی است.")
        return {c: math.nan for c in ORDERBOOK_FEATURE_COLUMNS}

    bid_p, bid_q = _parse_book(bids)
    ask_p, ask_q = _parse_book(asks)

    n_levels = min(len(bid_p), len(ask_p), max_levels)
    if n_levels == 0:
        return {c: math.nan for c in ORDERBOOK_FEATURE_COLUMNS}

    bid_p, bid_q = bid_p[:n_levels], bid_q[:n_levels]
    ask_p, ask_q = ask_p[:n_levels], ask_q[:n_levels]

    best_bid = float(bid_p[0])
    best_ask = float(ask_p[0])
    mid = 0.5 * (best_bid + best_ask)
    spread_pct = (best_ask - best_bid) / mid if mid > 0 else math.nan

    def imbalance(levels: int) -> float:
        b = float(np.sum(bid_q[:levels]))
        a = float(np.sum(ask_q[:levels]))
        if (b + a) <= 0:
            return math.nan
        return (b - a) / (b + a)

    bid_vol_20 = float(np.sum(bid_q))
    ask_vol_20 = float(np.sum(ask_q))

    return {
        "mid_price": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_pct": spread_pct,
        "imbalance_1": imbalance(1),
        "imbalance_5": imbalance(5),
        "imbalance_10": imbalance(10),
        "imbalance_20": imbalance(20),
        "bid_vol_1": float(bid_q[0]),
        "ask_vol_1": float(ask_q[0]),
        "bid_vol_20": bid_vol_20,
        "ask_vol_20": ask_vol_20,
        "depth_ratio_20": (bid_vol_20 / ask_vol_20) if ask_vol_20 > 0 else math.nan,
    }


# ------------------------------------------------------------------
def load_latest_orderbook_features(symbol: str,
                                   ob_dir: str = "data_cache/orderbook",
                                   max_age_seconds: float = 300.0):
    """
    خواندن آخرین نمونه‌ی ویژگی‌های Order Book برای یک نماد.

    اگر فایل وجود نداشت، خالی بود، یا آخرین نمونه قدیمی‌تر از
    max_age_seconds بود → None برمی‌گرداند (برای اتصال به تفسیر AI).

    :param symbol: نام نماد (مثل BTCUSDT)
    :param ob_dir: پوشه‌ی CSVهای اردربوک
    :param max_age_seconds: حداکثر سن مجاز آخرین نمونه (پیش‌فرض ۵ دقیقه)
    """
    path = os.path.join(ob_dir, f"{symbol}_orderbook.csv")
    if not os.path.exists(path):
        logger.info(f"{symbol}: فایل اردربوک پیدا نشد: {path}")
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    except Exception as e:
        logger.warning(f"{symbol}: خطا در خواندن {path}: {e}")
        return None
    if df.empty:
        return None

    last = df.iloc[-1]
    if max_age_seconds is not None:
        try:
            age = (pd.Timestamp.now(tz="UTC").tz_localize(None)
                   - last["timestamp"]).total_seconds()
        except Exception:
            age = None
        if age is not None and age > max_age_seconds:
            logger.info(f"{symbol}: آخرین نمونه‌ی اردربوک {age:.0f} ثانیه قبل است — رد شد.")
            return None

    return {
        "timestamp": str(last["timestamp"]),
        "imbalance_1": float(last["imbalance_1"]),
        "imbalance_5": float(last["imbalance_5"]),
        "imbalance_10": float(last["imbalance_10"]),
        "imbalance_20": float(last["imbalance_20"]),
        "depth_ratio_20": float(last["depth_ratio_20"]),
        "spread_pct": float(last["spread_pct"]),
    }

