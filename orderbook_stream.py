"""
orderbook_stream.py
-------------------
کلکتور زنده‌ی «دفتر سفارش» (Order Book) فیوچرز بایننس از طریق WebSocket.

چرا؟ داده‌ی دفتر سفارش (نقدینگی در انتظار در هر سطح قیمت) تنها منبع
اطلاعاتی است که در کندل/تِیکر وجود ندارد و بایننس تاریخچه‌ی آن را با
API رایگان نمی‌دهد. این اسکریپت از «همین لحظه به بعد» داده را جمع می‌کند
تا بعداً به عنوان ویژگی‌های جدید (imbalance، عمق، spread) استفاده شود.

نحوه‌ی کار:
  - اتصال به جریان ترکیبی `@depth20@100ms` (۲۰ سطح برتر، به‌روزرسانی هر ۱۰۰ms)
  - نگه‌داشتن آخرین اسنپ‌شات هر نماد
  - هر `--interval` ثانیه، ویژگی‌ها را محاسبه و به CSV ضمیمه می‌کند

نحوه‌ی اجرا:
    python orderbook_stream.py --duration 60          # تست ۶۰ ثانیه‌ای (خودکار متوقف می‌شود)
    python orderbook_stream.py                         # اجرای بی‌وقفه (Ctrl+C برای توقف)
    python orderbook_stream.py --interval 300          # نمونه‌برداری هر ۵ دقیقه

خروجی: data_cache/orderbook/<SYMBOL>_orderbook.csv

⚠️ صادقانه: این داده «از حالا به بعد» جمع می‌شود؛ بک‌تست تاریخی روی آن
   ممکن نیست. بعد از چند روز/هفته جمع‌آوری، می‌توان با کندل‌ها join کرد.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import time

from orderbook import compute_orderbook_features, ORDERBOOK_FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("OrderBookStream")

WS_BASE = "wss://fstream.binance.com"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_LEVELS = 20
DEFAULT_INTERVAL = 60          # نمونه‌برداری هر ۶۰ ثانیه
BOOK_MAX_AGE = 10.0            # اگر اسنپ‌شات قدیمی‌تر از این بود، نمونه نگیر
RECONNECT_BASE_DELAY = 2.0     # بک‌آف اولیه برای اتصال مجدد
RECONNECT_MAX_DELAY = 60.0


def _get_connect():
    """اتصال websockets با سازگاری بین نسخه‌های مختلف کتابخانه."""
    try:
        from websockets.asyncio.client import connect as _c  # websockets >= 14
        return _c
    except ImportError:
        try:
            from websockets.client import connect as _c  # websockets 10-13
            return _c
        except ImportError:
            import websockets
            return websockets.connect  # خیلی قدیمی


async def _sample(books, out_files, interval, duration, start_time, stop_event):
    """وظیفه‌ی نمونه‌برداری: هر interval ثانیه، ویژگی‌ها را بنویس.
    وقتی duration تمام شد، stop_event را فعال می‌کند تا کل کلکتور بسته شود."""
    sample_count = 0
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        if stop_event.is_set():
            break
        now = time.time()
        for symbol in list(books.keys()):
            entry = books.get(symbol)
            if entry is None:
                continue
            bids, asks, ts = entry
            age = now - ts
            if age > BOOK_MAX_AGE:
                logger.debug(f"{symbol}: اسنپ‌شات قدیمی است ({age:.1f}s) — رد شد.")
                continue
            feats = compute_orderbook_features(bids, asks)
            feats["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
            _append_csv(out_files[symbol], feats)
            sample_count += 1
        logger.info(f"📊 نمونه‌گیری: {sample_count} ردیف کل | "
                    f"نمادهای فعال: {sum(1 for s in books.values() if s)}")

        if duration is not None and time.time() - start_time >= duration:
            logger.info("⏱️ مدت‌زمان تست تمام شد — در حال بستن کلکتور...")
            stop_event.set()
            return


def _append_csv(path, row: dict):
    """افزودن یک ردیف به CSV (با ساخت هدر در اولین نوشتن)."""
    header = ["timestamp"] + ORDERBOOK_FEATURE_COLUMNS
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _streams_url(symbols, levels):
    streams = "/".join(f"{s.lower()}@depth{levels}@100ms" for s in symbols)
    return f"{WS_BASE}/stream?streams={streams}"


async def _run(symbols, levels, interval, duration, outdir):
    os.makedirs(outdir, exist_ok=True)
    books = {s: None for s in symbols}
    out_files = {s: os.path.join(outdir, f"{s}_orderbook.csv") for s in symbols}
    stop_event = asyncio.Event()

    connect = _get_connect()
    url = _streams_url(symbols, levels)
    logger.info(f"🔌 اتصال به: {url}")

    start_time = time.time()
    sampler = asyncio.create_task(
        _sample(books, out_files, interval, duration, start_time, stop_event))

    delay = RECONNECT_BASE_DELAY
    try:
        while not stop_event.is_set():
            try:
                async with connect(url, ping_interval=20, ping_timeout=20) as ws:
                    logger.info("✅ متصل شد. منتظر پیام‌های depth...")
                    delay = RECONNECT_BASE_DELAY
                    async for raw in ws:
                        if stop_event.is_set():
                            break  # پایان مدت‌زمان تست — خروج تمیز
                        msg = json.loads(raw)
                        stream = msg.get("stream", "")
                        data = msg.get("data", {})
                        symbol = data.get("s") or stream.split("@")[0].upper()
                        if symbol not in books:
                            continue
                        books[symbol] = (data.get("b", []), data.get("a", []), time.time())
            except asyncio.CancelledError:
                logger.info("🛑 توقف درخواستی (Ctrl+C).")
                break
            except Exception as e:
                if stop_event.is_set():
                    break
                logger.error(f"⚠️ خطای اتصال: {e} — تلاش مجدد در {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY)
    finally:
        if not sampler.done():
            sampler.cancel()
        try:
            await sampler
        except asyncio.CancelledError:
            pass
        logger.info("✅ کلکتور متوقف شد.")


def main():
    parser = argparse.ArgumentParser(description="کلکتور زنده‌ی Order Book بایننس فیوچرز")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS,
                        help="عمق سطوح (5/10/20)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="فاصله‌ی نمونه‌برداری به ثانیه (پیش‌فرض 60)")
    parser.add_argument("--duration", type=int, default=None,
                        help="توقف خودکار بعد از N ثانیه (برای تست)")
    parser.add_argument("--outdir", default="data_cache/orderbook")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]
    try:
        asyncio.run(_run(symbols, args.levels, args.interval,
                         args.duration, args.outdir))
    except KeyboardInterrupt:
        logger.info("🛑 Ctrl+C دریافت شد.")


if __name__ == "__main__":
    main()
