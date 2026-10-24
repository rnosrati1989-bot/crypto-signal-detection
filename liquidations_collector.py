"""
liquidations_collector.py
-------------------------
جمع‌آوری مداومِ «لیکوییدیشن» فیوچرز و ذخیره‌ی محلی — رایگان و بدون کلید API.

چرا این فایل لازم است؟
  صرافی‌ها فید *زنده‌ی* لیکوییدیشن را رایگان می‌دهند، ولی **تاریخچه** نه
  (بررسی‌شده در ۲۰۲۶-۱۰):
      • Binance  /fapi/v1/allForceOrders          → بازنشسته شده (HTTP 404)
      • Bybit    /v5/market/liquidation            → وجود ندارد (HTTP 404)
      • OKX      /api/v5/public/liquidation-orders → ✅ رایگان (فقط پنجره‌ی اخیر)
  پس تنها راه رایگانِ ساختن «ویژگی لیکوییدیشن» برای مدل این است که از همین
  حالا خودمان جمع کنیم. هر روزی که دیر شروع شود، یک روز داده از دست می‌رود.

منابع داده
  • okx  (پیش‌فرض، REST، بدون کلید):
      GET /api/v5/public/liquidation-orders?instType=SWAP&uly=BTC-USDT&state=filled
      - پنجره‌ی نگهداری کوتاه است (≈۱ ساعت) و صفحه‌بندی به عقب با پارامتر
        after=<ms> کار می‌کند → با polling هر ۲۰-۳۰ ثانیه و چند صفحه، حتی در
        بازار شلوغ هم چیزی از دست نمی‌رود.
      - تبدیل تعداد قرارداد به دارایی: qty = sz × ctVal × ctMult
        (ctVal از public/instruments؛ مثلاً BTC-USDT-SWAP → 0.01 BTC)
  • binance-ws (اختیاری، بزرگ‌ترین صرافی): WebSocket «!forceOrder@arr»
      نیازمند: pip install websocket-client
      payload: o.s=symbol, o.S=SELL(long liquidated)/BUY(short), o.ap=price, o.z=filled qty

خروجی
  data_cache/liquidations/{SYMBOL}.csv
      timestamp,venue,side,price,qty,notional
  side: SELL = لیکویید شدن لانگ | BUY = لیکویید شدن شورت
  notional: ارزش دلاری سفارش

⚠️ صداقت: این مخزن فقط از لحظه‌ی اولین اجرا داده دارد. برای آموزش مدل باید با
   --tail-days پنجره‌ای انتخاب شود که با دوره‌ی جمع‌آوری هم‌پوشانی داشته باشد.

اجرا (از داخل پوشه‌ی پروژه):
    venv\\Scripts\\python.exe liquidations_collector.py                # حلقه‌ی مداوم
    venv\\Scripts\\python.exe liquidations_collector.py --once         # یک پاس (تست)
    venv\\Scripts\\python.exe liquidations_collector.py --source binance-ws
"""

import argparse
import csv
import logging
import os
import signal
import sys
import time

# غیرفعال کردن پراکسی برای ارتباط محلی (سازگار با بقیه‌ی پروژه)
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import pandas as pd
import requests

from config.settings import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LiqCollector")

# --- تنظیمات -----------------------------------------------------------
LIQ_DIR = os.path.join("data_cache", "liquidations")
COLUMNS = ["timestamp", "venue", "side", "price", "qty", "notional"]

OKX_BASE = "https://www.okx.com"
OKX_LIQ_URL = f"{OKX_BASE}/api/v5/public/liquidation-orders"
OKX_INSTR_URL = f"{OKX_BASE}/api/v5/public/instruments"

REQUEST_TIMEOUT = 20          # ثانیه
PAGE_LIMIT = 100              # سقف OKX
PAGE_SLEEP = 0.25             # فاصله‌ی بین صفحه‌ها (مودبانه با ریت‌لیمیت)
HEARTBEAT_EVERY = 10          # هر چند پاس، یک خط وضعیت چاپ شود

# توقف نرم با Ctrl+C
_STOP = False


def _sigint(_signum, _frame):
    global _STOP
    _STOP = True
    logger.info("درخواست توقف دریافت شد — ذخیره‌سازی و خروج...")


def okx_inst_id(symbol: str) -> str:
    """BTCUSDT → BTC-USDT-SWAP"""
    base = symbol.upper()
    if base.endswith("USDT"):
        base = base[:-4]
    return f"{base}-USDT-SWAP"


def okx_uly(symbol: str) -> str:
    """BTCUSDT → BTC-USDT"""
    base = symbol.upper()
    if base.endswith("USDT"):
        base = base[:-4]
    return f"{base}-USDT"


def fetch_okx_multiplier(symbol: str):
    """
    ضریب تبدیل قرارداد به دارایی پایه (ctVal × ctMult) از public/instruments.
    اگر نماد روی OKX نباشد، None برمی‌گرداند (نماد رد می‌شود، خطا نمی‌دهد).
    """
    try:
        r = requests.get(OKX_INSTR_URL,
                         params={"instType": "SWAP", "instId": okx_inst_id(symbol)},
                         timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json().get("data") or []
        if not data:
            return None
        d = data[0]
        return {"ctVal": float(d["ctVal"]),
                "ctMult": float(d.get("ctMult") or 1.0),
                "ctValCcy": d.get("ctValCcy", "")}
    except Exception as e:
        logger.warning(f"خواندن ضریب قرارداد {symbol} ناموفق بود: {e}")
        return None


def _row_ts_ms(row: dict) -> int:
    return int(row.get("ts") or row.get("time") or 0)


def fetch_okx_page(symbol: str, after_ms=None, limit: int = PAGE_LIMIT):
    """یک صفحه از لیکوییدیشن‌های اخیر. after_ms = «قدیمی‌تر از این زمان»."""
    params = {
        "instType": "SWAP",
        "uly": okx_uly(symbol),
        "state": "filled",
        "limit": str(limit),
    }
    if after_ms:
        params["after"] = str(int(after_ms))
    try:
        r = requests.get(OKX_LIQ_URL, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            logger.warning(f"OKX {symbol}: HTTP {r.status_code}")
            return None
        j = r.json()
        if j.get("code") != "0":
            logger.warning(f"OKX {symbol}: {j.get('code')} {j.get('msg')}")
            return None
        rows = []
        for grp in j.get("data") or []:
            rows.extend(grp.get("details") or [])
        return rows
    except Exception as e:
        logger.warning(f"OKX {symbol}: خطای اتصال — {e}")
        return None


def okx_rows_to_records(symbol: str, rows, mult, seen, floor_ms):
    """
    تبدیل ردیف‌های خام OKX به رکوردهای قابل‌ذخیره (بدون تکرار).
    کلید یکتایی: (زمان، سمت، قیمت، مقدار، سرایز) — چون OKX شناسه‌ی سفارش نمی‌دهد.
    """
    out = []
    for row in rows:
        try:
            ts_ms = _row_ts_ms(row)
            if ts_ms <= 0 or ts_ms < floor_ms:
                continue
            side = str(row.get("side", "")).upper()          # SELL/BUY
            price = float(row.get("bkPx") or 0.0)
            sz = float(row.get("sz") or 0.0)
            if price <= 0.0 or sz <= 0.0:
                continue
            qty = sz * mult["ctVal"] * mult["ctMult"]        # قرارداد → دارایی پایه
            key = (ts_ms, side, round(price, 8), round(qty, 12))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True)
                               .tz_localize(None).isoformat(sep=" "),
                "venue": "okx",
                "side": side,
                "price": price,
                "qty": qty,
                "notional": qty * price,
            })
        except Exception:
            continue
    return out


def binance_rows_to_records(symbols, payload, seen):
    """تبدیل payload وب‌سوکت بایننس به رکوردها (فقط نمادهای درخواستی)."""
    out = []
    try:
        o = payload.get("o") or {}
        sym = str(o.get("s", "")).upper()
        if sym not in symbols:
            return out
        side = str(o.get("S", "")).upper()      # SELL = لانگ لیکویید، BUY = شورت لیکویید
        price = float(o.get("ap") or o.get("p") or 0.0)
        qty = float(o.get("z") or o.get("q") or 0.0)
        ts_ms = int(o.get("T") or payload.get("E") or 0)
        if price <= 0.0 or qty <= 0.0 or ts_ms <= 0:
            return out
        key = (ts_ms, side, round(price, 8), round(qty, 12))
        if key in seen:
            return out
        seen.add(key)
        out.append({
            "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True)
                           .tz_localize(None).isoformat(sep=" "),
            "venue": "binance",
            "side": side,
            "price": price,
            "qty": qty,
            "notional": qty * price,
        })
    except Exception:
        pass
    return out


class Store:
    """نویسنده‌ی append-only برای یک نماد + حافظه‌ی تکرارگیری."""

    def __init__(self, symbol: str, out_dir: str = LIQ_DIR):
        self.symbol = symbol.upper()
        self.dir = out_dir
        self.path = os.path.join(out_dir, f"{self.symbol}.csv")
        self.seen = set()
        self.fh = None
        self.writer = None
        self.added_total = 0

    def open(self) -> int:
        """فایل را باز می‌کند و حافظه‌ی تکرارگیری را از ردیف‌های قبلی پر می‌کند."""
        os.makedirs(self.dir, exist_ok=True)
        is_new = not os.path.exists(self.path)
        if not is_new:
            try:
                tail = pd.read_csv(self.path, on_bad_lines="skip").tail(50000)
                for _, r in tail.iterrows():
                    ts = pd.Timestamp(r["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    self.seen.add((int(ts.timestamp() * 1000), str(r["side"]).upper(),
                                   round(float(r["price"]), 8), round(float(r["qty"]), 12)))
            except Exception as e:
                logger.warning(f"{self.symbol}: پرکردن حافظه‌ی تکرارگیری ناموفق بود: {e}")
        self.fh = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, fieldnames=COLUMNS)
        if is_new:
            self.writer.writeheader()
            self.flush()
        return len(self.seen)

    def add(self, records) -> int:
        for rec in records:
            self.writer.writerow({k: rec.get(k, "") for k in COLUMNS})
        self.added_total += len(records)
        return len(records)

    def flush(self):
        if self.fh:
            self.fh.flush()
            try:
                os.fsync(self.fh.fileno())
            except OSError:
                pass

    def close(self):
        self.flush()
        if self.fh:
            self.fh.close()
            self.fh = None


def collect_okx_once(stores, mults, pages, floor_ms):
    """یک پاس کامل واکشی از OKX برای همه‌ی نمادها."""
    total = 0
    now_ms = int(time.time() * 1000)
    floor = floor_ms if floor_ms > 0 else (now_ms - 2 * 3600 * 1000)   # حداکثر ۲ ساعت عقب
    for symbol, store in stores.items():
        mult = mults.get(symbol)
        if mult is None:
            continue
        rows, after = [], None
        for _ in range(int(pages)):
            page = fetch_okx_page(symbol, after_ms=after)
            if not page:
                break
            rows.extend(page)
            oldest = min(_row_ts_ms(r) for r in page)
            if oldest <= floor or len(page) < PAGE_LIMIT:
                break
            if after is not None and oldest >= after:      # جلوگیری از حلقه‌ی بی‌پایان
                break
            after = oldest
            time.sleep(PAGE_SLEEP)
        recs = okx_rows_to_records(symbol, rows, mult, store.seen, floor)
        total += store.add(recs)
    for store in stores.values():
        store.flush()
    return total


def collect_binance_ws(stores, symbols):
    """حالت اختیاری: وب‌سوکت بایننس (!forceOrder@arr) — نیازمند websocket-client."""
    try:
        import websocket  # type: ignore  # از بسته‌ی websocket-client
    except ImportError:
        logger.error("برای منبع binance-ws نصب کنید: pip install websocket-client")
        return 1

    url = "wss://fstream.binance.com/ws/!forceOrder@arr"

    def on_message(_ws, message):
        try:
            import json
            payload = json.loads(message)
            o = payload.get("o") or {}
            sym = str(o.get("s", "")).upper()
            store = stores.get(sym)
            if store is None:
                return
            recs = binance_rows_to_records({sym}, payload, store.seen)
            if recs:
                store.add(recs)
                store.flush()  # جریان زنده است → فوری روی دیسک نوشته شود
                logger.debug(f"بایننس {sym}: {len(recs)} رکورد ذخیره شد.")
        except Exception as e:
            logger.debug(f"پیام بایننس پردازش نشد: {e}")

    logger.info(f"اتصال به {url} ...")
    try:
        ws = websocket.WebSocketApp(url, on_message=on_message)
        ws.run_forever(ping_interval=30, ping_timeout=10)
    except Exception as e:
        logger.error(f"وب‌سوکت بایننس قطع شد: {e}")
        return 1
    return 0


def main():
    global _STOP
    parser = argparse.ArgumentParser(description="جمع‌آوری لیکوییدیشن فیوچرز (رایگان)")
    parser.add_argument("--symbols", nargs="+", default=Config.DEFAULT_SYMBOLS,
                        help=f"نمادها (پیش‌فرض: {','.join(Config.DEFAULT_SYMBOLS)})")
    parser.add_argument("--source", choices=["okx", "binance-ws"], default="okx",
                        help="منبع داده (پیش‌فرض okx؛ binance-ws نیازمند websocket-client)")
    parser.add_argument("--poll-seconds", type=float, default=25.0,
                        help="فاصله‌ی هر پاس (پیش‌فرض ۲۵ ثانیه؛ پنجره‌ی OKX ≈۱ ساعت است)")
    parser.add_argument("--pages", type=int, default=4,
                        help="حداکثر صفحه در هر پاس (هر صفحه ۱۰۰ رکورد)")
    parser.add_argument("--hours-back", type=float, default=2.0,
                        help="در اجرای اول تا این حد عقب برو (پیش‌فرض ۲ ساعت)")
    parser.add_argument("--out-dir", default=LIQ_DIR)
    parser.add_argument("--once", action="store_true",
                        help="فقط یک پاس اجرا کن و خارج شو (برای تست)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    signal.signal(signal.SIGINT, _sigint)

    symbols = [s.upper() for s in args.symbols]
    stores = {s: Store(s, args.out_dir) for s in symbols}

    print("=" * 78)
    print(f"💥 جمع‌آور لیکوییدیشن — منبع: {args.source} | نمادها: {', '.join(symbols)}")
    print(f"   مقصد: {os.path.abspath(args.out_dir)}")
    print("=" * 78)

    for symbol, store in stores.items():
        seeded = store.open()
        print(f"   • {symbol}: {store.path} (رکوردهای قبلی برای تکرارگیری: {seeded})")

    if args.source == "binance-ws":
        return collect_binance_ws(stores, set(symbols))

    # OKX: ضریب قرارداد هر نماد یک‌بار خوانده می‌شود
    mults = {}
    for symbol in symbols:
        mult = fetch_okx_multiplier(symbol)
        if mult is None:
            print(f"   ⚠️ {symbol}: روی OKX فیوچرز پیدا نشد — رد شد.")
        else:
            mults[symbol] = mult
            print(f"   • {symbol}: ctVal={mult['ctVal']} ({mult['ctValCcy']}) × {mult['ctMult']}")

    if not mults:
        print("\n❌ هیچ نمادی روی OKX قابل استفاده نبود.")
        return 1

    floor_ms = int((time.time() - args.hours_back * 3600) * 1000)
    passes = 0
    while not _STOP:
        passes += 1
        added = collect_okx_once(stores, mults, args.pages, floor_ms)
        if added:
            logger.info(f"پاس {passes}: {added} لیکوییدیشن جدید ذخیره شد.")
        elif passes % HEARTBEAT_EVERY == 1:
            logger.info(f"پاس {passes}: چیز جدیدی نبود "
                        f"(مجموع ذخیره‌شده: {sum(s.added_total for s in stores.values())}).")
        if args.once:
            break
        # بعد از اولین پاس، کفِ زمانی را جلو می‌کشیم (فقط داده‌ی تازه)
        floor_ms = int((time.time() - max(600.0, args.poll_seconds * 4)) * 1000)
        # انتظار قابل‌توقف
        slept = 0.0
        while slept < args.poll_seconds and not _STOP:
            time.sleep(min(1.0, args.poll_seconds - slept))
            slept += 1.0

    for store in stores.values():
        store.close()
    print("=" * 78)
    for symbol, store in stores.items():
        print(f"   ✅ {symbol}: مجموع این اجرا {store.added_total} رکورد → {store.path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
