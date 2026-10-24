"""
extract_pdf_text.py
-------------------
استخراج متن از PDFهای پوشه‌ی Learning/.

ابتدا سعی می‌کند از کتابخانه‌ی pypdf استفاده کند (دقیق‌ترین روش — نصب:
    pip install pypdf
) و اگر نصب نبود، به روش دستی با zlib (کتابخانه‌ی استاندارد) برمی‌گردد.

⚠️ محدودیت صادقانه: PDFهای اسکن‌شده (تصویری) متن قابل استخراج ندارند و نیاز به
   OCR دارند — اگر خروجی خالی/خیلی کم بود، یعنی فایل تصویری است.

کاربرد:
    python extract_pdf_text.py
"""

import re
import sys
import zlib
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parent / "Learning"

# نام‌های خروجی کوتاه (برای جلوگیری از خطای طول مسیر در ویندوز)
OUTPUT_NAMES = {
    "evidence": "aronson_evidence_based.txt",
    "trading": "harris_market_microstructure.txt",
}


def _short_output_name(pdf_path: Path) -> Path:
    name = pdf_path.stem.lower()
    for key, out in OUTPUT_NAMES.items():
        if key in name:
            return LEARNING_DIR / out
    return LEARNING_DIR / (pdf_path.stem[:20] + ".txt")


# ----------------------------------------------------------------------
# روش ۱: pypdf (دقیق)
# ----------------------------------------------------------------------
def extract_with_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return None

    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)


# ----------------------------------------------------------------------
# روش ۲: دستی با zlib (پشتیبان)
# ----------------------------------------------------------------------
_STR_RE = re.compile(rb"\((?:[^()\\]|\\.)*\)")
_ESCAPE_MAP = {
    b"n": "\n", b"r": "\r", b"t": "\t", b"b": "\b", b"f": "\f",
    b"(": "(", b")": ")", b"\\": "\\",
}


def _pdf_string_to_text(s: bytes) -> str:
    inner = s[1:-1]
    out = []
    i = 0
    n = len(inner)
    while i < n:
        c = inner[i:i + 1]
        if c == b"\\" and i + 1 < n:
            nxt = inner[i + 1:i + 2]
            if nxt in _ESCAPE_MAP:
                out.append(_ESCAPE_MAP[nxt])
                i += 2
            elif nxt.isdigit():
                octs = inner[i + 1:i + 4]
                try:
                    out.append(chr(int(octs, 8)))
                except ValueError:
                    out.append("?")
                i += 4
            else:
                i += 2
        else:
            out.append(chr(c[0]))
            i += 1
    return "".join(out)


def _ascii85_decode(data: bytes) -> bytes:
    """رمزگشایی ASCII85 (~> ... <~) اگر جریان این‌طور کد شده باشد."""
    if not data.startswith(b"<~"):
        return data
    try:
        import base64
        return base64.a85decode(data[2:-2].rstrip(b"~>"), adobe=False)
    except Exception:
        return data


def extract_manual(pdf_path: Path) -> str:
    data = pdf_path.read_bytes()
    streams = []
    for m in re.finditer(rb"stream\r?\n", data):
        start = m.end()
        end = data.find(b"endstream", start)
        if end == -1:
            continue
        streams.append(data[start:end].rstrip(b"\r\n"))

    text_chunks = []
    for raw in streams:
        raw = _ascii85_decode(raw)
        decompressed = None
        for candidate in (raw, raw[1:], raw[2:]):
            try:
                decompressed = zlib.decompress(candidate)
                break
            except Exception:
                continue
        if decompressed is None:
            continue
        for m in _STR_RE.finditer(decompressed):
            text_chunks.append(_pdf_string_to_text(m.group(0)))

    return "".join(text_chunks)


# ----------------------------------------------------------------------
def _clean_text(body: str) -> str:
    lines = [ln.strip() for ln in body.splitlines()]
    seen = set()
    unique = []
    for ln in lines:
        if ln and ln not in seen:
            seen.add(ln)
            unique.append(ln)
    return "\n".join(unique)


def main():
    pdfs = sorted(LEARNING_DIR.glob("*.pdf"))
    if not pdfs:
        print("❌ فایل PDFای در پوشه‌ی Learning/ پیدا نشد.")
        return

    for pdf in pdfs:
        out = _short_output_name(pdf)
        print(f"📄 پردازش: {pdf.name}")
        print(f"   خروجی: {out.name}")

        text = extract_with_pypdf(pdf)
        method = "pypdf"
        if text is None:
            print("   (pypdf نصب نیست — روش دستی zlib)")
            text = extract_manual(pdf)
            method = "zlib"

        cleaned = _clean_text(text)
        out.write_text(cleaned, encoding="utf-8")
        print(f"   روش: {method} | کاراکتر استخراج‌شده: {len(cleaned):,}")

        # نمونه‌ی کوچک برای بررسی کیفیت
        sample = " ".join(cleaned.split())[:300]
        if sample:
            print(f"   نمونه: {sample}...")
        else:
            print("   ⚠️ متنی استخراج نشد — احتمالاً PDF اسکن‌شده/تصویری است (نیاز به OCR).")
        print()


if __name__ == "__main__":
    main()
