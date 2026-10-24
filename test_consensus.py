"""
test_consensus.py
-----------------
تست سریع منطق Ensemble Consensus — بدون نیاز به LM Studio.

منطق خالص (پارسر + ترکیب) را با چند مورد نمونه بررسی می‌کند تا مطمئن
شویم ترکیب دو نظر درست کار می‌کند.

نحوه‌ی اجرا:
    python test_consensus.py
"""

import sys

from strategies.ai_brain import AIBrain

PASS = 0
FAIL = 0


def check(name, got, expected):
    global PASS, FAIL
    ok = got == expected
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: انتظار {expected!r} بود، ولی {got!r} آمد")


def test_parse_verdict():
    print("\n— پارسر خط جمع‌بندی —")
    v = AIBrain._parse_verdict(
        "بازار صعودی به نظر می‌رسد.\n"
        "VERDICT_LINE: bias=bullish | confidence=0.8 | watch=break of 77500"
    )
    check("استخراج از پاسخ واقعی", v, {
        "bias": "bullish", "confidence": 0.8, "watch": "break of 77500"})

    v2 = AIBrain._parse_verdict("اصلاً خط جمع‌بندی ننوشتم.")
    check("بدون خط → None", v2, None)

    v3 = AIBrain._parse_verdict("")
    check("خالی → None", v3, None)

    v4 = AIBrain._parse_verdict(
        "VERDICT_LINE: bias=BEARISH | confidence=0.9 | watch=support 76000")
    check("حروف بزرگ", v4, {
        "bias": "bearish", "confidence": 0.9, "watch": "support 76000"})

    v5 = AIBrain._parse_verdict(
        "VERDICT_LINE: bias=weird | confidence=abc | watch=x")
    check("مقدار نامعتبر → bias=unknown", v5, {
        "bias": "unknown", "confidence": None, "watch": "x"})


def test_consensus():
    print("\n— منطق ترکیب —")
    bull_high = {"bias": "bullish", "confidence": 0.8, "watch": "a"}
    bull_low = {"bias": "bullish", "confidence": 0.5, "watch": "b"}
    bear = {"bias": "bearish", "confidence": 0.7, "watch": "c"}
    neutral = {"bias": "neutral", "confidence": 0.6, "watch": "d"}

    r1 = AIBrain.compute_consensus(bull_high, bull_high)
    check("هر دو صعودی با اطمینان بالا → توافق قوی",
          r1["level"], "strong_agreement")

    r2 = AIBrain.compute_consensus(bull_low, bull_low)
    check("هر دو صعودی با اطمینان کم → توافق معمولی",
          r2["level"], "agreement")

    r3 = AIBrain.compute_consensus(bull_high, bear)
    check("مخالف → اختلاف نظر", r3["level"], "disagreement")

    r4 = AIBrain.compute_consensus(neutral, bull_high)
    check("بی‌طرف + صعودی → یک‌جانبه", r4["level"], "one_neutral")

    r5 = AIBrain.compute_consensus(neutral, neutral)
    check("هر دو بی‌طرف", r5["level"], "neutral_agreement")

    r6 = AIBrain.compute_consensus(None, bull_high)
    check("پاسخ ناقص → در دسترس نیست", r6["level"], "unavailable")

    r7 = AIBrain.compute_consensus({"bias": "mixed", "confidence": 0.7, "watch": "x"},
                                   {"bias": "mixed", "confidence": 0.6, "watch": "y"})
    check("هر دو مختلط", r7["level"], "mixed_agreement")


def test_bias_in_summary():
    print("\n— متن‌های فارسی جمع‌بندی —")
    r = AIBrain.compute_consensus(
        {"bias": "bullish", "confidence": 0.75, "watch": "x"},
        {"bias": "bullish", "confidence": 0.70, "watch": "y"})
    assert "توافق قوی" in r["summary"], f"متن جمع‌بندی نادرست: {r['summary']}"
    assert "صعودی" in r["summary"]
    print("  ✅ جمع‌بندی فارسی شامل «توافق قوی» و «صعودی» است")


if __name__ == "__main__":
    print("🧪 تست منطق Ensemble Consensus")
    test_parse_verdict()
    test_consensus()
    test_bias_in_summary()
    print(f"\n📊 نتیجه: {PASS} موفق | {FAIL} ناموفق")
    sys.exit(1 if FAIL else 0)
