# 📋 فهرست کامل اندیکاتورها، استراتژی‌ها و الگوهای پروژه

_تاریخ: 2026-10-03 (به‌روزرسانی بعد از بک‌تست کامل) — تهیه‌شده بر اساس کد (نه حافظه)_

---

## ۱) اندیکاتورهای تکنیکال (ماژول `indicators/` — با pandas_ta)

### مومنتوم و اسیلاتورها (`indicators/momentum.py`)
| اندیکاتور | پارامتر |
|---|---|
| RSI | 14 |
| Stochastic RSI | 14, k=3, d=3 |
| CCI | 20 |
| Ultimate Oscillator | پیش‌فرض |
| Williams %R | 14 |
| Awesome Oscillator | 5/34 |
| TRIX | 18, signal 9 |
| KDJ | K/D/J |

### روند (`indicators/trend.py`)
| اندیکاتور | پارامتر |
|---|---|
| EMA | 9 / 21 / 50 / 200 |
| MACD | 12, 26, 9 |
| ADX / ADXR / DMP / DMN | 14 (پیاده‌سازی دستی با گارد [0,100]) |
| SuperTrend | 7, ضریب 3 |
| Parabolic SAR | 0.02, max 0.2 |
| ROC | 12 |
| ابر ایچیموکو | پیش‌فرض |

### نوسان (`indicators/volatility.py`)
| اندیکاتور | پارامتر |
|---|---|
| بولینگر (Bollinger) | 20, 2σ + Bandwidth + %B |
| کلتنر (Keltner) | 20, 2×ATR |
| ATR | 14 |
| دونچیان (Donchian) | 20/20 |

### حجم (`indicators/volume.py`)
| اندیکاتور | پارامتر |
|---|---|
| OBV | — |
| CMF چایکین | 20 (دستی با گارد تقسیم‌برصفر) |
| MFI | 14 |
| VWAP | + fallback دستی |
| VROC | 14 |

## ۲) ویژگی‌های اختصاصی فیوچرز (`features.py`)
- **فشار خرید**: `taker_buy_ratio` + میانگین ۲۴ (`taker_buy_ratio_ma_24`)
- **فاندینگ**: `funding_ma_32` (≈ ۸ ساعت) + `funding_change_32`
- **Open Interest**: `oi_change_4/24/96`، `oi_value_change_24`، `oi_to_volume`، `funding_oi`
  - ⚠️ در دیتاست نهایی به‌دلیل NaN بالا (≈96٪) **حذف می‌شوند** (داده تنها ~۱۲۰ روز اخیر دارد)
- **بازده‌ها**: `ret_1/4/24/96`، `volatility_24`، `range_pct`

## ۳) ویژگی‌های نرمال‌شده/ایستا (جایگزین پایه‌ی سطوح مطلق)
`ema9_dist`، `ema21_dist`، `ema50_dist`، `ema200_dist`، `ema9_21_cross`، `atr_pct`، `vol_regime`، `range_pos`، `vol_z`، `obv_z`، `vwap_dist`، `kc_pos`، `dc_pos`، `bbw_slope`، `macdh_norm`، `macdh_slope`، `rsi_slope`، `psar_dist_atr`، `st_dist`، `st_dir`، `hour_sin/cos`، `streak`، `pat_net_3/12`

## ۴) ویژگی‌های Smart Money (`smart_money.py` — با فلگ `--include-smart-money`)
- **CVD**: `cvd_z`، `cvd_change_4/24`، `cvd_divergence_24`
- **پروفایل حجم**: `price_to_poc`، `pos_in_va`، `va_width_pct`، `poc_concentration`
- **وایکاف**: `spring`، `upthrust`، `pos_in_range`، `spring_count_24`، `upthrust_count_24`
- **فشار سفارش**: `buy_pressure_ma/z/momentum`

## ۵) الگوهای کندلی (`strategies/patterns.py`)
مجموعه الگوهای صعودی/نزولی با شرط‌های شکلی (bull/bear/shape) → در دیتاست به‌صورت جمع‌بندی `pat_net_3/12` وارد می‌شود.

## ۶) الگوهای کلاسیک نموداری (`strategies/patterns.py` — با فلگ `--include-chart-patterns`)
- دو سقف / دو کف (Double Top/Bottom)، سر و شانه (Head & Shoulders)، مثلث‌ها (Triangles)، پرچم/پننت (Flag/Pennant)، گارتلی هارمونیک (Harmonic Gartley)
- هر الگو فقط در لحظه‌ی «تکمیل» خودش به سیگنال تبدیل می‌شود (بدون look-ahead) و به‌صورت شمارش ایستا وارد مدل می‌شود: `chart_pat_net_24`، `chart_pat_net_96`، `chart_pat_count_96`

## ۷) لایه رژیم و امتیازدهی (`scoring.py` + `market_profile.py` — با فلگ `--include-scores`)
- فیلتر رژیم: ADX، انبساط نوسان، هم‌سویی روند (RegimeFilter)
- تشخیص رژیم بازار: strong_trend / weak_trend / ranging / high_volatility (MarketRegime)
- امتیاز سیگنال: روند + مومنتوم + نوسان + حجم + پاداش هم‌نوایی (SignalScorer)
- ستون‌های ورودی مدل: `score_0_100`، `score_raw`، `score_dir`، `score_conf`، `score_trend/momentum/volatility/volume/confluence`، `regime_*` (one-hot)، `regime_strength`، `trend_dir_num`، `filter_passed`

## ۸) پلن معاملاتی (`trade_plan.py` — در بکتست Walk-Forward فعال است)
ورود در بسته‌شدن کندل سیگنال | **SL = ATR × 1.5** | **TP = 2R** | خروج زمانی (max_hold=horizon) | اولویت SL در برخورد هم‌زمان | فیلتر ورود ADX≥30 | کارمزد تیکر فیوچرز بایننس = 0.05٪/طرف

## ۹) لایه هوش مصنوعی
- **هسته ML (بکتست):** دو مدل دوجهته **LightGBM** (up/down) + quantile threshold روی احتمالات (0.75/0.90)
- **لایه LLM (`ai_brain.py`):** LM Studio محلی (OpenAI-سازگار) — پیش‌فرض `meta-llama-3-8b-instruct` و `mistral-7b-instruct-v0.3` + کنسنسوس (محصول)

## ۱۰) ماژول‌های فرعی/آزمایشی
- دفتر سفارش (`orderbook.py`، `analyze_orderbook.py`) — بدون داده تاریخی، بکتست‌پذیر نیست
- برچسب‌ها: `fwd_ret_{h}` با آستانه 0.2٪ → `target_up` / `target_down`

---

# ✅ چه چیزی واقعاً در بکتست فعال است؟ (پیکربندی کامل)

> **دستور بکتست کامل** (هر ۴ جزء هم‌زمان):
> `build_dataset.py --interval 1h --horizons 4 16 --include-smart-money --include-chart-patterns --include-scores --include-absolute-levels`

**✅ داخل دیتاست/بکتست هستند (با پیکربندی کامل):**
- همه‌ی اندیکاتورهای تکنیکال (بخش ۱) + نسخه‌های نرمال‌شده‌ی آن‌ها (بخش ۳)
- ویژگی‌های فیوچرز (funderate، فشار خرید) + بازده‌ها
- الگوهای کندلی (`pat_net_3/12`) + الگوهای نموداری (`chart_pat_net_24/96`، `chart_pat_count_96`)
- لایه‌ی رژیم/امتیازدهی (۱۶ ستون `score_*` / `regime_*` / `filter_passed`)
- Smart Money (۱۶ ستون CVD/وایکاف/فشار سفارش)
- **سطوح مطلق قیمت** (EMA_9/21/50/200، BB_*، KC_*، DC_*، VWAP، OBV، ایچیموکو، PSAR، SuperTrend) — با `--include-absolute-levels` نگه داشته می‌شن
- فیلتر ADX≥30، قوانین SL=ATR×1.5 / TP=2R، فاندینگ و کارمزد در شبیه‌سازی
- برچسب‌ها: `target_up` / `target_down` + دو مدل LightGBM + quantile 0.90 + کنترل شافل (`--shuffle-labels`)

**❌ در بکتست نیستند:**
- ستون‌های OI (به‌دلیل NaN≈96٪ خودکار حذف می‌شن)
- مدل‌های LLM (لایه محصول — مستقل از بکتست)
- دفتر سفارش (بدون داده تاریخی)

---

# 📊 نتایج نهایی Walk-Forward — پیکربندی کامل (1h، q0.90، ۱۶۰ پنجره، ۲۰۲۰-۰۳ تا ۲۰۲۶-۱۰)

| پیکربندی | میانگین Expectancy | پنجره‌های مثبت | معاملات |
|---|---|---|---|
| fee = 0 (واقعی) | **+0.0827R** | 100/160 (62٪) | 4,662 |
| fee = 0.02٪ (میکر) | +0.0590R | 90/160 (56٪) | 4,662 |
| fee = 0.05٪ (تیکر) | +0.0235R | 80/160 (50٪) | 4,662 |
| fee = 0 (شافل — کنترل) | **+0.0003R** | 84/160 (52٪) | 6,007 |

**تفکیک سال‌به‌سال (واقعی fee=0):** 2020: +0.068R | 2021: +0.166R | 2022: +0.023R | 2023: +0.068R | 2024: +0.143R | 2025: +0.031R | 2026: +0.062R — **در هر ۷ سال مثبت**

**نتیجه‌ی کلیدی:**
- edge خالص (واقعی − شافل) ≈ **+0.082R** با شافل ≈ صفر → edge واقعی است نه مصنوع انتخاب.
- تنها سال ضعیف: ۲۰۲۲ (خرس) که واقعی < شافل به‌اندازه‌ی کوچیک.
- نقطه‌ی سربه‌سر کارمزد ≈ 0.07٪/طرف (بدون مدل‌سازی slippage/spread)؛ با کارمزد میکر در همه‌ی سال‌ها ≥ صفر، با تیکر فقط در سال‌های روندی (۲۰۲۱/۲۰۲۴) سودآور.
- ۴ جزء هم‌زمان بهترین و پایدارترین نتیجه‌ی کل مسیر: ۷ سال پایداری + شافل صفر + ۶۲٪ پنجره‌های مثبت.
- ⚠️ بک‌تست (هرچقدر ساخت‌یافته) تضمین آینده نیست؛ قبل معامله‌ی واقعی paper-trading لازم است.