# 📊 Crypto Technical Analysis & Decision-Support Dashboard

یک ابزار تحلیل تکنیکال برای بازار ارز دیجیتال که داده‌ی زنده‌ی Binance رو با
اندیکاتورهای فنی استاندارد، تشخیص الگو، و تحلیل متنی دو مدل هوش مصنوعی محلی
(Llama 3 + Mistral) ترکیب می‌کنه.

> 👷 **به‌روزرسانی ۲۰۲۶-۱۰-۰۳ — وضعیت edge:** بعد از بک‌تست کامل (هر ۴ جزء
> feature: الگوهای نموداری + لایه رژیم/اسکور + Smart Money + سطوح مطلق قیمت)،
> پیکربندی 1h/q0.90 در Walk-Forward سرتاسری (۱۶۰ پنجره، ۲۰۲۰-۰۳ → ۲۰۲۶-۱۰)
> **edge خالص ≈ +0.08R** با کنترل شافل ≈ صفر و Expectancy مثبت در **هر ۷ سال**
> نشون داد (جداول کامل: `ROADMAP.md` و `docs/indicators_strategies.md`).
> ⚠️ با این همه: بک‌تست تضمین آینده نیست و پیش از معامله‌ی واقعی،
> paper-trading و مدل‌سازی slippage/spread لازم است.

> ⚠️ **این پروژه یک ربات معاملاتی خودکار یا سیستم "سیگنال سودآور" نیست.**
> بعد از تست جدی (بک‌تست تکی، بهینه‌سازی Train/Test، و Walk-Forward روی ۶
> پنجره‌ی زمانی مختلف — نتایج کامل در بخش «نتایج بک‌تست» پایین همین فایل)،
> فرمول امتیازدهی فعلی **هیچ برتری آماری (edge) قابل‌اتکایی روی BTCUSDT
> نشون نداد**. هدف این پروژه ارائه‌ی یه تصویر تکنیکال شفاف و قابل‌فهمه که
> خودت با تجربه و قضاوت خودت باهاش تصمیم بگیری — نه جایگزینی برای تصمیم
> معاملاتی.

## 🎯 چیزی که این ابزار انجام میده
- **۲۰+ اندیکاتور فنی واقعی** (روند، مومنتوم، نوسان، حجم) روی داده‌ی زنده‌ی Binance
- **تشخیص الگوهای کندلی و نموداری** (لیست کامل پایین‌تر — بخشی پیاده شده، بخشی در دست ساخت)
- **تحلیل متنی از دو مدل AI محلی** (Llama 3 + Mistral) که وضعیت فعلی بازار رو به زبان ساده توضیح میدن
- **موتور بک‌تست + Walk-Forward Validation** برای این‌که خودت هم بتونی هر تغییری که میدی رو با داده‌ی واقعی بسنجی، نه فقط حدس بزنی

## 🎯 ویژگی‌های اصلی

### 📊 جمع‌آوری داده‌های بازار
- ✅ Binance API integration (با retry/backoff و pagination)
- ✅ Historical & real-time OHLCV (از طریق REST)
- ✅ Multi-timeframe analysis
- ✅ WebSocket streaming — پیاده شد برای Order Book depth
  (`orderbook_stream.py` — جریان `@depth20@100ms` با نمونه‌برداری دوره‌ای)

### 🔧 اندیکاتورهای فنی (همه‌ی این بخش واقعی و تست‌شده‌ست ✅)

#### Trend (7)
EMA (9/21/50/200)، MACD، ADX+DMI، SuperTrend، Parabolic SAR، ROC، Ichimoku Cloud

#### Momentum & Oscillators (8)
RSI، Stochastic RSI، CCI، Ultimate Oscillator، Williams %R، Awesome Oscillator، TRIX، KDJ

#### Volatility (5)
Bollinger Bands، Keltner Channels، ATR، Donchian Channels، BB Width & %B

#### Volume (5)
OBV، CMF، MFI، VWAP، Volume Rate of Change

#### Smart Money (4) 🟡
- ✅ **Cumulative Volume Delta (CVD)** — دلتای واقعی حجم از taker_buy_volume +
  واگرایی قیمت/CVD (تشخیص توزیع/انباشت پنهان)
- ✅ **Market / Volume Profile** — POC + Value Area (۷۰٪) + موقعیت قیمت نسبت به آن‌ها
- ✅ **Wyckoff (ساده‌شده)** — Spring/Upthrust (شکار استاپ) + موقعیت در رنج چرخان
- ✅ **فشار سفارش (تقریبی)** — Z-score و شتاب فشار خرید از taker data
- ⚠️ **نتیجه‌ی آزمایش (۱۴۰۵/۰۵/۳۰):** این ویژگی‌ها به‌تنهایی سیگنال دارند
  (sm-only AUC=0.624 روی 15m)، ولی با ۷۰ ویژگی موجود تکراری‌اند و در هیچ
  فرمی edge اضافی به مدل کامل ندادند (تفاوت AUC: 15m=-0.012 تا -0.020).
  در `features.py` به‌صورت اختیاری (`include_smart_money`) خاموش‌اند.
  ابزارهای آزمایش: `compare_smart_money.py` و `analyze_sm_features.py`.
  ارزش پیش‌بینی‌کنندگی آن‌ها هنوز با `compare_smart_money.py` آزمایش نشده است.
- ✅ **Order Book واقعی** — پیاده شد (`orderbook.py` + `orderbook_stream.py`):
  imbalance در ۴ عمق، spread، عمق bid/ask
- ⚠️ داده‌ی Order Book فقط «از حالا به بعد» جمع می‌شود (بایننس تاریخچه نمی‌دهد)؛
  بک‌تست تاریخی روی آن ممکن نیست — ابتدا باید چند روز/هفته داده جمع شود

### 🎨 الگوهای تکنیکال

#### Candlestick — ✅ ۱۴ الگو پیاده شده (Doji, Hammer, Hanging Man shape, Inverted Hammer/Shooting Star, Marubozu, Spinning Top, Engulfing, Harami, Piercing Line, Dark Cloud Cover, Morning/Evening Star, Three White Soldiers/Black Crows)
#### Reversal — ✅ Double Top/Bottom، Head & Shoulders + Inverse H&S پیاده شدن
#### Continuation — ✅ Triangles (Ascending/Descending/Symmetrical) + Flag/Pennant پیاده شدن
#### Harmonic — ✅ Gartley پیاده شد (رایج‌ترین الگوی هارمونیک) | بقیه (Butterfly, Crab, ...) ❌ برنامه‌ریزی‌شده — ⚠️ کم‌قطعیت‌ترین الگوی این پروژه، با احتیاط بیشتری بهش نگاه کن

### 🎯 استراتژی‌های تجاری جداگانه
❌ فعلاً به‌جای ۲۰+ استراتژی مجزا، یک فرمول امتیازدهی واحد (`SignalScorer`) داریم
که ترکیبی وزن‌دار از اندیکاتورهای بالاست. تفکیک به استراتژی‌های مستقل هنوز
انجام نشده.

### 🧠 AI Brain (Llama 3 + Mistral)
- ✅ تحلیل متنی مستقل از دو مدل، بر اساس داده‌ی فنی واقعی همون لحظه
- ❌ Ensemble consensus واقعی (رأی‌گیری/ترکیب دو نظر) — فعلاً فقط دو نظر جدا نمایش داده میشه
- ❌ News impact assessment — پیاده نشده

### 📈 Technical Score (نه "سیگنال خرید/فروش قطعی")
یک عدد ۰ تا ۱۰۰ که وضعیت فعلی بازار رو از نظر ترکیب اندیکاتورها خلاصه می‌کنه:
- Trend Score (0-40)
- Momentum Score (0-30)
- Volatility Score (0-15)
- Volume Score (0-15)

⚠️ طبق نتایج Walk-Forward Validation (پایین همین فایل)، این عدد **به‌تنهایی
edge آماری اثبات‌شده‌ای نداره** — به‌عنوان یه خلاصه‌ی سریع از وضعیت فعلی
اندیکاتورها نگاهش کن، نه یه توصیه‌ی معاملاتی.

### ⚠️ Risk Management
- 🟡 محاسبه‌ی Stop-loss/Take-profit مبتنی بر ATR — فقط داخل موتور بک‌تست پیاده شده، نه در اجرای زنده
- ❌ Position sizing، Portfolio allocation، Drawdown monitoring زنده — پیاده نشده

## 📉 نتایج بک‌تست (تا الان) — چرا این پروژه دیگه ادعای "سودآوری" نمی‌کنه

روی BTCUSDT/1h، چهار آزمایش جدا انجام شد:

| آزمایش | نتیجه |
|---|---|
| بک‌تست تکی (۱۰۰۰ کندل) | Win Rate 31.7%، Expectancy منفی (-0.049 R) |
| بهینه‌سازی Train/Test | بهترین پارامترهای Train روی Test همه منفی شدن (overfitting واضح) |
| Walk-Forward (۶ پنجره‌ی متوالی) | ۵۰٪ پنجره‌ها مثبت، میانگین وزن‌دار منفی (-0.083 R)، پارامتر «بهینه» هر پنجره کاملاً ناپایدار |
| Walk-Forward + فیلتر رژیم بازار (ADX) | بهبودی حاصل نشد (-0.106 R) |

**نتیجه:** ترکیب اندیکاتورهای استاندارد با فرمول امتیازدهی خطی و آستانه‌ی ثابت،
روی این نماد/تایم‌فریم edge آماری قابل‌اتکایی نداره. این پروژه به همین دلیل
از "ربات سیگنال" به "ابزار تحلیل" تغییر جهت داد. اگه می‌خوای خودت این تست‌ها
رو تکرار/گسترش بدی، اسکریپت‌های `backtest.py`، `optimize_backtest.py` و
`walk_forward.py` در دسترسن.

## 🛠️ نصب و راه‌اندازی
### الزامات
- Python 3.10+
- LM Studio (به عنوان سرور محلی استنتاج هوش مصنوعی)
- Binance API Keys
- 16GB+ RAM
- کارت گرافیک NVIDIA (ترجیحاً سری RTX با حداقل 12GB VRAM برای اجرای روان مدل‌های 8B و 7B)

### مراحل نصب

# 1. Clone repository
git clone https://github.com/rnosrati1989-bot/crypto-signal-detection.git
cd crypto-signal-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup LM Studio & Models
# 1. نرم‌افزار LM Studio را از سایت رسمی (https://lmstudio.ai) دانلود و نصب کنید.
# 2. مدل‌های Llama 3 8B و Mistral 7B را از داخل محیط LM Studio جستجو و دانلود کنید.
# 3. به بخش "Local Server" (آیکون <-> در منوی سمت چپ) بروید.
# 4. مدل دانلود شده را در بالا انتخاب و لود کنید.
# 5. بر روی دکمه "Start Server" کلیک کنید تا سرور محلی روی پورت 1234 (HTTP) فعال شود.

# 5. Configure environment
cp .env.example .env
# فایل .env را باز کرده و کلیدهای Binance و آدرس محلی سرور LM Studio را وارد کنید.

# 6. Run the system
python main.py

# 7. اجرای چند نماد/تایم‌فریم به‌صورت پیوسته
python multi_run.py

# 8. بک‌تست منطق اسکورینگ روی داده‌ی تاریخی (قبل از استفاده‌ی واقعی حتماً اجرا کنید)
python backtest.py --symbol BTCUSDT --timeframe 1h --limit 1000 --atr-mult 1.5 --rr 2.0

# 9. بهینه‌سازی پارامتر با اعتبارسنجی Train/Test (یک بازه)
python optimize_backtest.py --symbol BTCUSDT --timeframe 1h

# 10. اعتبارسنجی Walk-Forward روی چند پنجره‌ی زمانی متوالی (دقیق‌ترین آزمون)
python walk_forward.py --symbol BTCUSDT --timeframe 1h --n-windows 6

## 🚶 Walk-Forward Validation
`walk_forward.py` داده‌ی تاریخی رو به چند پنجره‌ی متوالی تقسیم می‌کنه (پیش‌فرض:
۶ پنجره، هرکدوم ۷۰۰ کندل Train + ۳۰۰ کندل Test). توی هر پنجره بهترین پارامتر
روی Train پیدا میشه و بلافاصله روی Test (داده‌ی کاملاً ندیده و بعد از Train)
امتحان میشه. این دقیق‌ترین روش موجود در این پروژه برای فهمیدن اینه که آیا
سیستم واقعاً edge پایدار داره یا فقط روی یه بازه‌ی خاص شانسی خوب بوده.

## 🧪 بک‌تست (Backtest)
اسکریپت `backtest.py` منطق فعلی `SignalScorer` رو روی داده‌ی تاریخی Binance
شبیه‌سازی می‌کنه: ورود در کندلِ بعد از سیگنال (بدون look-ahead bias)، حد ضرر/
حد سود بر اساس ATR، و خروجی شامل تعداد معاملات، Win Rate، میانگین R،
Expectancy، Profit Factor و Max Drawdown.

⚠️ این ابزار کارمزد صرافی، اسلیپیج و ریسک overfitting رو لحاظ نمی‌کنه؛ نتیجه‌ی
مثبت در بک‌تست به‌هیچ‌وجه تضمین سود در آینده نیست. قبل از هرگونه استفاده‌ی
واقعی از این پروژه برای معامله، حتماً این خروجی رو با احتیاط بررسی کنید.
