# 🚀 Advanced Crypto Signal Detection System

یک سیستم جامع و پیشرفته برای تشخیص سیگنال‌های تجاری در بازارهای ارز دیجیتال با استفاده از:
- **20+ اندیکاتور فنی**
- **30+ الگوی تکنیکال**
- **20+ استراتژی تجاری**
- **AI هوشمند (Llama 3.8B + Mistral 7B)**
- **API Binance Real-time**
- **سیستم نمره‌دهی جامع**

## 🎯 ویژگی‌های اصلی

### 📊 جمع‌آوری داده‌های بازار
- ✅ Binance API integration
- ✅ Real-time price feeds (OHLCV)
- ✅ Multi-timeframe analysis
- ✅ WebSocket streaming
- ✅ Historical data backfill

### 🔧 اندیکاتورهای فنی

#### Trend Indicators (8)
- EMA/SMA Multi-timeframe
- MACD + Divergence
- ADX + DMI
- Parabolic SAR
- Ichimoku Cloud
- ROC
- KDJ
- SuperTrend

#### Momentum & Oscillators (8)
- RSI Multi-period
- Stochastic RSI
- CCI
- Ultimate Oscillator
- Williams %R
- Awesome Oscillator
- TRIX
- DMI

#### Volatility (7)
- Bollinger Bands
- Keltner Channels
- ATR
- Donchian Channels
- Volatility Index
- BB Width & %B
- ATR-adjusted levels

#### Volume (6)
- OBV
- Volume Rate of Change
- CMF
- MFI
- VWAP
- Volume Profile

#### Smart Money (5)
- Wyckoff Distribution/Accumulation
- Market Profile
- Cumulative Delta
- Smart Money Zones
- Orderflow Analysis

### 🎨 الگوهای تکنیکال

#### Reversal (12)
Head & Shoulders, Double Top/Bottom, Triple patterns, Rounding, V-Shape, Wedges, Diamond, Cup & Handle, etc.

#### Continuation (10)
Flags, Pennants, Triangles, Rectangles, Coils, etc.

#### Candlestick (15)
Star patterns, Engulfing, Harami, Doji, Hammer, etc.

#### Harmonic (15)
Gartley, Butterfly, Crab, Shark, Cypher, AB=CD, XAB=XCD, Fibonacci ratios, etc.

### 🎯 استراتژی‌های تجاری

#### Trend Following (6)
- EMA Cloud Breakout
- Ichimoku Breakout
- SuperTrend + RSI
- PSAR Strategy
- ADX + DMI
- MA Ribbon

#### Mean Reversion (6)
- RSI Extremes
- BB Squeeze Breakout
- Stochastic RSI
- CMF + MFI
- %B and Width
- KDJ Crossover

#### Harmonic (5)
- Gartley 222
- Elliot Wave
- Pivot Points
- Cup & Handle
- H&S Pattern

#### Smart Money (3)
- Wyckoff Phases
- Liquidity Grab
- Delta Imbalance

#### Multi-Timeframe (4)
- Multi-TF Alignment
- Macro + Technical
- Volatility Expansion
- Support/Resistance

### 🧠 AI Brain (Llama 3.8B + Mistral 7B)
- Signal analysis and confirmation
- Pattern recognition
- News impact assessment
- Risk management
- Ensemble consensus

### 📈 Signal Scoring
- Trend Score (0-40)
- Momentum Score (0-30)
- Volatility Score (0-15)
- Volume Score (0-10)
- Pattern Score (0-5)
- **Total: 0-100**

### ⚠️ Risk Management
- Stop-loss calculation
- Position sizing
- Risk/Reward ratio enforcement
- Portfolio allocation
- Drawdown monitoring

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
