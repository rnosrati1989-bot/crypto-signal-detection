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
- Ollama (برای Llama و Mistral)
- Binance API Keys
- 16GB+ RAM

### مراحل نصب

```bash
# 1. Clone repository
git clone https://github.com/rnosrati1989-bot/crypto-signal-detection.git
cd crypto-signal-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup Ollama
# Download and install Ollama from https://ollama.ai
# Run models:
ollama run llama-3-8b
ollama run mistral-7b

# 5. Configure environment
cp .env.example .env
# Edit .env with your Binance API keys and settings

# 6. Run the system
python main.py
