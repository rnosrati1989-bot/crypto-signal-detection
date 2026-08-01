import os
from dotenv import load_dotenv

# لود کردن فایل .env
load_dotenv()

class Config:
    # تنظیمات عمومی پروژه
    LOG_FILE = os.getenv("LOG_FILE", "logs/trading_bot.log")
    
    # تنظیمات صرافی بایننس
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    
    # تنظیمات هوش مصنوعی از فایل .env
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    MODEL_LLAMA = os.getenv("MODEL_LLAMA", "meta-llama-3-8b-instruct")
    MODEL_MISTRAL = os.getenv("MODEL_MISTRAL", "mistralai/mistral-7b-instruct-v0.3")

    
    # Models Directory
    MODELS_DIR = os.getenv("MODELS_DIR", "./models")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/crypto_signals")
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/crypto_signals.log")
    
    # Trading Settings
    TEST_MODE = os.getenv("TEST_MODE", "True").lower() in ("true", "1", "yes")
    MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.5"))
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "2.0"))
    
    # Notifications (Telegram)
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # جفت ارزها و تایم‌فریم‌های پیش‌فرض پروژه
    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    TIMEFRAMES = ["15m", "1h", "4h", "1d"]
