import os
from dotenv import load_dotenv

# لود کردن فایل .env
load_dotenv()

class Config:
    # Binance API Configuration
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    
    # LM Studio Configuration
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "meta-llama-3-8b-instruct")
    
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
