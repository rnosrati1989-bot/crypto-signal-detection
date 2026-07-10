import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

load_dotenv()

@dataclass
class BinanceConfig:
    """Binance API Configuration"""
    api_key: str = os.getenv('BINANCE_API_KEY', '')
    api_secret: str = os.getenv('BINANCE_API_SECRET', '')
    testnet: bool = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    
    # Trading parameters
    leverage: int = int(os.getenv('LEVERAGE', '10'))
    position_size_percent: float = float(os.getenv('POSITION_SIZE_PERCENT', '2'))
    stop_loss_percent: float = float(os.getenv('STOP_LOSS_PERCENT', '5'))
    take_profit_percent: float = float(os.getenv('TAKE_PROFIT_PERCENT', '10'))
    
    # Symbols to trade
    symbols: list = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 
        'DOGEUSDT', 'XRPUSDT', 'SOLUSDT', 'MATICUSDT'
    ]
    
    # Timeframes
    timeframes: list = ['1m', '5m', '15m', '1h', '4h', 'D']

@dataclass
class TelegramConfig:
    """Telegram Bot Configuration"""
    token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    enabled: bool = bool(os.getenv('TELEGRAM_BOT_TOKEN'))

@dataclass
class LMStudioConfig:
    """LM Studio (Ollama) Configuration"""
    base_url: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    model: str = os.getenv('OLLAMA_MODEL', 'mistral')
    enabled: bool = True

@dataclass
class DatabaseConfig:
    """Database Configuration"""
    url: str = os.getenv('DATABASE_URL', 'sqlite:///crypto_signals.db')
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379')

@dataclass
class ServerConfig:
    """API Server Configuration"""
    host: str = os.getenv('API_HOST', '0.0.0.0')
    port: int = int(os.getenv('API_PORT', '5000'))
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')

class Config:
    """Main Configuration Class"""
    binance = BinanceConfig()
    telegram = TelegramConfig()
    lm_studio = LMStudioConfig()
    database = DatabaseConfig()
    server = ServerConfig()
    
    # Global settings
    MAX_SIGNALS_PER_DAY = 20
    MIN_CONFIDENCE = 0.65
    MIN_VOLUME_24H = 1_000_000
    CACHE_TTL = 3600  # 1 hour
