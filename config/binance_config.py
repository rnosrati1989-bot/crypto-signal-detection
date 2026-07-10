"""Binance API Configuration"""
import os
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

class BinanceConfig(BaseSettings):
    """Binance API Configuration"""
    
    # API Keys
    API_KEY = os.getenv('BINANCE_API_KEY', '')
    SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    
    # Base URLs
    BASE_URL = 'https://api.binance.com'
    TESTNET_URL = 'https://testnet.binance.vision'
    
    # WebSocket
    WS_URL = 'wss://stream.binance.com:9443'
    
    # Trading Pairs
    DEFAULT_PAIRS = [
        'BTCUSDT',
        'ETHUSDT',
        'BNBUSDT',
        'ADAUSDT',
        'XRPUSDT',
        'SOLSDT',
        'DOGEUSDT',
        'POLKADOT'
    ]
    
    # Timeframes
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    # Default Analysis Timeframe
    ANALYSIS_TIMEFRAMES = ['1h', '4h', '1d']
    
    # Rate Limiting
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    # Data Collection
    LOOKBACK_PERIOD = 500  # Number of candles to fetch
    UPDATE_INTERVAL = 60  # Seconds between updates
    
    class Config:
        env_file = '.env'

binance_config = BinanceConfig()
