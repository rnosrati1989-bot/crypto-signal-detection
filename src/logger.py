import logging
import logging.handlers
import os
from datetime import datetime
from config import Config

class Logger:
    """Centralized logging system"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = logging.getLogger('crypto-signal-detection')
        self.logger.setLevel(Config.server.log_level)
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(Config.server.log_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # File Handler (Daily rotation)
        file_handler = logging.handlers.RotatingFileHandler(
            f'logs/app_{datetime.now().strftime("%Y%m%d")}.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        file_handler.setLevel(Config.server.log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        # Error File Handler
        error_handler = logging.FileHandler('logs/error.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def get_logger(self, name: str = None):
        return logging.getLogger(name or 'crypto-signal-detection')

# Singleton instance
logger = Logger().logger
