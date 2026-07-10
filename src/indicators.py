import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from ta import momentum, trend, volatility, volume
from src.logger import logger

class TechnicalIndicators:
    """Technical Indicators Calculator"""
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        return talib.EMA(data, timeperiod=period)
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return talib.SMA(data, timeperiod=period)
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        macd, signal_line, histogram = talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.Series(macd), pd.Series(signal_line), pd.Series(histogram)
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        return talib.RSI(data, timeperiod=period)
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(data, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        return pd.Series(upper), pd.Series(middle), pd.Series(lower)
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        return talib.ATR(high, low, close, timeperiod=period)
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        return talib.ADX(high, low, close, timeperiod=period)
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        k, d = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
        return pd.Series(k), pd.Series(d)
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        return talib.OBV(close, volume)
    
    @staticmethod
    def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        return talib.MFI(high, low, close, volume, timeperiod=period)
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        return talib.CCI(high, low, close, timeperiod=period)
    
    @staticmethod
    def calculate_aroon(high: pd.Series, low: pd.Series, period: int = 25) -> Tuple[pd.Series, pd.Series]:
        """Aroon Indicator"""
        aroon_up, aroon_down = talib.AROON(high, low, timeperiod=period)
        return pd.Series(aroon_up), pd.Series(aroon_down)
    
    @staticmethod
    def calculate_psar(high: pd.Series, low: pd.Series, iaf: float = 0.02, maxaf: float = 0.2) -> pd.Series:
        """Parabolic SAR"""
        return talib.SAR(high, low, acceleration=iaf, maximum=maxaf)
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        return talib.WILLR(high, low, close, timeperiod=period)
    
    @staticmethod
    def calculate_roc(data: pd.Series, period: int = 12) -> pd.Series:
        """Rate of Change"""
        return talib.ROC(data, timeperiod=period)
    
    @staticmethod
    def calculate_trix(data: pd.Series, period: int = 15) -> pd.Series:
        """TRIX - Triple Exponential Moving Average"""
        return talib.TRIX(data, timeperiod=period)
    
    @staticmethod
    def calculate_ultosc(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod1: int = 7, timeperiod2: int = 14, timeperiod3: int = 28) -> pd.Series:
        """Ultimate Oscillator"""
        return talib.ULTOSC(high, low, close, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
    
    @staticmethod
    def calculate_dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Directional Movement Index"""
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
        adx = talib.ADX(high, low, close, timeperiod=period)
        return pd.Series(plus_di), pd.Series(minus_di), pd.Series(adx)
    
    @staticmethod
    def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """KDJ Indicator (Stochastic variant)"""
        k, d = talib.STOCH(high, low, close, fastk_period=period, slowk_period=3, slowd_period=3)
        j = 3 * k - 2 * d
        return pd.Series(k), pd.Series(d), pd.Series(j)
    
    @staticmethod
    def calculate_awesome_oscillator(high: pd.Series, low: pd.Series, period1: int = 5, period2: int = 34) -> pd.Series:
        """Awesome Oscillator"""
        hl2 = (high + low) / 2
        ao = hl2.rolling(period1).mean() - hl2.rolling(period2).mean()
        return ao
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> Dict:
        """Calculate all indicators for a dataframe"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            indicators = {
                'ema_20': TechnicalIndicators.calculate_ema(close, 20),
                'ema_50': TechnicalIndicators.calculate_ema(close, 50),
                'ema_200': TechnicalIndicators.calculate_ema(close, 200),
                'sma_20': TechnicalIndicators.calculate_sma(close, 20),
                'sma_50': TechnicalIndicators.calculate_sma(close, 50),
                'sma_200': TechnicalIndicators.calculate_sma(close, 200),
                'rsi': TechnicalIndicators.calculate_rsi(close, 14),
                'atr': TechnicalIndicators.calculate_atr(high, low, close, 14),
                'adx': TechnicalIndicators.calculate_adx(high, low, close, 14),
                'cci': TechnicalIndicators.calculate_cci(high, low, close, 20),
                'williams_r': TechnicalIndicators.calculate_williams_r(high, low, close, 14),
                'mfi': TechnicalIndicators.calculate_mfi(high, low, close, volume, 14),
                'obv': TechnicalIndicators.calculate_obv(close, volume),
                'roc': TechnicalIndicators.calculate_roc(close, 12),
                'trix': TechnicalIndicators.calculate_trix(close, 15),
                'awesome_os': TechnicalIndicators.calculate_awesome_oscillator(high, low, 5, 34),
            }
            
            # Multi-value indicators
            macd, signal, hist = TechnicalIndicators.calculate_macd(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = hist
            
            bb_up, bb_mid, bb_low = TechnicalIndicators.calculate_bollinger_bands(close, 20, 2)
            indicators['bb_upper'] = bb_up
            indicators['bb_middle'] = bb_mid
            indicators['bb_lower'] = bb_low
            
            stoch_k, stoch_d = TechnicalIndicators.calculate_stochastic(high, low, close, 14, 3)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            
            k, d, j = TechnicalIndicators.calculate_kdj(high, low, close, 9)
            indicators['kdj_k'] = k
            indicators['kdj_d'] = d
            indicators['kdj_j'] = j
            
            plus_di, minus_di, adx_dmi = TechnicalIndicators.calculate_dmi(high, low, close, 14)
            indicators['plus_di'] = plus_di
            indicators['minus_di'] = minus_di
            indicators['adx_dmi'] = adx_dmi
            
            aroon_up, aroon_down = TechnicalIndicators.calculate_aroon(high, low, 25)
            indicators['aroon_up'] = aroon_up
            indicators['aroon_down'] = aroon_down
            
            indicators['psar'] = TechnicalIndicators.calculate_psar(high, low, 0.02, 0.2)
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
