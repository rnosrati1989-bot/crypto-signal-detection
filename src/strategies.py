"""
Trading Strategies Module
Implements various trading strategies
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np


class StrategyType(Enum):
    """Trading strategy types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"


@dataclass
class StrategySignal:
    """Strategy signal result"""
    strategy: StrategyType
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    timeframe: str


class TrendFollowingStrategy:
    """Trend Following Strategy - استراتژی پیروی از روند"""
    
    def __init__(self):
        self.name = "Trend Following"
    
    def analyze(self, prices: np.ndarray, short_ma: float, long_ma: float) -> Optional[StrategySignal]:
        """تحلیل سیگنال"""
        if short_ma > long_ma:
            # صعودی
            confidence = min((short_ma / long_ma - 1) * 100, 100)
            action = "BUY"
        elif short_ma < long_ma:
            # نزولی
            confidence = min((1 - short_ma / long_ma) * 100, 100)
            action = "SELL"
        else:
            action = "HOLD"
            confidence = 0
        
        if action == "HOLD":
            return None
        
        current_price = prices[-1]
        atr = np.mean(np.abs(np.diff(prices[-14:])))
        
        if action == "BUY":
            target = current_price + (atr * 3)
            stop_loss = current_price - atr
        else:
            target = current_price - (atr * 3)
            stop_loss = current_price + atr
        
        risk_reward = abs(target - current_price) / abs(stop_loss - current_price)
        
        return StrategySignal(
            strategy=StrategyType.TREND_FOLLOWING,
            action=action,
            confidence=confidence,
            entry_price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            timeframe="4H"
        )


class MeanReversionStrategy:
    """Mean Reversion Strategy - استراتژی بازگشت به میانگین"""
    
    def __init__(self):
        self.name = "Mean Reversion"
    
    def analyze(self, prices: np.ndarray, rsi: float, bollinger_upper: float, bollinger_lower: float) -> Optional[StrategySignal]:
        """تحلیل سیگنال"""
        current_price = prices[-1]
        ma = np.mean(prices[-20:])
        
        action = None
        confidence = 0
        
        # اگر قیمت زیر باند پایین و RSI < 30
        if current_price < bollinger_lower and rsi < 30:
            action = "BUY"
            confidence = (30 - rsi) * 2  # 0-60
        # اگر قیمت بالای باند بالا و RSI > 70
        elif current_price > bollinger_upper and rsi > 70:
            action = "SELL"
            confidence = (rsi - 70) * 2  # 0-60
        else:
            return None
        
        if action == "BUY":
            target = ma
            stop_loss = current_price - (current_price - bollinger_lower) * 0.5
        else:
            target = ma
            stop_loss = current_price + (bollinger_upper - current_price) * 0.5
        
        risk_reward = abs(target - current_price) / abs(stop_loss - current_price) if stop_loss != current_price else 0
        
        return StrategySignal(
            strategy=StrategyType.MEAN_REVERSION,
            action=action,
            confidence=min(confidence, 100),
            entry_price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            timeframe="1H"
        )


class MomentumStrategy:
    """Momentum Strategy - استراتژی مومنتوم"""
    
    def __init__(self):
        self.name = "Momentum"
    
    def analyze(self, prices: np.ndarray, macd: float, macd_signal: float, volume: np.ndarray) -> Optional[StrategySignal]:
        """تحلیل سیگنال"""
        current_price = prices[-1]
        
        action = None
        confidence = 0
        
        # MACD crossover
        if macd > macd_signal and macd > 0:
            action = "BUY"
            confidence = min(abs(macd) * 10, 100)
        elif macd < macd_signal and macd < 0:
            action = "SELL"
            confidence = min(abs(macd) * 10, 100)
        else:
            return None
        
        recent_volume = np.mean(volume[-5:])
        if recent_volume < np.mean(volume[-20:]) * 0.8:
            confidence *= 0.7  # کاهش confidence اگر volume کم باشد
        
        atr = np.mean(np.abs(np.diff(prices[-14:])))
        
        if action == "BUY":
            target = current_price + (atr * 2)
            stop_loss = current_price - atr * 1.5
        else:
            target = current_price - (atr * 2)
            stop_loss = current_price + atr * 1.5
        
        risk_reward = abs(target - current_price) / abs(stop_loss - current_price)
        
        return StrategySignal(
            strategy=StrategyType.MOMENTUM,
            action=action,
            confidence=confidence,
            entry_price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            timeframe="15M"
        )


class BreakoutStrategy:
    """Breakout Strategy - استراتژی شکست سطح"""
    
    def __init__(self):
        self.name = "Breakout"
        self.lookback = 20
    
    def analyze(self, prices: np.ndarray, volume: np.ndarray) -> Optional[StrategySignal]:
        """تحلیل سیگنال"""
        current_price = prices[-1]
        recent_high = np.max(prices[-self.lookback:-1])
        recent_low = np.min(prices[-self.lookback:-1])
        
        action = None
        confidence = 0
        
        # Breakout بالا
        if current_price > recent_high:
            action = "BUY"
            distance_percent = (current_price - recent_high) / recent_high * 100
            confidence = min(distance_percent * 5, 100)
        # Breakdown پایین
        elif current_price < recent_low:
            action = "SELL"
            distance_percent = (recent_low - current_price) / recent_low * 100
            confidence = min(distance_percent * 5, 100)
        else:
            return None
        
        # Volume confirmation
        if volume[-1] > np.mean(volume[-5:]) * 1.5:
            confidence *= 1.1
        
        range_size = recent_high - recent_low
        
        if action == "BUY":
            target = current_price + range_size
            stop_loss = recent_low
        else:
            target = current_price - range_size
            stop_loss = recent_high
        
        risk_reward = abs(target - current_price) / abs(stop_loss - current_price)
        
        return StrategySignal(
            strategy=StrategyType.BREAKOUT,
            action=action,
            confidence=min(confidence, 100),
            entry_price=current_price,
            target_price=target,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            timeframe="4H"
        )


class StrategyEngine:
    """Strategy Engine - موتور استراتژی‌ها"""
    
    def __init__(self):
        self.strategies = {
            StrategyType.TREND_FOLLOWING: TrendFollowingStrategy(),
            StrategyType.MEAN_REVERSION: MeanReversionStrategy(),
            StrategyType.MOMENTUM: MomentumStrategy(),
            StrategyType.BREAKOUT: BreakoutStrategy(),
        }
    
    def run_all_strategies(self, market_data: dict) -> List[StrategySignal]:
        """اجرای تمام استراتژی‌ها"""
        signals = []
        
        prices = market_data.get('prices', [])
        if len(prices) < 20:
            return signals
        
        # Trend Following
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-20:])
        signal = self.strategies[StrategyType.TREND_FOLLOWING].analyze(prices, short_ma, long_ma)
        if signal:
            signals.append(signal)
        
        # Mean Reversion
        rsi = market_data.get('rsi', 50)
        bb_upper = market_data.get('bb_upper', np.max(prices))
        bb_lower = market_data.get('bb_lower', np.min(prices))
        signal = self.strategies[StrategyType.MEAN_REVERSION].analyze(prices, rsi, bb_upper, bb_lower)
        if signal:
            signals.append(signal)
        
        # Momentum
        macd = market_data.get('macd', 0)
        macd_signal = market_data.get('macd_signal', 0)
        volume = market_data.get('volume', [])
        signal = self.strategies[StrategyType.MOMENTUM].analyze(prices, macd, macd_signal, volume)
        if signal:
            signals.append(signal)
        
        # Breakout
        volume = market_data.get('volume', [])
        signal = self.strategies[StrategyType.BREAKOUT].analyze(prices, volume)
        if signal:
            signals.append(signal)
        
        return signals
