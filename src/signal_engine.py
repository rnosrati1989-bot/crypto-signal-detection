"""
Signal Generation Engine
Generates and validates trading signals
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_WEAK = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class TradeSignal:
    """Trade signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float  # 0-100
    strength: SignalStrength
    timeframe: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    indicators: Dict[str, float] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    
    risk_reward_ratio: float = 0.0
    win_probability: float = 0.0
    
    def __post_init__(self):
        """محاسبه نسبت ریسک/ریوارد"""
        if self.stop_loss != self.entry_price:
            self.risk_reward_ratio = abs(self.target_price - self.entry_price) / abs(self.stop_loss - self.entry_price)
    
    def is_valid(self) -> bool:
        """اعتبارسنجی سیگنال"""
        # حداقل confidence
        if self.confidence < 60:
            return False
        
        # حداقل risk/reward
        if self.risk_reward_ratio < 1.5:
            return False
        
        # بررسی logic
        if self.action == "BUY":
            if self.target_price <= self.entry_price or self.stop_loss >= self.entry_price:
                return False
        elif self.action == "SELL":
            if self.target_price >= self.entry_price or self.stop_loss <= self.entry_price:
                return False
        
        return True


class SignalValidator:
    """Signal validation"""
    
    @staticmethod
    def validate_signal(signal: TradeSignal) -> bool:
        """اعتبارسنجی سیگنال"""
        return signal.is_valid()
    
    @staticmethod
    def filter_signals(signals: List[TradeSignal], min_confidence: float = 70) -> List[TradeSignal]:
        """فیلتر کردن سیگنال‌ها"""
        filtered = [s for s in signals if s.confidence >= min_confidence and s.is_valid()]
        return sorted(filtered, key=lambda s: s.confidence, reverse=True)


class SignalScorer:
    """Signal scoring system"""
    
    @staticmethod
    def calculate_strength(confidence: float, risk_reward: float, pattern_count: int, strategy_count: int) -> SignalStrength:
        """محاسبه قدرت سیگنال"""
        score = 0
        
        # Confidence (0-40 امتیاز)
        score += (confidence / 100) * 40
        
        # Risk/Reward (0-30 امتیاز)
        rr_normalized = min(risk_reward / 3, 1)  # 3:1 نسبت بهترین
        score += rr_normalized * 30
        
        # Pattern confirmation (0-15 امتیاز)
        pattern_score = min(pattern_count / 3, 1) * 15
        score += pattern_score
        
        # Strategy agreement (0-15 امتیاز)
        strategy_score = min(strategy_count / 4, 1) * 15
        score += strategy_score
        
        # تعیین strength
        if score >= 80:
            return SignalStrength.VERY_STRONG
        elif score >= 65:
            return SignalStrength.STRONG
        elif score >= 50:
            return SignalStrength.MODERATE
        elif score >= 35:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    @staticmethod
    def calculate_win_probability(confidence: float, strength: SignalStrength, historical_win_rate: float = 0.55) -> float:
        """محاسبه احتمال برد"""
        confidence_factor = confidence / 100
        strength_factor = (strength.value + 1) / 5
        
        probability = (confidence_factor * 0.6 + strength_factor * 0.4) * historical_win_rate
        return min(probability, 0.95)


class SignalEngine:
    """Main Signal Engine"""
    
    def __init__(self):
        self.validator = SignalValidator()
        self.scorer = SignalScorer()
        self.signals_history: List[TradeSignal] = []
    
    def generate_signal(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        indicators: Dict[str, float],
        patterns: List[str],
        strategies: List[str],
        timeframe: str = "4H"
    ) -> Optional[TradeSignal]:
        """تولید سیگنال"""
        
        # محاسبه confidence از indicators
        confidence = self._calculate_confidence(indicators)
        
        # محاسبه strength
        strength = self.scorer.calculate_strength(
            confidence,
            abs(target_price - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 0,
            len(patterns),
            len(strategies)
        )
        
        # ایجاد سیگنال
        signal = TradeSignal(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            strength=strength,
            timeframe=timeframe,
            indicators=indicators,
            patterns=patterns,
            strategies=strategies,
            win_probability=self.scorer.calculate_win_probability(confidence, strength)
        )
        
        # اعتبارسنجی
        if not self.validator.validate_signal(signal):
            return None
        
        # ذخیره در history
        self.signals_history.append(signal)
        
        return signal
    
    def _calculate_confidence(self, indicators: Dict[str, float]) -> float:
        """محاسبه confidence از indicators"""
        confidence_scores = []
        
        # RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30 or rsi > 70:
                confidence_scores.append(min(abs(50 - rsi) / 20, 1) * 20)
        
        # MACD
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            if (macd > macd_signal and macd > 0) or (macd < macd_signal and macd < 0):
                confidence_scores.append(20)
        
        # Bollinger Bands
        if 'bb_position' in indicators:
            bb_pos = indicators['bb_position']
            if bb_pos < 0.2 or bb_pos > 0.8:
                confidence_scores.append(20)
        
        # Moving Averages
        if 'ma_trend' in indicators:
            confidence_scores.append(indicators['ma_trend'] * 20)
        
        # Volume
        if 'volume_ratio' in indicators:
            vol_ratio = indicators['volume_ratio']
            if vol_ratio > 1.2:
                confidence_scores.append(20)
        
        # میانگین کل
        if confidence_scores:
            return min(sum(confidence_scores) / len(confidence_scores), 100)
        
        return 50
    
    def merge_signals(self, signals: List[TradeSignal]) -> Optional[TradeSignal]:
        """ترکیب چند سیگنال"""
        if not signals:
            return None
        
        # فیلتر سیگنال‌های معتبر
        valid_signals = [s for s in signals if self.validator.validate_signal(s)]
        if not valid_signals:
            return None
        
        # بررسی اتفاق نظر
        buy_signals = sum(1 for s in valid_signals if s.action == "BUY")
        sell_signals = sum(1 for s in valid_signals if s.action == "SELL")
        
        if buy_signals == 0 and sell_signals == 0:
            return None
        
        # تعیین جهت
        if buy_signals > sell_signals:
            action = "BUY"
            selected = [s for s in valid_signals if s.action == "BUY"]
        else:
            action = "SELL"
            selected = [s for s in valid_signals if s.action == "SELL"]
        
        # میانگین‌گیری
        avg_entry = np.mean([s.entry_price for s in selected])
        avg_target = np.mean([s.target_price for s in selected])
        avg_stop = np.mean([s.stop_loss for s in selected])
        avg_confidence = np.mean([s.confidence for s in selected])
        
        # جمع‌آوری اطلاعات
        all_patterns = []
        all_strategies = []
        all_indicators = {}
        
        for s in selected:
            all_patterns.extend(s.patterns)
            all_strategies.extend(s.strategies)
            all_indicators.update(s.indicators)
        
        # ایجاد سیگنال ترکیبی
        merged_signal = TradeSignal(
            symbol=selected[0].symbol,
            action=action,
            entry_price=avg_entry,
            target_price=avg_target,
            stop_loss=avg_stop,
            confidence=min(avg_confidence * (len(selected) / 4), 100),  # افزایش confidence با تعداد سیگنال
            strength=self.scorer.calculate_strength(
                avg_confidence,
                abs(avg_target - avg_entry) / abs(avg_stop - avg_entry) if avg_stop != avg_entry else 0,
                len(set(all_patterns)),
                len(set(all_strategies))
            ),
            timeframe=selected[0].timeframe,
            patterns=list(set(all_patterns)),
            strategies=list(set(all_strategies)),
            indicators=all_indicators
        )
        
        return merged_signal
    
    def get_signal_stats(self) -> Dict:
        """آماری سیگنال‌ها"""
        if not self.signals_history:
            return {}
        
        buy_signals = [s for s in self.signals_history if s.action == "BUY"]
        sell_signals = [s for s in self.signals_history if s.action == "SELL"]
        
        return {
            'total_signals': len(self.signals_history),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_confidence': np.mean([s.confidence for s in self.signals_history]),
            'avg_win_probability': np.mean([s.win_probability for s in self.signals_history]),
            'strong_signals': sum(1 for s in self.signals_history if s.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG])
        }
