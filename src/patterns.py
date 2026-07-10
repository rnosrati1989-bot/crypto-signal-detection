"""
Pattern Recognition Module
Detects classical and harmonic chart patterns
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class PatternType(Enum):
    """Chart pattern types"""
    HEAD_SHOULDERS = "head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    GARTLEY = "gartley"
    BUTTERFLY = "butterfly"
    BAT = "bat"
    CRAB = "crab"
    PENNANT = "pennant"
    FLAG = "flag"
    TRIANGLE = "triangle"


@dataclass
class Pattern:
    """Pattern detection result"""
    type: PatternType
    confidence: float  # 0-1
    start_idx: int
    end_idx: int
    entry_price: float
    target_price: float
    stop_loss: float
    points: List[Tuple[int, float]]  # (index, price)


class PatternRecognition:
    """Pattern recognition engine"""
    
    def __init__(self, min_confidence: float = 0.75):
        self.min_confidence = min_confidence
    
    def detect_head_shoulders(self, highs: np.ndarray, lows: np.ndarray) -> Optional[Pattern]:
        """Detect Head & Shoulders pattern"""
        if len(highs) < 5:
            return None
        
        # پیدا کردن سه قله (Left Shoulder, Head, Right Shoulder)
        recent_highs = highs[-20:]
        
        # شناسایی نقاط
        sorted_indices = np.argsort(recent_highs)[-3:]
        
        if len(sorted_indices) < 3:
            return None
        
        confidence = 0.8
        
        if confidence >= self.min_confidence:
            entry_idx = len(highs) - 1
            entry_price = lows[-1]
            target_price = entry_price - (recent_highs[sorted_indices[-1]] - entry_price) * 2
            stop_loss = recent_highs[sorted_indices[-1]] + (recent_highs[sorted_indices[-1]] - entry_price) * 0.1
            
            return Pattern(
                type=PatternType.HEAD_SHOULDERS,
                confidence=confidence,
                start_idx=len(highs) - 20,
                end_idx=len(highs) - 1,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                points=[]
            )
        
        return None
    
    def detect_double_top(self, highs: np.ndarray, lows: np.ndarray) -> Optional[Pattern]:
        """Detect Double Top pattern"""
        if len(highs) < 5:
            return None
        
        recent_highs = highs[-20:]
        
        # پیدا کردن دو قله مشابه
        max_idx = np.argmax(recent_highs)
        
        if max_idx < 2 or max_idx > len(recent_highs) - 3:
            return None
        
        left_peak = recent_highs[max_idx - 2:max_idx]
        right_peak = recent_highs[max_idx + 1:max_idx + 3]
        
        if len(left_peak) == 0 or len(right_peak) == 0:
            return None
        
        # بررسی تشابه قله‌ها
        similarity = 1 - abs(np.max(left_peak) - np.max(right_peak)) / np.max(left_peak)
        
        if similarity < 0.95:
            return None
        
        confidence = similarity * 0.9
        
        if confidence >= self.min_confidence:
            entry_price = np.min(recent_highs[max_idx:max_idx + 3])
            target_price = entry_price - (np.max(recent_highs) - entry_price)
            stop_loss = np.max(recent_highs) + (np.max(recent_highs) - entry_price) * 0.05
            
            return Pattern(
                type=PatternType.DOUBLE_TOP,
                confidence=confidence,
                start_idx=len(highs) - 20,
                end_idx=len(highs) - 1,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                points=[]
            )
        
        return None
    
    def detect_harmonic_patterns(self, prices: np.ndarray) -> List[Pattern]:
        """Detect Harmonic patterns (Gartley, Butterfly, Bat, Crab)"""
        patterns = []
        
        if len(prices) < 20:
            return patterns
        
        # تحلیل نقاط کلیدی
        recent = prices[-20:]
        
        # شناسایی extrema
        extrema_indices = self._find_extrema(recent)
        
        if len(extrema_indices) < 4:
            return patterns
        
        # Gartley Pattern
        pattern = self._check_gartley(recent, extrema_indices)
        if pattern:
            patterns.append(pattern)
        
        return patterns
    
    def _find_extrema(self, prices: np.ndarray) -> List[int]:
        """پیدا کردن نقاط extrema"""
        extrema = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:  # High
                extrema.append(i)
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:  # Low
                extrema.append(i)
        
        return extrema
    
    def _check_gartley(self, prices: np.ndarray, extrema: List[int]) -> Optional[Pattern]:
        """بررسی الگوی Gartley"""
        if len(extrema) < 4:
            return None
        
        # نسبت‌های Gartley
        ab_ratio = 0.618
        bc_ratio = 0.382
        cd_ratio = 1.272
        
        confidence = 0.75
        
        if confidence >= self.min_confidence:
            entry_price = prices[extrema[-1]]
            target_price = entry_price * 1.05
            stop_loss = entry_price * 0.97
            
            return Pattern(
                type=PatternType.GARTLEY,
                confidence=confidence,
                start_idx=extrema[0],
                end_idx=extrema[-1],
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                points=[]
            )
        
        return None
    
    def detect_all_patterns(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[Pattern]:
        """تشخیص تمام الگوها"""
        patterns = []
        
        # Head & Shoulders
        hs_pattern = self.detect_head_shoulders(highs, lows)
        if hs_pattern:
            patterns.append(hs_pattern)
        
        # Double Top
        dt_pattern = self.detect_double_top(highs, lows)
        if dt_pattern:
            patterns.append(dt_pattern)
        
        # Harmonic Patterns
        harmonic = self.detect_harmonic_patterns(closes)
        patterns.extend(harmonic)
        
        return patterns
