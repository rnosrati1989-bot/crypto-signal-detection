"""
Risk Management Module
Manages trading risk, position sizing, and portfolio management
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from enum import Enum


class RiskLevel(Enum):
    """Risk levels"""
    CONSERVATIVE = 1
    MODERATE = 2
    AGGRESSIVE = 3


@dataclass
class Position:
    """Trading position"""
    symbol: str
    entry_price: float
    quantity: float
    side: str  # LONG, SHORT
    entry_time: str
    stop_loss: float
    take_profit: float
    risk_amount: float
    reward_amount: float
    
    @property
    def current_risk(self) -> float:
        """ریسک موقعیت"""
        return self.risk_amount
    
    @property
    def current_reward(self) -> float:
        """پاداش موقعیت"""
        return self.reward_amount
    
    @property
    def risk_reward_ratio(self) -> float:
        """نسبت ریسک/ریوارد"""
        if self.risk_amount == 0:
            return 0
        return self.reward_amount / self.risk_amount


@dataclass
class PortfolioMetrics:
    """Portfolio metrics"""
    total_balance: float
    total_equity: float
    total_margin_used: float
    used_margin_percent: float
    free_margin: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float


class RiskCalculator:
    """Risk calculation utilities"""
    
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """محاسبه اندازه موقعیت"""
        risk_amount = account_balance * (risk_percent / 100)
        price_difference = abs(entry_price - stop_loss)
        
        if price_difference == 0:
            return 0
        
        quantity = risk_amount / price_difference
        return quantity
    
    @staticmethod
    def calculate_stop_loss(
        entry_price: float,
        atr: float,
        multiplier: float = 2.0,
        side: str = "LONG"
    ) -> float:
        """محاسبه stop loss"""
        if side == "LONG":
            return entry_price - (atr * multiplier)
        else:
            return entry_price + (atr * multiplier)
    
    @staticmethod
    def calculate_take_profit(
        entry_price: float,
        atr: float,
        multiplier: float = 3.0,
        side: str = "LONG"
    ) -> float:
        """محاسبه take profit"""
        if side == "LONG":
            return entry_price + (atr * multiplier)
        else:
            return entry_price - (atr * multiplier)
    
    @staticmethod
    def calculate_risk_reward_ratio(
        entry_price: float,
        take_profit: float,
        stop_loss: float
    ) -> float:
        """محاسبه نسبت ریسک/ریوارد"""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0
        
        return reward / risk


class PositionSizer:
    """Position sizing strategies"""
    
    def __init__(self, account_balance: float, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.account_balance = account_balance
        self.risk_level = risk_level
        self.risk_percent = self._get_risk_percent()
    
    def _get_risk_percent(self) -> float:
        """تعیین درصد ریسک بر اساس سطح ریسک"""
        if self.risk_level == RiskLevel.CONSERVATIVE:
            return 1.0
        elif self.risk_level == RiskLevel.MODERATE:
            return 2.0
        else:  # AGGRESSIVE
            return 3.0
    
    def calculate_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float
    ) -> float:
        """محاسبه اندازه با در نظر گرفتن confidence"""
        base_size = RiskCalculator.calculate_position_size(
            self.account_balance,
            self.risk_percent,
            entry_price,
            stop_loss
        )
        
        # افزایش اندازه بر اساس confidence
        confidence_multiplier = 0.5 + (confidence / 100 * 0.5)
        
        return base_size * confidence_multiplier
    
    def calculate_multiple_positions(
        self,
        positions_data: List[Dict]
    ) -> Dict:
        """محاسبه اندازه برای چند موقعیت"""
        total_risk = 0
        sizes = {}
        
        # اول محاسبه کل ریسک
        for pos_data in positions_data:
            risk = abs(pos_data['entry_price'] - pos_data['stop_loss'])
            total_risk += risk
        
        # سپس تخصیص سرمایه
        for pos_data in positions_data:
            risk = abs(pos_data['entry_price'] - pos_data['stop_loss'])
            risk_percent = risk / total_risk if total_risk > 0 else 0
            
            size = self.calculate_size(
                pos_data['entry_price'],
                pos_data['stop_loss'],
                pos_data.get('confidence', 70)
            ) * risk_percent
            
            sizes[pos_data['symbol']] = size
        
        return sizes


class PortfolioManager:
    """Portfolio management"""
    
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict] = []
    
    def add_position(self, position: Position) -> bool:
        """اضافه کردن موقعیت"""
        required_margin = position.quantity * position.entry_price
        
        if required_margin > self.current_balance:
            return False
        
        self.positions[position.symbol] = position
        self.current_balance -= position.risk_amount
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """بستن موقعیت"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if position.side == "LONG":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        trade_data = {
            'symbol': symbol,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'pnl_percent': (pnl / (position.quantity * position.entry_price)) * 100,
            'side': position.side,
            'duration': 'TBD'
        }
        
        self.closed_trades.append(trade_data)
        self.current_balance += position.quantity * exit_price
        del self.positions[symbol]
        
        return trade_data
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """دریافت معیارهای پورتفولیو"""
        # محاسبه equity
        open_pnl = 0
        for position in self.positions.values():
            open_pnl += position.reward_amount if position.reward_amount > 0 else 0
        
        total_equity = self.current_balance + open_pnl
        
        # محاسبه win rate
        if self.closed_trades:
            winning_trades = sum(1 for t in self.closed_trades if t['pnl'] > 0)
            win_rate = winning_trades / len(self.closed_trades)
        else:
            win_rate = 0.5
        
        # محاسبه profit factor
        total_profit = sum(max(t['pnl'], 0) for t in self.closed_trades)
        total_loss = sum(abs(min(t['pnl'], 0)) for t in self.closed_trades)
        profit_factor = total_profit / total_loss if total_loss > 0 else 1.0
        
        # محاسبه drawdown
        pnls = [t['pnl'] for t in self.closed_trades]
        if pnls:
            cumulative_pnl = np.cumsum(pnls)
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - peak) / peak
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
        
        return PortfolioMetrics(
            total_balance=self.current_balance,
            total_equity=total_equity,
            total_margin_used=sum(p.risk_amount for p in self.positions.values()),
            used_margin_percent=(sum(p.risk_amount for p in self.positions.values()) / self.initial_balance) * 100 if self.initial_balance > 0 else 0,
            free_margin=self.current_balance,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=0.0,  # نیاز به محاسبه‌ی دقیق‌تر
            sortino_ratio=0.0   # نیاز به محاسبه‌ی دقیق‌تر
        )
    
    def get_active_positions(self) -> List[Position]:
        """دریافت موقعیت‌های فعال"""
        return list(self.positions.values())
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """بررسی stop loss"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        if position.side == "LONG":
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss
    
    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """بررسی take profit"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        if position.side == "LONG":
            return current_price >= position.take_profit
        else:
            return current_price <= position.take_profit
