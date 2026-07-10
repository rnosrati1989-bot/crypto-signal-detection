from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from config import Config
from src.logger import logger
from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
import pandas as pd

class BinanceClient:
    """Binance Futures Trading Client"""
    
    def __init__(self):
        self.client = Client(
            api_key=Config.binance.api_key,
            api_secret=Config.binance.api_secret,
            testnet=Config.binance.testnet
        )
        self.testnet = Config.binance.testnet
        self.leverage = Config.binance.leverage
        logger.info(f"Binance Client initialized (Testnet: {self.testnet})")
    
    async def get_account_balance(self) -> Dict:
        """Get account balance"""
        try:
            account = self.client.futures_account()
            return {
                'total_balance': float(account['totalWalletBalance']),
                'available_balance': float(account['availableBalance']),
                'total_unrealized_profit': float(account['totalUnrealizedProfit']),
                'positions': account['positions']
            }
        except BinanceAPIException as e:
            logger.error(f"Error fetching balance: {e}")
            return None
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Get candlestick data"""
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
        except BinanceAPIException as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return None
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol"""
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            return True
        except BinanceAPIException as e:
            logger.error(f"Error setting leverage: {e}")
            return False
    
    async def open_position(
        self,
        symbol: str,
        side: str,  # LONG or SHORT
        quantity: float,
        stop_loss_price: float,
        take_profit_price: float
    ) -> Dict:
        """Open a futures position with stop loss and take profit"""
        try:
            order_side = "BUY" if side == "LONG" else "SELL"
            
            # Open position
            position_order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"Position opened: {side} {quantity} {symbol}")
            
            # Set stop loss
            sl_order = self.client.futures_create_order(
                symbol=symbol,
                side="SELL" if side == "LONG" else "BUY",
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_loss_price,
                timeInForce='GTC'
            )
            
            # Set take profit
            tp_order = self.client.futures_create_order(
                symbol=symbol,
                side="SELL" if side == "LONG" else "BUY",
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=take_profit_price,
                timeInForce='GTC'
            )
            
            return {
                'status': 'success',
                'position_order': position_order['orderId'],
                'stop_loss_order': sl_order['orderId'],
                'take_profit_order': tp_order['orderId'],
                'side': side,
                'symbol': symbol,
                'quantity': quantity
            }
        except BinanceOrderException as e:
            logger.error(f"Error opening position: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def close_position(self, symbol: str, quantity: float, side: str) -> Dict:
        """Close a position"""
        try:
            close_side = "SELL" if side == "LONG" else "BUY"
            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"Position closed: {side} {quantity} {symbol}")
            return {'status': 'success', 'orderId': order['orderId']}
        except BinanceOrderException as e:
            logger.error(f"Error closing position: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            account = self.client.futures_account()
            positions = []
            
            for pos in account['positions']:
                if float(pos['positionAmt']) != 0:
                    positions.append({
                        'symbol': pos['symbol'],
                        'quantity': float(pos['positionAmt']),
                        'entry_price': float(pos['entryPrice']),
                        'mark_price': float(pos['markPrice']),
                        'unrealized_profit': float(pos['unrealizedProfit']),
                        'liquidation_price': float(pos['liquidationPrice'])
                    })
            
            return positions
        except BinanceAPIException as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol trading info"""
        try:
            info = self.client.futures_exchange_info()
            for sym in info['symbols']:
                if sym['symbol'] == symbol:
                    return {
                        'symbol': sym['symbol'],
                        'status': sym['status'],
                        'base_asset': sym['baseAsset'],
                        'quote_asset': sym['quoteAsset'],
                        'price_precision': sym['pricePrecision'],
                        'quantity_precision': sym['quantityPrecision']
                    }
            return None
        except BinanceAPIException as e:
            logger.error(f"Error fetching symbol info: {e}")
            return None
