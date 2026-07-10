"""
AI Brain Module - LLM Integration
Analyzes signals and provides AI-powered insights
"""

import json
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class AIModel(Enum):
    """Available AI models"""
    LLAMA2 = "llama2"
    MISTRAL = "mistral"
    NEURAL_CHAT = "neural-chat"
    OPENHERMES = "openhermes"


@dataclass
class AIAnalysis:
    """AI analysis result"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    reasoning: str
    confidence: float  # 0-1
    risks: List[str]
    opportunities: List[str]
    market_sentiment: str  # BULLISH, BEARISH, NEUTRAL
    price_targets: Dict[str, float]  # short, medium, long term
    time_horizon: str


class AIBrain:
    """AI Brain - Ollama integration"""
    
    def __init__(self, model: AIModel = AIModel.MISTRAL, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Ollama client"""
        try:
            import requests
            self.session = requests.Session()
            # تست اتصال
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Ollama at {self.base_url}")
            else:
                logger.warning(f"Ollama connection failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
    
    def analyze_signal(self, signal_data: Dict) -> Optional[AIAnalysis]:
        """تحلیل سیگنال با هوش مصنوعی"""
        if not self.session:
            return self._fallback_analysis(signal_data)
        
        try:
            # آماده‌سازی prompt
            prompt = self._prepare_analysis_prompt(signal_data)
            
            # ارسال به Ollama
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model.value,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_analysis_response(result.get('response', ''), signal_data)
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
        
        return self._fallback_analysis(signal_data)
    
    def _prepare_analysis_prompt(self, signal_data: Dict) -> str:
        """آماده‌سازی prompt برای تحلیل"""
        symbol = signal_data.get('symbol', 'BTC/USDT')
        price = signal_data.get('price', 0)
        action = signal_data.get('action', 'HOLD')
        confidence = signal_data.get('confidence', 0)
        indicators = signal_data.get('indicators', {})
        patterns = signal_data.get('patterns', [])
        
        prompt = f"""
As a professional crypto trader, analyze this trading signal:

Symbol: {symbol}
Current Price: ${price}
Signal Action: {action}
Confidence: {confidence}%

Technical Indicators:
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- Bollinger Bands Position: {indicators.get('bb_position', 'N/A')}
- Moving Averages Trend: {indicators.get('ma_trend', 'N/A')}
- Volume Ratio: {indicators.get('volume_ratio', 'N/A')}

Chart Patterns Detected:
{', '.join(patterns) if patterns else 'No patterns detected'}

Please provide:
1. Trading Action (BUY/SELL/HOLD)
2. Confidence Level (0-100)
3. Key Risks (list 2-3)
4. Key Opportunities (list 2-3)
5. Market Sentiment (BULLISH/BEARISH/NEUTRAL)
6. Price Targets (short-term, medium-term, long-term)
7. Suggested Time Horizon (minutes/hours/days)

Format your response as JSON.
"""
        return prompt
    
    def _parse_analysis_response(self, response: str, signal_data: Dict) -> Optional[AIAnalysis]:
        """تجزیه و تحلیل پاسخ"""
        try:
            # استخراج JSON از response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return self._fallback_analysis(signal_data)
            
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            return AIAnalysis(
                symbol=signal_data.get('symbol', 'BTC/USDT'),
                action=parsed.get('action', signal_data.get('action', 'HOLD')),
                reasoning=parsed.get('reasoning', ''),
                confidence=min(int(parsed.get('confidence', 50)) / 100, 1.0),
                risks=parsed.get('risks', []),
                opportunities=parsed.get('opportunities', []),
                market_sentiment=parsed.get('sentiment', 'NEUTRAL'),
                price_targets={
                    'short_term': float(parsed.get('short_target', signal_data.get('price', 0))),
                    'medium_term': float(parsed.get('medium_target', signal_data.get('price', 0))),
                    'long_term': float(parsed.get('long_target', signal_data.get('price', 0)))
                },
                time_horizon=parsed.get('time_horizon', '4H')
            )
        
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._fallback_analysis(signal_data)
    
    def _fallback_analysis(self, signal_data: Dict) -> AIAnalysis:
        """تحلیل بازگشتی (بدون Ollama)"""
        symbol = signal_data.get('symbol', 'BTC/USDT')
        price = signal_data.get('price', 0)
        action = signal_data.get('action', 'HOLD')
        confidence = signal_data.get('confidence', 50)
        
        # تخمین بر اساس سیگنال
        risks = ["Market volatility", "Unexpected news", "Technical failure"]
        opportunities = ["Trend continuation", "Pattern breakout", "Volume confirmation"]
        
        sentiment = "BULLISH" if action == "BUY" else "BEARISH" if action == "SELL" else "NEUTRAL"
        
        target_diff = price * 0.05 if action == "BUY" else price * (-0.05)
        
        return AIAnalysis(
            symbol=symbol,
            action=action,
            reasoning=f"Signal based on technical analysis with {confidence}% confidence",
            confidence=confidence / 100,
            risks=risks,
            opportunities=opportunities,
            market_sentiment=sentiment,
            price_targets={
                'short_term': price + target_diff,
                'medium_term': price + target_diff * 2,
                'long_term': price + target_diff * 3
            },
            time_horizon="4H"
        )
    
    def generate_report(self, symbols: List[str], analyses: List[AIAnalysis]) -> str:
        """تولید گزارش تحلیلی"""
        report = "# Crypto Trading Analysis Report\n\n"
        
        for symbol, analysis in zip(symbols, analyses):
            report += f"## {symbol}\n"
            report += f"- **Action**: {analysis.action}\n"
            report += f"- **Confidence**: {analysis.confidence * 100:.1f}%\n"
            report += f"- **Sentiment**: {analysis.market_sentiment}\n"
            report += f"- **Reasoning**: {analysis.reasoning}\n\n"
            
            if analysis.risks:
                report += "### Risks\n"
                for risk in analysis.risks:
                    report += f"- {risk}\n"
                report += "\n"
            
            if analysis.opportunities:
                report += "### Opportunities\n"
                for opp in analysis.opportunities:
                    report += f"- {opp}\n"
                report += "\n"
            
            report += f"### Price Targets\n"
            report += f"- Short-term: ${analysis.price_targets['short_term']:.2f}\n"
            report += f"- Medium-term: ${analysis.price_targets['medium_term']:.2f}\n"
            report += f"- Long-term: ${analysis.price_targets['long_term']:.2f}\n"
            report += f"- Time Horizon: {analysis.time_horizon}\n\n"
        
        return report


class EnsembleAI:
    """Ensemble of multiple AI models"""
    
    def __init__(self, models: List[AIModel] = None):
        self.models = models or [AIModel.MISTRAL, AIModel.LLAMA2]
        self.brains = [AIBrain(model) for model in self.models]
    
    def analyze_consensus(self, signal_data: Dict) -> Optional[AIAnalysis]:
        """تحلیل با اجماع چند مدل"""
        analyses = []
        
        for brain in self.brains:
            analysis = brain.analyze_signal(signal_data)
            if analysis:
                analyses.append(analysis)
        
        if not analyses:
            return None
        
        # اجماع
        actions = [a.action for a in analyses]
        confidences = [a.confidence for a in analyses]
        sentiments = [a.market_sentiment for a in analyses]
        
        # بیش‌تر رای رفتار
        action = max(set(actions), key=actions.count)
        
        # میانگین confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        # بیش‌تر رای sentiment
        sentiment = max(set(sentiments), key=sentiments.count)
        
        # میانگین price targets
        avg_targets = {
            'short_term': sum(a.price_targets['short_term'] for a in analyses) / len(analyses),
            'medium_term': sum(a.price_targets['medium_term'] for a in analyses) / len(analyses),
            'long_term': sum(a.price_targets['long_term'] for a in analyses) / len(analyses),
        }
        
        return AIAnalysis(
            symbol=signal_data.get('symbol', 'BTC/USDT'),
            action=action,
            reasoning="Consensus analysis from multiple AI models",
            confidence=avg_confidence,
            risks=analyses[0].risks,
            opportunities=analyses[0].opportunities,
            market_sentiment=sentiment,
            price_targets=avg_targets,
            time_horizon=analyses[0].time_horizon
        )
