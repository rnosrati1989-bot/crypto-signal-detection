"""AI Models Configuration"""
import os
from pathlib import Path
from pydantic import BaseSettings

class AIModelsConfig(BaseSettings):
    """AI Models Configuration for Llama and Mistral"""
    
    # Model Names
    PRIMARY_MODEL = 'llama-3-8b'  # Primary brain
    SECONDARY_MODEL = 'mistral-7b'  # Secondary brain for validation
    
    # Ollama Configuration
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_TIMEOUT = 120  # Seconds
    
    # Model Paths (if using local files)
    MODELS_DIR = Path(os.getenv('MODELS_DIR', './models'))
    LLAMA_MODEL_PATH = MODELS_DIR / 'llama-3-8b-gguf'
    MISTRAL_MODEL_PATH = MODELS_DIR / 'mistral-7b-gguf'
    
    # Generation Parameters
    GENERATION_PARAMS = {
        'temperature': 0.3,  # Low for consistent analysis
        'top_p': 0.9,
        'top_k': 40,
        'num_predict': 500,
        'repeat_penalty': 1.1,
        'repeat_last_n': 64,
    }
    
    # Context Window
    MAX_TOKENS = 2048
    CONTEXT_WINDOW = 4096
    
    # Prompts Configuration
    PROMPTS = {
        'signal_analysis': '''You are an expert cryptocurrency trading analyst. 
Analyze the following market data and technical indicators.
Provide a clear signal (BUY/SELL/HOLD) with confidence score (0-100).
Explain your reasoning based on:
1. Technical indicators
2. Price patterns
3. Market structure
4. Risk assessment
Format: SIGNAL: [BUY/SELL/HOLD] | CONFIDENCE: [0-100] | REASONING: [...]''',
        
        'pattern_recognition': '''Analyze the following price data and identify:
1. Chart patterns
2. Support/Resistance levels
3. Key price levels
4. Potential breakout points
Provide specific pattern names and confidence levels.''',
        
        'news_impact': '''Given the following news/macro event, assess its impact on cryptocurrency market:
1. Bullish or Bearish?
2. Impact severity (Low/Medium/High)
3. Affected assets
4. Time horizon
Provide structured analysis.''',
        
        'risk_assessment': '''Analyze the risk for this trading setup:
1. Downside risk
2. Upside potential
3. Risk/Reward ratio
4. Stop-loss placement
5. Position sizing recommendation''',
    }
    
    # Model Capabilities
    CAPABILITIES = {
        'llama-3-8b': {
            'speed': 'medium',
            'accuracy': 'high',
            'reasoning': 'excellent',
            'context_length': 8192,
        },
        'mistral-7b': {
            'speed': 'fast',
            'accuracy': 'high',
            'reasoning': 'good',
            'context_length': 32768,
        }
    }
    
    # Ensemble Strategy
    ENSEMBLE_MODE = True  # Use both models for consensus
    CONSENSUS_THRESHOLD = 0.75  # 75% agreement required
    
    class Config:
        env_file = '.env'

ai_models_config = AIModelsConfig()
