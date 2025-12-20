"""
Simple cost calculator for LLM models with persistence.
"""
import json
import os
from pathlib import Path

# Default Prices in USD per 1M tokens (Input / Output)
DEFAULT_PRICES = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    
    # Anthropic
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-opus-20240229": (15.00, 75.00),
    
    # Google
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}

CACHE_FILE = Path.home() / ".agentic_browser_costs.json"

def load_prices() -> dict:
    """Load prices from cache or return defaults."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                saved = json.load(f)
                # Merge with defaults to ensure new models exist
                prices = DEFAULT_PRICES.copy()
                # Convert list back to tuple if needed (JSON stores tuples as lists)
                for k, v in saved.items():
                    prices[k] = tuple(v)
                return prices
        except Exception as e:
            print(f"Error loading custom costs: {e}")
    return DEFAULT_PRICES.copy()

def save_prices(prices: dict):
    """Save prices to cache."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(prices, f, indent=2)
    except Exception as e:
        print(f"Error saving costs: {e}")

# Global prices
CURRENT_PRICES = load_prices()

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost for a request."""
    # Reload in case GUI updated it (simple way to sync)
    # Optimization: in a real app, use IPC or file watch, but for now reload is safe for low freq
    # implementation: we won't reload every call for perf, but we can rely on import reload or shared memory
    # Actually, for separate process (GUI vs Agent), file is the sync point. 
    # The agent runs in a subprocess. It loads once at start.
    # PROVISION: The agent needs to reload if changed? 
    # Agent is short-lived or we can reload here. Let's reload if file mtime changed?
    # For simplicity, we'll reload every call (costs are low freq, usually 1 call per few seconds).
    
    global CURRENT_PRICES
    # Minimal check for file update could go here, but let's just use what we have for now.
    # The user changing cost settings usually happens BEFORE run.
    # If done during run, agent might not see it until restart. That is acceptable for V1.
    
    model = model_name.lower() if model_name else ""
    features = None
    
    # Exact match
    if model in CURRENT_PRICES:
        features = CURRENT_PRICES[model]
    else:
        # Prefix match
        for key, price in CURRENT_PRICES.items():
            if key in model or model in key:
                features = price
                break
    
    if not features:
        features = (2.50, 10.00)  # Default fallback
        
    input_cost = (input_tokens / 1_000_000) * features[0]
    output_cost = (output_tokens / 1_000_000) * features[1]
    
    return input_cost + output_cost

