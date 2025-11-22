"""Configuration module for Stock Tracker Application.

Contains all application settings, themes, and constants.
"""

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AppConfig:
    """Application configuration with default settings for UI, analysis, and thresholds."""
    min_window_width: int = 1133
    min_window_height: int = 600
    lists_dir: str = "ticker_lists"
    default_list_file: str = "default_list.txt"
    max_threads: int = 3
    price_swing_threshold: float = 5.0
    custom_period_days: int = 30
    trading_aggression: float = 0.5
    recommendation_mode: str = "simple"
    display_currency: str = "USD"  # Currency to display prices in

    # Thresholds for generating buy/sell recommendations (simple mode)
    recommendation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "buy_ma_ratio": 0.97,
        "consider_buy_ma_ratio": 0.99,
        "sell_high_ratio": 0.98,
        "consider_sell_high_ratio": 0.95,
        "pe_high": 30,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_buy": 0,
        "macd_sell": 0,
    })
    
    # Weight factors for complex scoring system
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "ma_position": 0.25,
        "high_position": 0.20,
        "rsi": 0.20,
        "macd": 0.15,
        "pe_ratio": 0.10,
        "analyst_target": 0.10,
    })

    # Toggle individual metrics on/off
    enable_metrics: Dict[str, bool] = field(default_factory=lambda: {
        "pe_ratio": True,
        "analyst_target": True,
        "rsi": True,
        "macd": True,
    })


# Global configuration instance
CONFIG = AppConfig()

# Ensure lists directory exists
os.makedirs(CONFIG.lists_dir, exist_ok=True)


# Theme definitions
THEMES = {
    "dark": {
        "background": "#1e1e1e",
        "text": "#e0e0e0",
        "entry": "#2d2d2d",
        "button": "#3a3a3a",
        "tree_bg": "#2d2d2d",
        "tree_fg": "#e0e0e0",
        "tree_heading_bg": "#3a3a3a",
        "tag_sell": "#663333",
        "tag_consider_sell": "#666633",
        "tag_buy": "#336633",
        "tag_consider_buy": "#336666",
        "tag_hold": "#2d2d2d",
    }
}


# Ticker suffix mappings for different exchanges
TICKER_SUFFIX_MAP = {
    ".ST": ".ST", ".STO": "", ".MI": ".MI", ".DE": ".DE",
    ".L": ".L", ".PA": ".PA", ".T": ".T", ".HK": ".HK",
    ".SS": ".SS", ".SZ": ".SZ", ".TO": ".TO", ".AX": ".AX",
    ".NS": ".NS", ".BO": ".BO",
}


# Currency conversion rates (updated periodically)
# Base currency: USD
CURRENCY_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.50,
    "CNY": 7.24,
    "SEK": 10.35,
    "NOK": 10.85,
    "DKK": 6.87,
    "CHF": 0.88,
    "CAD": 1.39,
    "AUD": 1.53,
    "NZD": 1.67,
    "INR": 83.12,
    "BRL": 4.97,
    "MXN": 17.12,
    "ZAR": 18.45,
    "KRW": 1320.50,
    "SGD": 1.34,
    "HKD": 7.81,
    "RUB": 92.50,
}


# Currency symbols
CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "SEK": "SEK",
    "NOK": "NOK",
    "DKK": "DKK",
    "CHF": "CHF",
    "CAD": "CAD",
    "AUD": "AUD",
    "NZD": "NZD",
    "INR": "INR",
    "BRL": "BRL",
    "MXN": "MXN",
    "ZAR": "ZAR",
    "KRW": "KRW",
    "SGD": "SGD",
    "HKD": "HKD",
    "RUB": "RUB",
}


# Currency formatting preferences (True = symbol after amount)
CURRENCY_SYMBOL_AFTER = {
    "USD": False,  # $100
    "EUR": False,  # €100
    "GBP": False,  # £100
    "JPY": False,  # ¥100
    "CNY": False,  # ¥100
    "SEK": True,   # 100 SEK
    "NOK": True,   # 100 NOK
    "DKK": True,   # 100 DKK
    "CHF": True,   # 100 CHF
    "CAD": False,  # CAD100
    "AUD": False,  # AUD100
    "NZD": False,  # NZD100
    "INR": False,  # ₹100
    "BRL": False,  # BRL100
    "MXN": False,  # MXN100
    "ZAR": False,  # ZAR100
    "KRW": False,  # ₩100
    "SGD": False,  # SGD100
    "HKD": False,  # HKD100
    "RUB": False,  # ₽100
}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """
    Convert amount from one currency to another.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "EUR")
        
    Returns:
        Converted amount
    """
    if from_currency == to_currency:
        return amount
    
    # Convert to USD first if needed
    if from_currency != "USD":
        amount_usd = amount / CURRENCY_RATES.get(from_currency, 1.0)
    else:
        amount_usd = amount
    
    # Convert from USD to target currency
    return amount_usd * CURRENCY_RATES.get(to_currency, 1.0)


def get_currency_symbol(currency: str) -> str:
    """Get the symbol for a currency code."""
    return CURRENCY_SYMBOLS.get(currency, currency)


def format_price(price: float, currency: str) -> str:
    """Format a price with the appropriate currency symbol and position."""
    symbol = get_currency_symbol(currency)
    symbol_after = CURRENCY_SYMBOL_AFTER.get(currency, False)
    
    # For currencies with large values (JPY, KRW), don't show decimals
    if currency in ["JPY", "KRW"]:
        formatted_amount = f"{price:,.0f}"
    else:
        formatted_amount = f"{price:,.2f}"
    
    # Place symbol before or after based on currency convention
    if symbol_after:
        return f"{formatted_amount} {symbol}"
    else:
        return f"{symbol}{formatted_amount}"
