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