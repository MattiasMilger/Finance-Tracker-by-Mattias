"""Stock Tracker Application using Tkinter and yFinance for real-time stock data analysis.

This application provides a GUI for tracking stock portfolios with real-time data fetching,
technical analysis metrics (RSI, MACD, P/E ratios), and customizable recommendations based
on moving averages and 52-week highs.
"""
import json
import logging
import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog
import yfinance as yf

from stock_search import open_stock_search


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    """Application configuration with default settings for UI, analysis, and thresholds."""
    min_window_width: int = 1063
    min_window_height: int = 600
    lists_dir: str = "ticker_lists"
    default_list_file: str = "default_list.txt"
    max_threads: int = 3  # Number of concurrent threads for fetching stock data
    price_swing_threshold: float = 5.0  # Minimum % change to flag in reasons
    custom_period_days: int = 30  # Default lookback period for historical analysis
    trading_aggression: float = 0.5  # 0.0 = conservative, 1.0 = aggressive
    recommendation_mode: str = "simple"  # "simple" or "complex"

    # Thresholds for generating buy/sell recommendations (simple mode)
    recommendation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "buy_ma_ratio": 0.97,  # Buy if price < 97% of 50-day MA
        "consider_buy_ma_ratio": 0.99,  # Consider buying if price < 99% of 50-day MA
        "sell_high_ratio": 0.98,  # Sell if price > 98% of 52-week high
        "consider_sell_high_ratio": 0.95,  # Consider selling if price > 95% of 52-week high
        "pe_high": 30,  # P/E ratio threshold for overvaluation warning
        "rsi_overbought": 70,  # RSI threshold for overbought condition
        "rsi_oversold": 30,  # RSI threshold for oversold condition
        "macd_buy": 0,  # MACD crossover buy signal threshold
        "macd_sell": 0,  # MACD crossover sell signal threshold
    })
    
    # Weight factors for complex scoring system
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "ma_position": 0.25,      # Weight for 50-day MA position
        "high_position": 0.20,    # Weight for 52-week high position
        "rsi": 0.20,              # Weight for RSI indicator
        "macd": 0.15,             # Weight for MACD momentum
        "pe_ratio": 0.10,         # Weight for P/E valuation
        "analyst_target": 0.10,   # Weight for analyst price target
    })

    # Toggle individual metrics on/off
    enable_metrics: Dict[str, bool] = field(default_factory=lambda: {
        "pe_ratio": True,
        "analyst_target": True,
        "rsi": True,
        "macd": True,
    })


CONFIG = AppConfig()
os.makedirs(CONFIG.lists_dir, exist_ok=True)


# --------------------------------------------------------------------------- #
# Theme & Constants
# --------------------------------------------------------------------------- #
THEMES = {
    "dark": {
        "background": "#1e1e1e",
        "text": "#e0e0e0",
        "entry": "#2d2d2d",
        "button": "#3a3a3a",
        "tree_bg": "#2d2d2d",
        "tree_fg": "#e0e0e0",
        "tree_heading_bg": "#3a3a3a",
        "tag_sell": "#663333",  # Red background for sell recommendations
        "tag_consider_sell": "#666633",  # Yellow-brown for consider selling
        "tag_buy": "#336633",  # Green background for buy recommendations
        "tag_consider_buy": "#336666",  # Blue-green for consider buying
        "tag_hold": "#2d2d2d",  # Default background for hold
    }
}

# Maps various ticker suffix formats to normalized versions
# Different exchanges use different suffixes (e.g., .ST for Stockholm, .L for London)
TICKER_SUFFIX_MAP = {
    ".ST": ".ST", ".STO": "", ".MI": ".MI", ".DE": ".DE",
    ".L": ".L", ".PA": ".PA", ".T": ".T", ".HK": ".HK",
    ".SS": ".SS", ".SZ": ".SZ", ".TO": ".TO", ".AX": ".AX",
    ".NS": ".NS", ".BO": ".BO",
}

# Configure logging to file for debugging and error tracking
logging.basicConfig(
    level=logging.INFO,
    filename="stock_tracker.log",
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def log_and_show(title: str, message: str, func_name: str, ticker: Optional[str] = None, msg_type: str = "error") -> None:
    """Log an error/warning and display a messagebox to the user.
    
    Args:
        title: Dialog box title
        message: Error message to display
        func_name: Name of function where error occurred
        ticker: Optional ticker symbol for context
        msg_type: Type of messagebox ('error', 'warning', 'info')
    """
    log_msg = f"{title}: {message}"
    if ticker:
        log_msg += f" (Ticker: {ticker})"
    log_msg += f" in {func_name}"
    logging.error(log_msg)
    getattr(messagebox, f"show{msg_type}")(title, message)


def load_ticker_list(filename: str) -> Tuple[List[str], float, int, str]:
    """Load a list of ticker symbols from a JSON file.
    
    Args:
        filename: Path to the JSON file containing ticker list
        
    Returns:
        Tuple of (list of ticker symbols, trading aggression value, custom period days, recommendation mode)
    """
    if not os.path.exists(filename):
        return [], 0.5, 30, "simple"
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both old format (list) and new format (dict with tickers and aggression)
        if isinstance(data, list):
            return [str(t).strip().upper() for t in data if str(t).strip()], 0.5, 30, "simple"
        else:
            tickers = [str(t).strip().upper() for t in data.get("tickers", []) if str(t).strip()]
            aggression = data.get("trading_aggression", 0.5)
            custom_period = data.get("custom_period_days", 30)
            rec_mode = data.get("recommendation_mode", "simple")
            return tickers, aggression, custom_period, rec_mode
    except Exception as e:
        log_and_show("Load Error", f"Failed to load ticker list: {e}", "load_ticker_list", msg_type="warning")
        return [], 0.5, 30, "simple"


def save_ticker_list(filename: str, tickers: List[str], aggression: float = 0.5, 
                     custom_period_days: int = 30, recommendation_mode: str = "simple") -> None:
    """Save a list of ticker symbols to a JSON file.
    
    Args:
        filename: Path where the JSON file should be saved
        tickers: List of ticker symbols to save
        aggression: Trading aggression level (0.0 to 1.0)
        custom_period_days: Custom lookback period in days
        recommendation_mode: Recommendation system mode ("simple" or "complex")
    """
    try:
        data = {
            "tickers": tickers,
            "trading_aggression": aggression,
            "custom_period_days": custom_period_days,
            "recommendation_mode": recommendation_mode
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log_and_show("Save Error", f"Failed to save ticker list: {e}", "save_ticker_list")


def export_to_csv(stock_data: List[Dict[str, Any]]) -> None:
    """Export stock data to CSV with Save As dialog.
    
    Flattens the nested stock data structure and exports all relevant fields
    including ticker info, metrics, and recommendations to a CSV file.
    
    Args:
        stock_data: List of dictionaries containing stock information
    """
    if not stock_data:
        messagebox.showwarning("Export Error", "No stock data to export.")
        return

    # Open Save As dialog with default filename based on current date/time
    default_name = f"stock_data_{datetime.now():%Y%m%d_%H%M%S}.csv"
    file_path = filedialog.asksaveasfilename(
        title="Export Data as CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=default_name
    )
    if not file_path:
        return  # User cancelled

    try:
        # Flatten nested dictionary structure for CSV export
        flat = []
        for d in stock_data:
            flat.append({
                "ticker": d["ticker"],
                "name": d["name"],
                "sector": d["sector"],
                "industry": d["industry"],
                "recommendation": d["recommendation"],
                "reasons": ", ".join(d["reasons"]),
                "1_day_%": d.get("price_swing_1d", "N/A"),
                f"{CONFIG.custom_period_days}_day_%": d.get("price_swing_1m", "N/A"),
                # Include relevant price fields from yfinance info dict
                **{k: v for k, v in d["info"].items() if k in (
                    "regularMarketPrice", "previousClose", "fiftyTwoWeekHigh",
                    "fiftyTwoWeekLow", "trailingPE", "fiftyDayAverage"
                )},
                **d["metrics"]
            })
        df = pd.DataFrame(flat)
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", f"Data exported successfully!\nSaved to:\n{file_path}")
    except Exception as e:
        log_and_show("Export Error", f"Failed to export to CSV: {e}", "export_to_csv")


# --------------------------------------------------------------------------- #
# Metric Functions
# --------------------------------------------------------------------------- #
def get_pe_ratio(info: dict) -> Tuple[Optional[float], None]:
    """Extract trailing P/E ratio from stock info.
    
    Args:
        info: Stock info dictionary from yfinance
        
    Returns:
        Tuple of (P/E ratio, None) - second value kept for consistency with other metrics
    """
    return info.get("trailingPE"), None


def get_analyst_target(info: dict, price: float) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Calculate analyst target price and percentage difference from current price.
    
    Args:
        info: Stock info dictionary from yfinance
        price: Current stock price
        
    Returns:
        Tuple of (target price, % difference, flag message)
    """
    target = info.get("targetMeanPrice")
    if target and price:
        diff = (target - price) / price * 100
        return target, diff, "Potential Upside" if price < target else None
    return None, None, None


def get_rsi(hist: pd.DataFrame, period: int = 14) -> Tuple[Optional[float], Optional[str]]:
    """Calculate Relative Strength Index (RSI) - momentum oscillator.
    
    RSI measures the speed and magnitude of price changes. Values above 70
    indicate overbought conditions, below 30 indicate oversold conditions.
    
    Args:
        hist: Historical price data DataFrame with 'Close' column
        period: Lookback period for RSI calculation (default: 14 days)
        
    Returns:
        Tuple of (RSI value, flag message if overbought/oversold)
    """
    if hist.empty or "Close" not in hist:
        return None, None
    
    # Calculate price changes (deltas)
    delta = hist["Close"].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # Calculate average gain and loss over the period
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    
    if avg_loss == 0:
        return 100.0, "Overbought"
    
    # RSI formula: 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Flag extreme conditions based on thresholds
    flag = (
        "Overbought" if rsi > CONFIG.recommendation_thresholds["rsi_overbought"]
        else "Oversold" if rsi < CONFIG.recommendation_thresholds["rsi_oversold"] else None
    )
    return rsi, flag


def get_macd(hist: pd.DataFrame, short: int = 12, long: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float]]:
    """Calculate MACD (Moving Average Convergence Divergence) - trend-following indicator.
    
    MACD shows the relationship between two moving averages of prices. A positive
    MACD indicates upward momentum, negative indicates downward momentum.
    
    Args:
        hist: Historical price data DataFrame with 'Close' column
        short: Short-term EMA period (default: 12)
        long: Long-term EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
        
    Returns:
        Tuple of (MACD line value, signal line value)
    """
    if hist.empty or "Close" not in hist:
        return None, None
    
    close = hist["Close"]
    
    # Calculate exponential moving averages
    ema_s = close.ewm(span=short, adjust=False).mean()
    ema_l = close.ewm(span=long, adjust=False).mean()
    
    # MACD line is the difference between short and long EMAs
    macd_line = ema_s - ema_l
    
    # Signal line is the EMA of the MACD line
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line.iloc[-1], sig_line.iloc[-1]


class MetricRegistry:
    """Registry pattern for metric calculation functions.
    
    Allows for dynamic registration and computation of various stock metrics
    without hardcoding function calls throughout the application.
    """
    def __init__(self) -> None:
        self._metrics: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        """Register a metric calculation function.
        
        Args:
            name: Unique identifier for the metric
            func: Callable that computes the metric
        """
        self._metrics[name] = func

    def compute(self, name: str, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
        """Compute a registered metric by name.
        
        Args:
            name: Metric identifier
            *args: Positional arguments for the metric function
            **kwargs: Keyword arguments for the metric function
            
        Returns:
            Result from the metric function, or (None, None) if not found
        """
        return self._metrics.get(name, lambda *_, **__: (None, None))(*args, **kwargs)


# Initialize and register all available metrics
registry = MetricRegistry()
registry.register("pe_ratio", get_pe_ratio)
registry.register("analyst_target", get_analyst_target)
registry.register("rsi", get_rsi)
registry.register("macd", get_macd)


# --------------------------------------------------------------------------- #
# Main Application
# --------------------------------------------------------------------------- #
class StockTrackerApp:
    """Main application class for the Stock Tracker GUI.
    
    Manages the user interface, stock data fetching, analysis, and display
    of recommendations and metrics in a sortable table format.
    """
    
    def __init__(self, root: tk.Tk, theme: str = "dark") -> None:
        """Initialize the Stock Tracker application.
        
        Args:
            root: Tkinter root window
            theme: Color theme name (currently only 'dark' supported)
        """
        self.root = root
        self.root.title("Stock Tracker by Mattias")
        self.root.minsize(CONFIG.min_window_width, CONFIG.min_window_height)
        self.theme = THEMES[theme]
        self.root.configure(bg=self.theme["background"])

        # Application state
        self.button_refs: Dict[str, tk.Button] = {}  # References to buttons for enable/disable
        self.stock_data: List[Dict[str, Any]] = []  # Fetched stock data with metrics
        self.current_tickers: List[str] = []  # List of tickers in current view
        self.current_list_name: str = ""  # Path to currently loaded list file
        self.unsaved_changes: bool = False  # Track if list has unsaved modifications
        self.filter_query: str = ""  # Current filter/search query
        self.trading_aggression: float = 0.5  # Trading aggression level (0.0-1.0)
        self.last_updated: Optional[str] = None  # Timestamp of last data fetch
        self.recommendation_mode: str = "simple"  # Recommendation mode: "simple" or "complex"

        # Sorting state
        self._sort_column: str = "Recommendation"
        self._sort_reverse: bool = False

        # UI element references
        self.rows = []  # List of row dictionaries for the table
        self.header_labels = []  # Column header label widgets
        self.search_entry: Optional[tk.Entry] = None

        # Build the UI and load default list
        self._setup_menu()
        self._setup_ui()
        self._load_default_list(silent=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_menu(self) -> None:
        """Create the menu bar with File, Trading Style, Period, and Help menus."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu - list management and export
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New List", command=self.new_list)
        file_menu.add_command(label="Open List", command=self.open_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Save List", command=self.save_current_list)
        file_menu.add_command(label="Save List as", command=self.save_list_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data As CSV", command=lambda: export_to_csv(self.stock_data))
        file_menu.add_separator()
        file_menu.add_command(label="Set List as Default", command=self.set_as_default)
        file_menu.add_command(label="Remove Default List", command=self.remove_default)

        # Trading Style menu
        trading_style_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Trading Style", menu=trading_style_menu)
        
        self.trading_style_var = tk.StringVar(value="Moderate")
        trading_styles = [
            ("Very Conservative", 0.1),
            ("Conservative", 0.3),
            ("Moderate", 0.5),
            ("Aggressive", 0.7),
            ("Very Aggressive", 0.9)
        ]
        
        for style_name, aggr_value in trading_styles:
            trading_style_menu.add_radiobutton(
                label=style_name,
                variable=self.trading_style_var,
                value=style_name,
                command=lambda v=aggr_value, n=style_name: self.set_trading_style(v, n)
            )
        
        # Add separator and recommendation mode submenu
        trading_style_menu.add_separator()
        rec_mode_menu = tk.Menu(trading_style_menu, tearoff=0)
        trading_style_menu.add_cascade(label="Recommendation System", menu=rec_mode_menu)
        
        self.rec_mode_var = tk.StringVar(value="simple")
        rec_mode_menu.add_radiobutton(
            label="Simple (Legacy)",
            variable=self.rec_mode_var,
            value="simple",
            command=lambda: self.set_recommendation_mode("simple")
        )
        rec_mode_menu.add_radiobutton(
            label="Complex (Multi-Factor Scoring)",
            variable=self.rec_mode_var,
            value="complex",
            command=lambda: self.set_recommendation_mode("complex")
        )

        # Period menu - configure lookback period
        period_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Period", menu=period_menu)
        period_menu.add_command(label="Set Custom Period (Days)...", command=self.set_custom_period)

        # Help menu - metric explanations
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Info – Metric Explanations", command=self.show_info_popup)

    def _setup_ui(self) -> None:
        """Create and layout all UI components."""
        # Main container frame
        mgmt = tk.LabelFrame(
            self.root, text="Ticker List & Analysis",
            bg=self.theme["background"], fg=self.theme["text"], padx=10, pady=5
        )
        mgmt.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Top control bar
        top_frame = tk.Frame(mgmt, bg=self.theme["background"])
        top_frame.pack(fill=tk.X, pady=(0, 5))

        # Current list name display
        name_frame = tk.Frame(top_frame, bg=self.theme["background"])
        name_frame.pack(side=tk.LEFT)
        tk.Label(name_frame, text="Current List:", bg=self.theme["background"], fg=self.theme["text"]).pack(side=tk.LEFT)
        self.list_name_lbl = tk.Label(
            name_frame, text="(none)", bg=self.theme["background"],
            fg=self.theme["text"], font=("Arial", 9, "italic")
        )
        self.list_name_lbl.pack(side=tk.LEFT, padx=(5, 0))

        # Fetch Data button (left side, prominent position)
        fetch_frame = tk.Frame(top_frame, bg=self.theme["background"])
        fetch_frame.pack(side=tk.LEFT, padx=(20, 0))
        fetch_btn = tk.Button(
            fetch_frame, text="Fetch Data", command=self.fetch_and_display,
            bg=self.theme["button"], fg=self.theme["text"], font=("Arial", 11, "bold"), width=15, height=1
        )
        fetch_btn.pack()
        self.button_refs["Fetch Data"] = fetch_btn

        # Add/Remove buttons
        add_frame = tk.Frame(top_frame, bg=self.theme["background"])
        add_frame.pack(side=tk.LEFT, padx=(20, 0))
        tk.Button(
            add_frame, text="Search & Add Stocks", command=self.open_search_dialog,
            bg=self.theme["button"], fg=self.theme["text"], width=20
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            add_frame, text="Remove Stocks", command=self.remove_selected,
            bg=self.theme["button"], fg=self.theme["text"], width=15
        ).pack(side=tk.LEFT, padx=2)
        
        # Search/Filter controls
        search_frame = tk.Frame(top_frame, bg=self.theme["background"])
        search_frame.pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(search_frame, text="Filter (Ticker/Name):", bg=self.theme["background"], fg=self.theme["text"]).pack(side=tk.LEFT)
        self.search_entry = tk.Entry(
            search_frame, width=20, bg=self.theme["entry"], fg=self.theme["text"],
            insertbackground=self.theme["text"]
        )
        self.search_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.search_entry.bind("<Return>", lambda e: self.filter_list(self.search_entry.get()))
        tk.Button(
            search_frame, text="Filter", command=lambda: self.filter_list(self.search_entry.get()),
            bg=self.theme["button"], fg=self.theme["text"]
        ).pack(side=tk.LEFT, padx=5)

        # Scrollable canvas for stock table
        table_container = tk.Frame(mgmt, bg=self.theme["background"])
        table_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(table_container, bg=self.theme["tree_bg"], highlightthickness=0)
        v_scroll = tk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scroll = tk.Scrollbar(mgmt, orient=tk.HORIZONTAL, command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Frame inside canvas to hold the table
        self.table_frame = tk.Frame(self.canvas, bg=self.theme["tree_bg"])
        self.canvas.create_window((0, 0), window=self.table_frame, anchor="nw")

        # Update scroll region when table changes size
        self.table_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Mouse wheel scrolling support
        def _on_mousewheel(event):
            self.canvas.yview_scroll(-1*(event.delta//120), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

        # Table header row
        self.header_frame = tk.Frame(self.table_frame, bg=self.theme["tree_heading_bg"])
        self.header_frame.pack(fill=tk.X)

        headers = [
            "Ticker", "Name", "Recommendation", "Price",
            "1 Day %", f"{CONFIG.custom_period_days} Day %",
            "Sector", "Industry", "Volume", "P/E", "Target %", "RSI", "MACD"
        ]
        # Column widths in characters - increased for Name, Sector, Industry to prevent truncation
        char_widths = [10, 30, 16, 10, 10, 12, 22, 25, 12, 8, 10, 8, 10]

        self.header_labels = []
        for i, (text, width) in enumerate(zip(headers, char_widths)):
            lbl = tk.Label(
                self.header_frame,
                text=text,
                bg=self.theme["tree_heading_bg"],
                fg=self.theme["text"],
                font=("Consolas", 10, "bold"),
                width=width,
                anchor="center",
                relief="raised",
                bd=1
            )
            lbl.grid(row=0, column=i, padx=0, pady=0, sticky="nsew")
            # Click header to sort by that column
            lbl.bind("<Button-1>", lambda e, col=text: self._sort_by_column(col))
            self.header_labels.append(lbl)
        
        # Configure columns to expand proportionally
        for i in range(len(headers)):
            self.header_frame.grid_columnconfigure(i, weight=1)

        # Bottom control bar with exit button
        bottom = tk.Frame(self.root, bg=self.theme["background"])
        bottom.pack(pady=10)
        exit_btn = tk.Button(bottom, text="Exit", command=self._on_closing, width=15,
                             bg=self.theme["button"], fg=self.theme["text"])
        exit_btn.pack()
        self.button_refs["Exit"] = exit_btn

        # Status label at bottom
        self.status_lbl = tk.Label(
            self.root, text="Ready", bg=self.theme["background"],
            fg=self.theme["text"], font=("Arial", 10)
        )
        self.status_lbl.pack(pady=(0, 2))
        
        # Last updated label
        self.last_updated_lbl = tk.Label(
            self.root, text="", bg=self.theme["background"],
            fg="#888888", font=("Arial", 8, "italic")
        )
        self.last_updated_lbl.pack(pady=(0, 5))

    def set_recommendation_mode(self, mode: str) -> None:
        """Set recommendation system mode (simple or complex).
        
        Args:
            mode: "simple" for legacy recommendations, "complex" for multi-factor scoring
        """
        self.recommendation_mode = mode
        self.rec_mode_var.set(mode)
        self.unsaved_changes = True
        
        mode_display = "Simple (Legacy)" if mode == "simple" else "Complex (Multi-Factor Scoring)"
        
        # If we have stock data, automatically fetch to update recommendations
        if self.stock_data and self.current_tickers:
            self.status_lbl.config(text=f"Recommendation system changed to {mode_display} - Updating...")
            self.root.update_idletasks()
            self.fetch_and_display()
        else:
            self.status_lbl.config(text=f"Recommendation system set to {mode_display}")

    def set_trading_style(self, aggression: float, style_name: str) -> None:
        """Set trading style from menu and update recommendations.
        
        Args:
            aggression: Trading aggression value (0.0 to 1.0)
            style_name: Display name of the trading style
        """
        self.trading_aggression = aggression
        self.trading_style_var.set(style_name)
        self.unsaved_changes = True
        
        # If we have stock data, automatically fetch to update recommendations
        if self.stock_data and self.current_tickers:
            self.status_lbl.config(text=f"Trading style changed to {style_name} - Updating recommendations...")
            self.root.update_idletasks()
            self.fetch_and_display()
        else:
            self.status_lbl.config(text=f"Trading style set to {style_name}")

    def _get_aggression_label(self, aggression: float) -> str:
        """Get descriptive label for aggression level.
        
        Args:
            aggression: Trading aggression value (0.0 to 1.0)
            
        Returns:
            Descriptive label string
        """
        if aggression <= 0.2:
            return "Very Conservative"
        elif aggression <= 0.4:
            return "Conservative"
        elif aggression <= 0.6:
            return "Moderate"
        elif aggression <= 0.8:
            return "Aggressive"
        else:
            return "Very Aggressive"
    
    def _on_aggression_change(self, value: str) -> None:
        """Handle trading aggression change - DEPRECATED, kept for compatibility.
        
        Args:
            value: New value as string
        """
        pass

    def _get_adjusted_thresholds(self) -> Dict[str, float]:
        """Calculate adjusted recommendation thresholds based on trading aggression.
        
        Conservative (0.0): Fewer trades, wider margins
        Aggressive (1.0): More trades, tighter margins
        
        Returns:
            Dictionary of adjusted threshold values
        """
        aggr = self.trading_aggression
        base = CONFIG.recommendation_thresholds.copy()
        
        # Adjust buy thresholds (lower aggression = wait for better deals)
        # Conservative: 0.95, Moderate: 0.97, Aggressive: 0.99
        base["buy_ma_ratio"] = 0.95 + (aggr * 0.04)
        base["consider_buy_ma_ratio"] = 0.97 + (aggr * 0.04)
        
        # Adjust sell thresholds (lower aggression = hold longer)
        # Conservative: 0.99, Moderate: 0.98, Aggressive: 0.96
        base["sell_high_ratio"] = 0.99 - (aggr * 0.03)
        base["consider_sell_high_ratio"] = 0.97 - (aggr * 0.04)
        
        # Adjust RSI thresholds (tighter ranges for aggressive trading)
        # Conservative: 75/25, Moderate: 70/30, Aggressive: 65/35
        base["rsi_overbought"] = 75 - (aggr * 10)
        base["rsi_oversold"] = 25 + (aggr * 10)
        
        return base
        
    def filter_list(self, query: str) -> None:
        """Filter displayed stocks by ticker symbol or company name.
        
        Args:
            query: Search string to match against ticker/name (case-insensitive)
        """
        self.filter_query = query.strip().upper()
        if self.search_entry:
            if not self.filter_query:
                self.search_entry.delete(0, tk.END)
                self.status_lbl.config(text="Filter cleared. Showing all stocks.")
            else:
                self.status_lbl.config(text=f"Filtered by: '{query}'")

        # Re-sort with current settings to maintain order
        self._sort_by_column(self._sort_column) 

        # Show/hide rows based on filter match
        filtered_count = 0
        for row in self.rows:
            ticker = row['ticker']
            name = row['data'].get('name', 'N/A') if row['data'] else 'N/A'
            
            # Check if filter query appears in ticker or name
            if not self.filter_query or self.filter_query in ticker or self.filter_query in name.upper():
                row["frame"].pack(fill=tk.X, pady=1)
                filtered_count += 1
            else:
                row["frame"].pack_forget()

        if self.filter_query:
            self.status_lbl.config(text=f"Showing {filtered_count}/{len(self.current_tickers)} stock(s) matching '{query}'.")

    def _update_sort_indicator(self):
        """Update header labels to show current sort column and direction."""
        for lbl in self.header_labels:
            base = lbl.cget("text").split(" ")[0]
            # Check if this is the sorted column
            if base == self._sort_column or (base.endswith("Day") and self._sort_column in ["1 Day %", f"{CONFIG.custom_period_days} Day %"]):
                arrow = "Down" if self._sort_reverse else "Up"
                lbl.config(text=base + " " + arrow)
            else:
                lbl.config(text=base)

    def _sort_by_column(self, col_text: str):
        """Sort the displayed rows by a specific column.
        
        Clicking the same column twice reverses the sort order.
        
        Args:
            col_text: Header text of the column to sort by
        """
        # Normalize column name (remove sort indicators)
        col_text = col_text.split(" ")[0]
        if col_text.endswith("Day"):
            col_text = "1 Day %" if "1" in col_text else f"{CONFIG.custom_period_days} Day %"

        # Toggle sort direction if clicking same column
        if self._sort_column == col_text:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = col_text
            self._sort_reverse = False

        self._update_sort_indicator()

        # Find column index from headers
        col_idx = next((i for i, h in enumerate(self.header_labels) if h.cget("text").split(" ")[0] == col_text), None)
        if col_idx is None:
            return
            
        # Special sorting logic for recommendation column (priority order)
        if self._sort_column == "Recommendation":
            priority_map = {
                "Sell": 0,
                "Consider Selling": 1, 
                "Buy": 2,
                "Consider Buying": 3,
                "Hold": 4, 
            }
            def rec_key_func(row):
                data = row["data"]
                if not data: return (5, "Z")
                rec = data.get("recommendation", "Unknown")
                return (priority_map.get(rec, 5), rec)
                
            self.rows.sort(key=rec_key_func, reverse=self._sort_reverse)
        else:
            # Generic sorting for other columns
            def key_func(row):
                val = row["labels"][col_idx].cget("text")
                # Handle missing/error values (sort to end)
                if val in ("N/A", "", "Error"):
                    return (1, 0)
                # Parse percentage values as floats
                if "%" in val:
                    try:
                        return (0, float(val.replace("%", "").replace("+", "").replace(" ", "")))
                    except:
                        return (1, 0)
                # Try numeric sort, fall back to string sort
                try:
                    return (0, float(val))
                except:
                    return (1, val.lower())

            self.rows.sort(key=key_func, reverse=self._sort_reverse)
        
        # Re-pack all rows in new order
        for row in self.rows:
            row["frame"].pack_forget()

        # Only show rows matching current filter
        for row in self.rows:
            ticker = row['ticker']
            name = row['data'].get('name', 'N/A') if row['data'] else 'N/A'
            if not self.filter_query or self.filter_query in ticker or self.filter_query in name.upper():
                row["frame"].pack(fill=tk.X, pady=1)

    def _update_list_display(self) -> None:
        """Rebuild the entire table display with current stock data.
        
        This method clears and recreates all row widgets based on current_tickers
        and stock_data. Rows are color-coded by recommendation and sorted by priority.
        """
        # Clear existing rows
        for row in self.rows:
            row["frame"].destroy()
        self.rows.clear()
        self.filter_query = ""
        if self.search_entry:
            self.search_entry.delete(0, tk.END)

        if not self.current_tickers:
            self.status_lbl.config(text="Ready")
            return

        # Build display data with recommendation priorities
        display_data = []
        for ticker in self.current_tickers:
            data = next((d for d in self.stock_data if d["ticker"] == ticker), None)
            priority = 6  # Default for missing data
            if data:
                # Priority order: Sell (0), Consider Selling (1), Buy (2), Consider Buying (3), Hold (4)
                priority = {
                    "Sell": 0,
                    "Consider Selling": 1,
                    "Buy": 2,            
                    "Consider Buying": 3,
                    "Hold": 4,
                }.get(data["recommendation"], 5)
            display_data.append((ticker, data, priority))

        # Sort by recommendation priority (most urgent first)
        display_data.sort(key=lambda x: x[2]) 

        # Column widths in characters - increased for Name, Sector, Industry to prevent truncation
        char_widths = [10, 30, 16, 10, 12, 10, 12, 22, 25, 8, 10, 8, 10]

        # Create a row for each ticker
        for ticker, data, _ in display_data:
            # Determine background color based on recommendation
            bg_color = self.theme["tag_hold"]
            if data:
                bg_color = {
                    "Sell": self.theme["tag_sell"],
                    "Consider Selling": self.theme["tag_consider_sell"],
                    "Buy": self.theme["tag_buy"],
                    "Consider Buying": self.theme["tag_consider_buy"],
                    "Hold": self.theme["tag_hold"]
                }.get(data["recommendation"], self.theme["tag_hold"])

            frame = tk.Frame(self.table_frame, bg=bg_color)
            frame.pack(fill=tk.X, pady=1)

            # Prepare values for all columns
            labels = []
            
            # Format volume for readability (K = thousands, M = millions, B = billions)
            volume_str = ""
            if data and data['info'].get('volume'):
                vol = data['info'].get('volume')
                if vol >= 1_000_000_000:
                    volume_str = f"{vol/1_000_000_000:.2f}B"
                elif vol >= 1_000_000:
                    volume_str = f"{vol/1_000_000:.2f}M"
                elif vol >= 1_000:
                    volume_str = f"{vol/1_000:.2f}K"
                else:
                    volume_str = str(vol)
            
            values = [
                ticker,
                data["name"] if data else "",
                data["recommendation"] if data else "",
                f"{data['info'].get('regularMarketPrice', ''):.2f}" if data and data['info'].get('regularMarketPrice') is not None else "",
                data.get("price_swing_1d", "N/A") if data else "N/A",
                data.get("price_swing_1m", "N/A") if data else "N/A",
                data["sector"] if data else "",
                data["industry"] if data else "",
                volume_str,
                f"{data['metrics'].get('P/E', ''):.1f}" if data and data['metrics'].get('P/E') else "",
                f"{data['metrics'].get('Target %', ''):+.1f}%" if data and data['metrics'].get('Target %') else "",
                f"{data['metrics'].get('RSI', ''):.1f}" if data and data['metrics'].get('RSI') else "",
                f"{data['metrics'].get('MACD', ''):+.3f}" if data and data['metrics'].get('MACD') else ""
            ]

            # Create label widgets for each column
            for i, (val, width) in enumerate(zip(values, char_widths)):
                fg = self.theme["tree_fg"]
                
                # Color-code percentage changes: green for positive, red for negative
                if i in [4, 5] and isinstance(val, str) and "%" in val and val != "N/A":
                    try:
                        num = float(val.replace("%", "").replace("+", "").replace(" ", ""))
                        fg = "#66ff99" if num > 0 else "#ff6b6b" if num < 0 else "#cccccc"
                    except:
                        pass
                
                # Truncate text if too long (prevents horizontal overflow)
                display_val = val
                if isinstance(val, str) and len(val) > width:
                    display_val = val[:width-2] + ".."
                
                lbl = tk.Label(
                    frame,
                    text=display_val,
                    bg=bg_color,
                    fg=fg,
                    font=("Consolas", 10),
                    width=width,
                    anchor="center",
                    relief="flat"
                )
                lbl.grid(row=0, column=i, padx=(0, 1), sticky="ew")
                labels.append(lbl)

            # Double-click row to show detailed popup
            click_cmd = partial(self.show_details_popup, ticker)
            frame.bind("<Double-1>", lambda e: click_cmd())
            for lbl in labels:
                lbl.bind("<Double-1>", lambda e: click_cmd())

            # Store row data for sorting and filtering
            self.rows.append({"frame": frame, "labels": labels, "data": data, "ticker": ticker})

        # Update status with fetch count
        fetched = len(self.stock_data)
        total = len(self.current_tickers)
        self.status_lbl.config(text=f"Fetched {fetched}/{total} stock(s).")

    def _normalize_ticker(self, t: str) -> str:
        """Normalize ticker symbol by applying exchange suffix mappings.
        
        Args:
            t: Raw ticker symbol
            
        Returns:
            Normalized ticker symbol with standard suffix format
        """
        t = t.upper().strip()
        for suf, norm in TICKER_SUFFIX_MAP.items():
            if t.endswith(suf):
                return t.replace(suf, norm)
        return t

    def _validate_ticker(self, t: str) -> bool:
        """Check if a ticker symbol has valid format.
        
        Args:
            t: Ticker symbol to validate
            
        Returns:
            True if format is valid (alphanumeric with dots/dashes, max 10 chars)
        """
        t = t.strip()
        if not t or len(t) > 10 or not all(c.isalnum() or c in ".-" for c in t):
            return False
        return True

    def _ticker_exists(self, t: str) -> bool:
        """Check if a ticker exists by querying yfinance.
        
        Args:
            t: Ticker symbol to check
            
        Returns:
            True if ticker has valid data available from yfinance
        """
        try:
            info = yf.Ticker(t).info
            return bool(info.get("regularMarketPrice") or info.get("shortName"))
        except Exception:
            return False

    def open_search_dialog(self) -> None:
        """Open the stock search dialog for adding new tickers.
        
        Uses the external stock_search module to provide search functionality.
        """
        def add_ticker_callback(ticker: str) -> bool:
            """Callback to add ticker from search dialog.
            
            Args:
                ticker: Ticker symbol to add
                
            Returns:
                True if added successfully, False if duplicate
            """
            norm = self._normalize_ticker(ticker)
            if norm in self.current_tickers:
                messagebox.showinfo("Duplicate", f"'{norm}' is already in your list.")
                return False
            
            # Add to top of list (most recent)
            self.current_tickers.insert(0, norm)
            self._update_list_display()
            self.unsaved_changes = True
            self.status_lbl.config(text=f"Added {norm} - Click 'Fetch Data' to analyze")
            return True
        
        open_stock_search(self.root, self.theme, add_ticker_callback)

    def remove_selected(self) -> None:
        """Open dialog to remove one or more tickers from the list.
        
        Displays a listbox with all current tickers where user can select
        multiple items to remove, or remove all at once.
        """
        if not self.rows:
            messagebox.showinfo("Empty List", "No stocks to remove.")
            return
        
        # Create removal dialog window
        remove_dialog = tk.Toplevel(self.root)
        remove_dialog.title("Remove Stocks")
        remove_dialog.geometry("500x450")
        remove_dialog.configure(bg=self.theme["background"])
        remove_dialog.transient(self.root)
        remove_dialog.grab_set()
        
        tk.Label(
            remove_dialog,
            text="Select stocks to remove:",
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 11, "bold")
        ).pack(pady=10)
        
        # Scrollable listbox for ticker selection
        list_frame = tk.Frame(remove_dialog, bg=self.theme["background"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,  # Allow multiple selections
            bg=self.theme["entry"],
            fg=self.theme["text"],
            font=("Consolas", 10),
            yscrollcommand=scrollbar.set
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox with tickers and names
        for ticker in self.current_tickers:
            data = next((d for d in self.stock_data if d["ticker"] == ticker), None)
            display_text = ticker
            if data:
                display_text += f" - {data['name']}"
            listbox.insert(tk.END, display_text)
        
        btn_frame = tk.Frame(remove_dialog, bg=self.theme["background"])
        btn_frame.pack(pady=10)
        
        def do_remove():
            """Remove selected tickers from the list."""
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("No Selection", "Please select at least one stock to remove.")
                return
            
            # Get tickers in reverse order to avoid index shifting issues
            tickers_to_remove = [self.current_tickers[i] for i in sorted(selected_indices, reverse=True)]
            
            # Remove from both lists
            for ticker in tickers_to_remove:
                self.current_tickers.remove(ticker)
                self.stock_data = [d for d in self.stock_data if d["ticker"] != ticker]
            
            self._update_list_display()
            self.unsaved_changes = True
            remove_dialog.destroy()
            messagebox.showinfo("Removed", f"Removed {len(tickers_to_remove)} stock(s).")
        
        def do_remove_all():
            """Remove all tickers from the list after confirmation."""
            if messagebox.askyesno("Remove All", "Are you sure you want to remove ALL stocks from the list?"):
                self.current_tickers.clear()
                self.stock_data.clear()
                self._update_list_display()
                self.unsaved_changes = True
                remove_dialog.destroy()
                messagebox.showinfo("Removed", "All stocks removed.")
        
        # Action buttons
        tk.Button(
            btn_frame,
            text="Remove Selected",
            command=do_remove,
            bg=self.theme["button"],
            fg=self.theme["text"],
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="Remove All",
            command=do_remove_all,
            bg=self.theme["button"],
            fg=self.theme["text"],
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="Cancel",
            command=remove_dialog.destroy,
            bg=self.theme["button"],
            fg=self.theme["text"],
            width=15
        ).pack(side=tk.LEFT, padx=5)

    def clear_all(self) -> None:
        """Deprecated - functionality moved to remove_selected dialog."""
        self.remove_selected()

    def set_custom_period(self) -> None:
        """Open dialog to set custom lookback period for historical analysis.
        
        The custom period affects price swing calculations and historical metrics.
        """
        pop = tk.Toplevel(self.root)
        pop.title("Custom Period")
        pop.geometry("300x120")
        pop.configure(bg=self.theme["background"])
        pop.transient(self.root)
        pop.grab_set()

        tk.Label(pop, text="Enter custom period (days):", bg=self.theme["background"], fg=self.theme["text"]).pack(pady=10)
        entry = tk.Entry(pop, width=10, justify="center")
        entry.insert(0, str(CONFIG.custom_period_days))
        entry.pack(pady=5)
        entry.select_range(0, tk.END)

        def save() -> None:
            """Save the new custom period and refresh data if available."""
            try:
                days = int(entry.get())
                if days <= 0:
                    raise ValueError
                CONFIG.custom_period_days = days
                
                # Update header label with new period
                self.header_labels[5].config(text=f"{days} Day %")
                
                # Mark as unsaved change
                self.unsaved_changes = True
                
                pop.destroy()
                
                # Refresh data if we have stocks loaded
                if self.stock_data:
                    self.fetch_and_display()
                else:
                    self._update_list_display()
            except ValueError:
                messagebox.showerror("Invalid", "Please enter a positive integer.")

        tk.Button(pop, text="OK", command=save, bg=self.theme["button"], fg=self.theme["text"]).pack(pady=5)

    def new_list(self) -> None:
        """Create a new empty ticker list, prompting to save current if unsaved."""
        if self.unsaved_changes and messagebox.askyesnocancel("Unsaved Changes", "Save current list before creating new?"):
            self.save_current_list()
        self.current_tickers.clear()
        self.stock_data.clear()
        self.current_list_name = ""
        self.trading_aggression = 0.5
        self.trading_style_var.set("Moderate")
        self._update_list_display()
        self.list_name_lbl.config(text="(none)")
        self.unsaved_changes = False

    def save_current_list(self) -> None:
        """Save the current ticker list to file.
        
        If no filename is set, prompts for "Save As" instead.
        """
        if not self.current_list_name:
            self.save_list_as()
            return
        save_ticker_list(self.current_list_name, self.current_tickers, self.trading_aggression, 
                        CONFIG.custom_period_days, self.recommendation_mode)
        self.unsaved_changes = False
        messagebox.showinfo("Saved", f"List saved as '{os.path.basename(self.current_list_name)}'")

    def save_list_as(self) -> None:
        """Save the current ticker list with a new filename."""
        fn = filedialog.asksaveasfilename(
            initialdir=CONFIG.lists_dir, title="Save Ticker List",
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if fn:
            save_ticker_list(fn, self.current_tickers, self.trading_aggression, 
                           CONFIG.custom_period_days, self.recommendation_mode)
            self.current_list_name = fn
            self.list_name_lbl.config(text=os.path.basename(fn))
            self.unsaved_changes = False

    def open_dialog(self) -> None:
        """Open an existing ticker list from file, prompting to save current if unsaved."""
        if self.unsaved_changes and messagebox.askyesnocancel("Unsaved Changes", "Save current list before loading?"):
            self.save_current_list()
        fn = filedialog.askopenfilename(
            initialdir=CONFIG.lists_dir, title="Load Ticker List",
            filetypes=[("JSON files", "*.json")]
        )
        if fn:
            self.current_tickers, self.trading_aggression, custom_period, rec_mode = load_ticker_list(fn)
            CONFIG.custom_period_days = custom_period
            self.recommendation_mode = rec_mode
            
            # Update header label with loaded period
            self.header_labels[5].config(text=f"{custom_period} Day %")
            
            style_name = self._get_aggression_label(self.trading_aggression)
            self.trading_style_var.set(style_name)
            self.rec_mode_var.set(rec_mode)
            self.current_list_name = fn
            self.stock_data.clear()
            self._update_list_display()
            self.list_name_lbl.config(text=os.path.basename(fn))
            self.unsaved_changes = False

    def _load_default_list(self, silent: bool = False) -> None:
        """Load the default ticker list if one is configured.
        
        Args:
            silent: If True, don't show info messagebox on successful load
        """
        path = os.path.join(CONFIG.lists_dir, CONFIG.default_list_file)
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            name = f.read().strip()
        full = os.path.join(CONFIG.lists_dir, name)
        if os.path.exists(full):
            self.current_tickers, self.trading_aggression, custom_period, rec_mode = load_ticker_list(full)
            CONFIG.custom_period_days = custom_period
            self.recommendation_mode = rec_mode
            
            # Update header label with loaded period
            self.header_labels[5].config(text=f"{custom_period} Day %")
            
            style_name = self._get_aggression_label(self.trading_aggression)
            self.trading_style_var.set(style_name)
            self.rec_mode_var.set(rec_mode)
            self.current_list_name = full
            self.stock_data.clear()
            self._update_list_display()
            self.list_name_lbl.config(text=name)
            self.unsaved_changes = False
            if not silent:
                messagebox.showinfo("Default Loaded", f"Loaded default list: {name}")

    def set_as_default(self) -> None:
        """Set the current list as the default to load on startup."""
        if not self.current_list_name:
            messagebox.showwarning("No List", "Save the current list first.")
            return
        path = os.path.join(CONFIG.lists_dir, CONFIG.default_list_file)
        with open(path, "w", encoding="utf-8") as f:
            f.write(os.path.basename(self.current_list_name))
        messagebox.showinfo("Default Set", f"'{os.path.basename(self.current_list_name)}' is now default.")

    def remove_default(self) -> None:
        """Remove the default list setting."""
        path = os.path.join(CONFIG.lists_dir, CONFIG.default_list_file)
        if os.path.exists(path):
            os.remove(path)
            messagebox.showinfo("Default Removed", "Default list removed.")
        else:
            messagebox.showinfo("No Default", "No default list is set.")
            
    def show_info_popup(self, event=None) -> None:
        """Display a popup explaining all metrics and recommendations."""
        info_text = (
            "Stock Tracker Metrics Explained:\n\n"
            "=== RECOMMENDATION SYSTEMS ===\n\n"
            "Simple (Legacy) Mode:\n"
            "  • Uses traditional threshold-based logic\n"
            "  • Compares price to 50-day MA and 52-week high\n"
            "  • Binary decision: Buy, Consider Buying, Hold, Consider Selling, Sell\n"
            "  • Good for straightforward, rule-based trading\n\n"
            "Complex (Multi-Factor Scoring) Mode:\n"
            "  • Advanced system that evaluates 6 weighted factors:\n"
            "    1. MA Position (25%): Distance from 50-day moving average\n"
            "    2. 52W Range (20%): Position in 52-week high/low range\n"
            "    3. RSI (20%): Momentum indicator for overbought/oversold\n"
            "    4. MACD (15%): Trend momentum strength\n"
            "    5. P/E Ratio (10%): Valuation metric\n"
            "    6. Analyst Target (10%): Professional price targets\n"
            "  • Generates score from -100 (strong sell) to +100 (strong buy)\n"
            "  • Considers multiple factors simultaneously for holistic view\n"
            "  • Score thresholds adjust based on trading aggression setting\n"
            "  • More nuanced than simple mode, considers full picture\n\n"
            "=== TRADING STYLE ===\n\n"
            "Trading Style: Adjusts recommendation sensitivity.\n"
            "  • Conservative: Requires larger price movements, fewer trades\n"
            "    Wider margins before recommending buy/sell\n"
            "  • Moderate: Balanced approach with standard thresholds\n"
            "  • Aggressive: Tighter thresholds, more frequent trading signals\n"
            "    Acts more like day trading with smaller movements\n\n"
            "=== METRICS ===\n\n"
            "Price: Current trading price of the stock.\n\n"
            "Volume: Number of shares traded. Important because:\n"
            "  • High volume = Strong interest and liquidity (easier to buy/sell)\n"
            "  • Low volume = Weak interest, harder to execute large trades\n"
            "  • Unusual volume spikes may signal news or major price movements\n"
            "  • Volume confirms trends: price moves with high volume are more reliable\n\n"
            "P/E Ratio: Trailing Price-to-Earnings Ratio.\n"
            "  • Low P/E (< 15): Potentially undervalued, good value\n"
            "  • Moderate P/E (15-30): Fair valuation\n"
            "  • High P/E (> 30): Potentially overvalued, expensive relative to earnings\n"
            "  Note: Growth stocks often have higher P/E ratios\n\n"
            "Target %: Percentage difference between the Analyst Mean Target Price and current price.\n\n"
            "RSI: Relative Strength Index (14-day). Momentum oscillator:\n"
            "  Thresholds adjust with trading style:\n"
            "  Conservative: >75 Overbought, <25 Oversold\n"
            "  Aggressive: >65 Overbought, <35 Oversold\n\n"
            "MACD: Moving Average Convergence Divergence (12/26/9 periods).\n"
            "  Helps identify momentum and trend direction.\n"
            "  Positive values suggest upward momentum, negative suggest downward.\n\n"
            "Day %: Price swing percentage for 1 day and the custom period (currently set to "
            f"{CONFIG.custom_period_days} days).\n\n"
            "Score (Complex Mode Only): Overall score from -100 to +100\n"
            "  • +100 to +30: Strong to moderate buy signals\n"
            "  • +30 to +15: Consider buying zone\n"
            "  • +15 to -15: Hold zone (neutral)\n"
            "  • -15 to -30: Consider selling zone\n"
            "  • -30 to -100: Moderate to strong sell signals\n"
            "  Thresholds adjust based on your trading aggression setting."
        )

        pop = tk.Toplevel(self.root)
        pop.title("Stock Tracker Metric Info")
        pop.geometry("900x700")
        pop.configure(bg=self.theme["background"])
        pop.transient(self.root)
        pop.grab_set()

        # Create scrollable text widget for long help text
        frame = tk.Frame(pop, bg=self.theme["background"])
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(
            frame,
            wrap=tk.WORD,
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 10),
            yscrollcommand=scrollbar.set,
            relief=tk.FLAT
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_widget.insert("1.0", info_text)
        text_widget.config(state=tk.DISABLED)
        
        scrollbar.config(command=text_widget.yview)
        
        tk.Button(pop, text="Close", command=pop.destroy, bg=self.theme["button"], fg=self.theme["text"]).pack(pady=5)
        
    def show_details_popup(self, ticker: str) -> None:
        """Display detailed information popup for a specific stock.
        
        Shows all available metrics, price data, and recommendation details
        in a formatted text display.
        
        Args:
            ticker: Stock ticker symbol to display details for
        """
        data = next((d for d in self.stock_data if d["ticker"] == ticker), None)
        if not data:
            messagebox.showerror("Error", f"No data found for {ticker}.")
            return

        pop = tk.Toplevel(self.root)
        pop.title(f"Details: {ticker} ({data['name']})")
        pop.configure(bg=self.theme["background"])
        pop.transient(self.root)
        pop.grab_set()

        info = data['info']
        metrics = data['metrics']
        
        # Build formatted detail text
        detail_text = f"Ticker: {ticker}\n"
        detail_text += f"Name: {data['name']}\n"
        detail_text += f"Recommendation: {data['recommendation']}"
        
        # Add score if in complex mode
        if self.recommendation_mode == "complex" and metrics.get("Score") is not None:
            detail_text += f" (Score: {metrics['Score']:+.1f})"
        
        detail_text += f"\nReasons: {', '.join(data['reasons']) if data['reasons'] else 'None'}\n"
        detail_text += f"Sector/Industry: {data['sector']} / {data['industry']}\n"
        detail_text += "--- Price & Performance ---\n"
        detail_text += f"Current Price: {info.get('regularMarketPrice', 'N/A'):.2f}\n"
        detail_text += f"Previous Close: {info.get('previousClose', 'N/A'):.2f}\n"
        
        # Format volume for readability
        volume = info.get('volume')
        if volume:
            if volume >= 1_000_000_000:
                vol_str = f"{volume/1_000_000_000:.2f}B"
            elif volume >= 1_000_000:
                vol_str = f"{volume/1_000_000:.2f}M"
            elif volume >= 1_000:
                vol_str = f"{volume/1_000:.2f}K"
            else:
                vol_str = f"{volume:,}"
            detail_text += f"Volume: {vol_str}\n"
        
        detail_text += f"50-Day Avg: {info.get('fiftyDayAverage', 'N/A'):.2f}\n"
        detail_text += f"52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A'):.2f}\n"
        detail_text += f"52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A'):.2f}\n"
        detail_text += f"1 Day Swing: {data.get('price_swing_1d', 'N/A')}\n"
        detail_text += f"{CONFIG.custom_period_days} Day Swing: {data.get('price_swing_1m', 'N/A')}\n"
        detail_text += "--- Key Metrics ---\n"
        
        pe = metrics.get('P/E')
        if pe:
            detail_text += f"Trailing P/E: {pe:.1f}\n"
        else:
            detail_text += f"Trailing P/E: N/A\n"
        
        if metrics.get('Target %') is not None:
             detail_text += f"Analyst Target Price: {metrics.get('Target', 'N/A'):.2f} ({metrics.get('Target %'):+.1f}%)\n"
        else:
             detail_text += f"Analyst Target Price: {metrics.get('Target', 'N/A'):.2f} (N/A)\n"
        detail_text += f"RSI: {metrics.get('RSI', 'N/A'):.1f}\n"
        detail_text += f"MACD: {metrics.get('MACD', 'N/A'):+.3f}\n"
        
        tk.Label(pop, text=detail_text, justify=tk.LEFT, padx=10, pady=10,
                 bg=self.theme["background"], fg=self.theme["text"], font=("Consolas", 10)).pack()
        tk.Button(pop, text="Close", command=pop.destroy, bg=self.theme["button"], fg=self.theme["text"]).pack(pady=5)

    def fetch_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch and analyze data for a single stock ticker.
        
        Downloads historical data, calculates metrics, and generates recommendations
        based on configured thresholds and selected recommendation mode.
        
        Args:
            ticker: Stock ticker symbol to fetch
            
        Returns:
            Dictionary containing all stock data, metrics, and recommendations
        """
        try:
            ticker = self._normalize_ticker(ticker)
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Fetch historical data for different time periods
            hist_1d = stock.history(period="2d", interval="1h")  # For 1-day swing
            period_str = f"{CONFIG.custom_period_days}d"
            hist_long = stock.history(period=period_str)  # For custom period and indicators

            # Extract basic info
            name = (info.get("shortName") or "N/A")[:30]
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            price = info.get("regularMarketPrice")

            # Calculate 1-day price swing
            swing_1d: Optional[float] = None
            if not hist_1d.empty and len(hist_1d) >= 2:
                close = hist_1d["Close"]
                swing_1d = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

            # Calculate custom period price swing
            swing_1m: Optional[float] = None
            if not hist_long.empty and len(hist_long) >= 2:
                close = hist_long["Close"]
                swing_1m = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

            # Calculate metrics
            metrics: Dict[str, Any] = {}
            if CONFIG.enable_metrics["pe_ratio"]:
                metrics["P/E"] = registry.compute("pe_ratio", info)[0]
            if CONFIG.enable_metrics["analyst_target"]:
                tgt, diff, _ = registry.compute("analyst_target", info, price)
                metrics["Target"] = tgt
                metrics["Target %"] = diff
            if CONFIG.enable_metrics["rsi"]:
                rsi, _ = registry.compute("rsi", hist_long)
                metrics["RSI"] = rsi
            if CONFIG.enable_metrics["macd"]:
                macd, _ = registry.compute("macd", hist_long)
                metrics["MACD"] = macd

            # Generate recommendation based on selected mode
            if self.recommendation_mode == "complex":
                rec, reasons, score = self._calculate_complex_recommendation(
                    info, price, hist_long, metrics, swing_1d
                )
                metrics["Score"] = score
            else:
                rec, reasons = self._calculate_simple_recommendation(
                    info, price, hist_long, metrics, swing_1d
                )

            return {
                "ticker": ticker, "name": name, "sector": sector, "industry": industry,
                "info": info, "recommendation": rec, "reasons": reasons,
                "metrics": metrics,
                "price_swing_1d": f"{swing_1d:+.2f}%" if swing_1d is not None else "N/A",
                "price_swing_1m": f"{swing_1m:+.2f}%" if swing_1m is not None else "N/A"
            }
        except Exception as e:
            # Return error placeholder if fetching fails
            log_and_show("Fetch Error", f"{ticker}: {e}", "fetch_stock_data", ticker)
            return {
                "ticker": ticker, "name": "Error", "sector": "N/A", "industry": "N/A",
                "info": {}, "recommendation": "Hold", "reasons": [], "metrics": {},
                "price_swing_1d": "N/A", "price_swing_1m": "N/A"
            }

    def _calculate_simple_recommendation(self, info: dict, price: float, hist_long: pd.DataFrame,
                                        metrics: Dict[str, Any], swing_1d: Optional[float]) -> Tuple[str, List[str]]:
        """Calculate recommendation using simple legacy logic.
        
        Args:
            info: Stock info dictionary
            price: Current stock price
            hist_long: Historical price data
            metrics: Calculated metrics dictionary
            swing_1d: 1-day price swing percentage
            
        Returns:
            Tuple of (recommendation, list of reasons)
        """
        rec = "Hold"
        reasons: List[str] = []
        
        # Get adjusted thresholds based on trading aggression
        thresholds = self._get_adjusted_thresholds()

        # Flag significant swings in reasons
        if swing_1d is not None and abs(swing_1d) >= CONFIG.price_swing_threshold:
            reasons.append(f"1-day {swing_1d:+.2f}%")

        # Recommendation based on 50-day moving average
        if price and "fiftyDayAverage" in info:
            ma = info["fiftyDayAverage"]
            if price < ma * thresholds["buy_ma_ratio"]:
                rec = "Buy"
                reasons.append("Below 50-day MA")
            elif price < ma * thresholds["consider_buy_ma_ratio"]:
                rec = "Consider Buying"
                reasons.append("Near 50-day MA")

        # Recommendation based on 52-week high
        if price and "fiftyTwoWeekHigh" in info:
            high = info["fiftyTwoWeekHigh"]
            if price > high * thresholds["sell_high_ratio"]:
                rec = "Sell"
                reasons.append("Near 52-week high")
            elif price > high * thresholds["consider_sell_high_ratio"] and rec == "Hold":
                rec = "Consider Selling"
                reasons.append("Approaching high")

        # Apply adjusted RSI thresholds
        rsi = metrics.get("RSI")
        if rsi is not None:
            if rsi > thresholds["rsi_overbought"]:
                reasons.append("Overbought")
            elif rsi < thresholds["rsi_oversold"]:
                reasons.append("Oversold")
                
        return rec, reasons

    def _calculate_complex_recommendation(self, info: dict, price: float, hist_long: pd.DataFrame,
                                         metrics: Dict[str, Any], swing_1d: Optional[float]) -> Tuple[str, List[str], float]:
        """Calculate recommendation using complex multi-factor scoring system.
        
        Scores range from -100 (strong sell) to +100 (strong buy).
        Each factor is weighted and normalized to contribute to the final score.
        
        Args:
            info: Stock info dictionary
            price: Current stock price
            hist_long: Historical price data
            metrics: Calculated metrics dictionary
            swing_1d: 1-day price swing percentage
            
        Returns:
            Tuple of (recommendation, list of reasons, overall score)
        """
        reasons: List[str] = []
        total_score = 0.0
        weights = CONFIG.score_weights
        thresholds = self._get_adjusted_thresholds()
        
        # Factor 1: 50-Day Moving Average Position (-100 to +100)
        if price and "fiftyDayAverage" in info:
            ma = info["fiftyDayAverage"]
            ma_ratio = (price / ma - 1) * 100  # Percentage from MA
            # Score: positive when below MA (buy signal), negative when above (sell signal)
            ma_score = max(-100, min(100, -ma_ratio * 5))  # Scale to -100/+100
            total_score += ma_score * weights["ma_position"]
            
            if ma_ratio < -3:
                reasons.append(f"Below MA ({ma_ratio:.1f}%)")
            elif ma_ratio > 2:
                reasons.append(f"Above MA (+{ma_ratio:.1f}%)")
        
        # Factor 2: 52-Week High Position (-100 to +100)
        if price and "fiftyTwoWeekHigh" in info and "fiftyTwoWeekLow" in info:
            high = info["fiftyTwoWeekHigh"]
            low = info["fiftyTwoWeekLow"]
            if high > low:
                # Position in 52-week range (0 = at low, 100 = at high)
                range_position = ((price - low) / (high - low)) * 100
                # Score: positive when low (undervalued), negative when high (overvalued)
                high_score = 100 - (range_position * 2)  # 0% = +100, 100% = -100
                total_score += high_score * weights["high_position"]
                
                if range_position > 90:
                    reasons.append(f"Near 52W high ({range_position:.0f}%)")
                elif range_position < 20:
                    reasons.append(f"Near 52W low ({range_position:.0f}%)")
        
        # Factor 3: RSI Indicator (-100 to +100)
        rsi = metrics.get("RSI")
        if rsi is not None:
            # Score: positive when oversold, negative when overbought
            if rsi > 50:
                rsi_score = -(rsi - 50) * 2  # 50-100 maps to 0 to -100
            else:
                rsi_score = (50 - rsi) * 2   # 0-50 maps to +100 to 0
            total_score += rsi_score * weights["rsi"]
            
            if rsi > thresholds["rsi_overbought"]:
                reasons.append(f"Overbought (RSI {rsi:.0f})")
            elif rsi < thresholds["rsi_oversold"]:
                reasons.append(f"Oversold (RSI {rsi:.0f})")
        
        # Factor 4: MACD Momentum (-100 to +100)
        macd = metrics.get("MACD")
        if macd is not None:
            # Normalize MACD to -100/+100 range (assuming typical MACD values -5 to +5)
            macd_score = max(-100, min(100, macd * 20))
            total_score += macd_score * weights["macd"]
            
            if macd > 0.5:
                reasons.append(f"Positive momentum (MACD {macd:+.2f})")
            elif macd < -0.5:
                reasons.append(f"Negative momentum (MACD {macd:+.2f})")
        
        # Factor 5: P/E Ratio Valuation (-100 to +100)
        pe = metrics.get("P/E")
        if pe and pe > 0:
            # Score: positive when low P/E (undervalued), negative when high (overvalued)
            # Typical P/E: 0-50, optimal around 15
            if pe < 15:
                pe_score = 50  # Undervalued
            elif pe > 30:
                pe_score = -50  # Overvalued
            else:
                pe_score = 50 - ((pe - 15) / 15 * 100)  # Linear between 15-30
            total_score += pe_score * weights["pe_ratio"]
            
            if pe > 30:
                reasons.append(f"High P/E ({pe:.1f})")
            elif pe < 10:
                reasons.append(f"Low P/E ({pe:.1f})")
        
        # Factor 6: Analyst Target Price (-100 to +100)
        target_pct = metrics.get("Target %")
        if target_pct is not None:
            # Score: positive when price below target (upside), negative when above (downside)
            target_score = max(-100, min(100, target_pct * 2))  # Scale to -100/+100
            total_score += target_score * weights["analyst_target"]
            
            if target_pct > 10:
                reasons.append(f"Upside potential ({target_pct:+.0f}%)")
            elif target_pct < -10:
                reasons.append(f"Downside risk ({target_pct:+.0f}%)")
        
        # Add price swing to reasons if significant
        if swing_1d is not None and abs(swing_1d) >= CONFIG.price_swing_threshold:
            reasons.append(f"1-day {swing_1d:+.2f}%")
        
        # Adjust score based on trading aggression
        aggression_multiplier = 0.5 + (self.trading_aggression * 0.5)  # 0.5 to 1.0
        adjusted_score = total_score * aggression_multiplier
        
        # Convert score to recommendation
        # Aggressive: tighter thresholds, Conservative: wider thresholds
        buy_threshold = 30 - (self.trading_aggression * 20)  # 30 to 10
        consider_buy_threshold = 15 - (self.trading_aggression * 10)  # 15 to 5
        sell_threshold = -30 + (self.trading_aggression * 20)  # -30 to -10
        consider_sell_threshold = -15 + (self.trading_aggression * 10)  # -15 to -5
        
        if adjusted_score >= buy_threshold:
            rec = "Buy"
        elif adjusted_score >= consider_buy_threshold:
            rec = "Consider Buying"
        elif adjusted_score <= sell_threshold:
            rec = "Sell"
        elif adjusted_score <= consider_sell_threshold:
            rec = "Consider Selling"
        else:
            rec = "Hold"
        
        return rec, reasons, round(adjusted_score, 1)

    def fetch_and_display(self) -> None:
        """Fetch data for all tickers and update the display.
        
        Uses ThreadPoolExecutor to fetch multiple stocks concurrently for better
        performance. Updates the UI progressively as each stock completes.
        """
        if not self.current_tickers:
            messagebox.showinfo("No Tickers", "Add tickers to track using the Search & Add button.")
            return

        # Update UI state for fetching
        self.status_lbl.config(text="Fetching data...")
        self.root.update_idletasks()
        for btn in self.button_refs.values():
            btn.config(state=tk.DISABLED)

        # Fetch stocks concurrently using thread pool
        new_data: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=CONFIG.max_threads) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_stock_data, t): t
                for t in self.current_tickers
            }
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_ticker)):
                try:
                    data = future.result()
                    new_data.append(data)
                    self.status_lbl.config(text=f"Fetched data for {data['ticker']} ({i+1}/{len(self.current_tickers)})...")
                    self.root.update_idletasks()
                except Exception as e:
                    log_and_show("Thread Error", f"Error fetching data for a ticker: {e}", "fetch_and_display")

        # Update display with new data
        self.stock_data = new_data
        self._update_list_display()

        # Update last updated timestamp
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated_lbl.config(text=f"Last updated: {self.last_updated}")

        # Re-enable UI controls
        for btn in self.button_refs.values():
            btn.config(state=tk.NORMAL)

    def _on_closing(self) -> None:
        """Handle window close event, prompting to save unsaved changes."""
        if self.unsaved_changes:
            if messagebox.askyesno("Exit", "Unsaved changes! Save before exiting?"):
                self.save_current_list()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StockTrackerApp(root)
    root.mainloop()
