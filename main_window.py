"""Main application window with all functionality.

This module contains the complete Stock Tracker application logic including:
- Data fetching and analysis
- UI management
- Recommendations (simple and complex modes)
- All helper functions
"""

import json
import logging
import os
import tkinter as tk
from tkinter import messagebox, filedialog
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from typing import Callable, List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import CONFIG, THEMES, TICKER_SUFFIX_MAP
from stock_search import open_stock_search
from stock_charts import open_chart_window


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="stock_tracker.log",
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def log_and_show(title: str, message: str, func_name: str, ticker: Optional[str] = None, msg_type: str = "error") -> None:
    """Log an error/warning and display a messagebox to the user."""
    log_msg = f"{title}: {message}"
    if ticker:
        log_msg += f" (Ticker: {ticker})"
    log_msg += f" in {func_name}"
    logging.error(log_msg)
    getattr(messagebox, f"show{msg_type}")(title, message)


def load_ticker_list(filename: str) -> Tuple[List[str], float, int, str]:
    """Load a list of ticker symbols from a JSON file."""
    if not os.path.exists(filename):
        return [], 0.5, 30, "simple"
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
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
    """Save a list of ticker symbols to a JSON file."""
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
    """Export stock data to CSV with Save As dialog."""
    if not stock_data:
        messagebox.showwarning("Export Error", "No stock data to export.")
        return

    default_name = f"stock_data_{datetime.now():%Y%m%d_%H%M%S}.csv"
    file_path = filedialog.asksaveasfilename(
        title="Export Data as CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=default_name
    )
    if not file_path:
        return

    try:
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


def format_volume(volume: Optional[int]) -> str:
    """Format volume number for display."""
    if not volume:
        return ""
    
    if volume >= 1_000_000_000:
        return f"{volume/1_000_000_000:.2f}B"
    elif volume >= 1_000_000:
        return f"{volume/1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.2f}K"
    else:
        return str(volume)


def get_aggression_label(aggression: float) -> str:
    """Get descriptive label for aggression level."""
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


# --------------------------------------------------------------------------- #
# Metric Functions
# --------------------------------------------------------------------------- #
def get_pe_ratio(info: dict) -> Tuple[Optional[float], None]:
    """Extract trailing P/E ratio from stock info."""
    return info.get("trailingPE"), None


def get_analyst_target(info: dict, price: float) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Calculate analyst target price and percentage difference."""
    target = info.get("targetMeanPrice")
    if target and price:
        diff = (target - price) / price * 100
        return target, diff, "Potential Upside" if price < target else None
    return None, None, None


def get_rsi(hist: pd.DataFrame, period: int = 14) -> Tuple[Optional[float], Optional[str]]:
    """Calculate Relative Strength Index (RSI)."""
    if hist.empty or "Close" not in hist:
        return None, None
    
    delta = hist["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    
    if avg_loss == 0:
        return 100.0, "Overbought"
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    flag = (
        "Overbought" if rsi > CONFIG.recommendation_thresholds["rsi_overbought"]
        else "Oversold" if rsi < CONFIG.recommendation_thresholds["rsi_oversold"] else None
    )
    return rsi, flag


def get_macd(hist: pd.DataFrame, short: int = 12, long: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float]]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    if hist.empty or "Close" not in hist:
        return None, None
    
    close = hist["Close"]
    ema_s = close.ewm(span=short, adjust=False).mean()
    ema_l = close.ewm(span=long, adjust=False).mean()
    
    macd_line = ema_s - ema_l
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line.iloc[-1], sig_line.iloc[-1]


class MetricRegistry:
    """Registry pattern for metric calculation functions."""
    
    def __init__(self) -> None:
        self._metrics: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        """Register a metric calculation function."""
        self._metrics[name] = func

    def compute(self, name: str, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
        """Compute a registered metric by name."""
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
    """Main application class for the Stock Tracker GUI."""
    
    def __init__(self, root: tk.Tk, theme: str = "dark") -> None:
        """Initialize the Stock Tracker application."""
        self.root = root
        self.root.title("Stock Tracker by Mattias")
        self.root.minsize(CONFIG.min_window_width, CONFIG.min_window_height)
        self.theme = THEMES[theme]
        self.root.configure(bg=self.theme["background"])

        # Application state
        self.button_refs: Dict[str, tk.Button] = {}
        self.stock_data: List[Dict[str, Any]] = []
        self.current_tickers: List[str] = []
        self.current_list_name: str = ""
        self.unsaved_changes: bool = False
        self.filter_query: str = ""
        self.trading_aggression: float = 0.5
        self.last_updated: Optional[str] = None
        self.recommendation_mode: str = "simple"

        # Sorting state
        self._sort_column: str = "Recommendation"
        self._sort_reverse: bool = False

        # UI element references
        self.rows = []
        self.header_labels = []
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

        # File menu
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

        # Period menu
        period_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Period", menu=period_menu)
        period_menu.add_command(label="Set Custom Period (Days)...", command=self.set_custom_period)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Info", command=self.show_info_popup)

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

        # Fetch Data button
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
            "Chart", "Ticker", "Name", "Recommendation", "Price",
            "1 Day %", f"{CONFIG.custom_period_days} Day %",
            "Sector", "Industry", "Volume", "P/E", "Target %", "RSI", "MACD"
        ]
        char_widths = [6, 10, 30, 18, 10, 10, 12, 22, 25, 12, 8, 10, 8, 10]

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
            if text != "Chart":
                lbl.bind("<Button-1>", lambda e, col=text: self._sort_by_column(col))
            self.header_labels.append(lbl)
        
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
        """Set recommendation system mode (simple or complex)."""
        self.recommendation_mode = mode
        self.rec_mode_var.set(mode)
        self.unsaved_changes = True
        
        mode_display = "Simple (Legacy)" if mode == "simple" else "Complex (Multi-Factor Scoring)"
        
        if self.stock_data and self.current_tickers:
            self.status_lbl.config(text=f"Recommendation system changed to {mode_display} - Updating...")
            self.root.update_idletasks()
            self.fetch_and_display()
        else:
            self.status_lbl.config(text=f"Recommendation system set to {mode_display}")

    def set_trading_style(self, aggression: float, style_name: str) -> None:
        """Set trading style from menu and update recommendations."""
        self.trading_aggression = aggression
        self.trading_style_var.set(style_name)
        self.unsaved_changes = True
        
        if self.stock_data and self.current_tickers:
            self.status_lbl.config(text=f"Trading style changed to {style_name} - Updating recommendations...")
            self.root.update_idletasks()
            self.fetch_and_display()
        else:
            self.status_lbl.config(text=f"Trading style set to {style_name}")

    def _get_adjusted_thresholds(self) -> Dict[str, float]:
        """Calculate adjusted recommendation thresholds based on trading aggression."""
        aggr = self.trading_aggression
        base = CONFIG.recommendation_thresholds.copy()
        
        base["buy_ma_ratio"] = 0.95 + (aggr * 0.04)
        base["consider_buy_ma_ratio"] = 0.97 + (aggr * 0.04)
        base["sell_high_ratio"] = 0.99 - (aggr * 0.03)
        base["consider_sell_high_ratio"] = 0.97 - (aggr * 0.04)
        base["rsi_overbought"] = 75 - (aggr * 10)
        base["rsi_oversold"] = 25 + (aggr * 10)
        
        return base
        
    def filter_list(self, query: str) -> None:
        """Filter displayed stocks by ticker symbol or company name."""
        self.filter_query = query.strip().upper()
        if self.search_entry:
            if not self.filter_query:
                self.search_entry.delete(0, tk.END)
                self.status_lbl.config(text="Filter cleared. Showing all stocks.")
            else:
                self.status_lbl.config(text=f"Filtered by: '{query}'")

        self._sort_by_column(self._sort_column) 

        filtered_count = 0
        for row in self.rows:
            ticker = row['ticker']
            name = row['data'].get('name', 'N/A') if row['data'] else 'N/A'
            
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
            base_text = lbl.cget("text").replace(" ↑", "").replace(" ↓", "")
            
            is_sorted = False
            if base_text == self._sort_column:
                is_sorted = True
            elif base_text.endswith("Day %") and self._sort_column in ["1 Day %", f"{CONFIG.custom_period_days} Day %"]:
                is_sorted = True
            
            if is_sorted:
                arrow = " ↓" if self._sort_reverse else " ↑"
                lbl.config(text=base_text + arrow)
            else:
                lbl.config(text=base_text)

    def _sort_by_column(self, col_text: str):
        """Sort the displayed rows by a specific column."""
        col_text = col_text.replace(" ↑", "").replace(" ↓", "")

        if self._sort_column == col_text:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = col_text
            self._sort_reverse = False

        self._update_sort_indicator()

        col_idx = next((i for i, h in enumerate(self.header_labels) 
                       if h.cget("text").replace(" ↑", "").replace(" ↓", "") == col_text), None)
        if col_idx is None:
            return
            
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
            def key_func(row):
                val = row["labels"][col_idx].cget("text")
                if val in ("N/A", "", "Error", "📊"):
                    return (1, 0)
                if "%" in val:
                    try:
                        return (0, float(val.replace("%", "").replace("+", "").replace(" ", "")))
                    except:
                        return (1, 0)
                try:
                    return (0, float(val))
                except:
                    return (1, val.lower())

            self.rows.sort(key=key_func, reverse=self._sort_reverse)
        
        for row in self.rows:
            row["frame"].pack_forget()

        for row in self.rows:
            ticker = row['ticker']
            name = row['data'].get('name', 'N/A') if row['data'] else 'N/A'
            if not self.filter_query or self.filter_query in ticker or self.filter_query in name.upper():
                row["frame"].pack(fill=tk.X, pady=1)

    def _update_list_display(self) -> None:
        """Rebuild the entire table display with current stock data."""
        for row in self.rows:
            row["frame"].destroy()
        self.rows.clear()
        self.filter_query = ""
        if self.search_entry:
            self.search_entry.delete(0, tk.END)

        if not self.current_tickers:
            self.status_lbl.config(text="Ready")
            return

        display_data = []
        for ticker in self.current_tickers:
            data = next((d for d in self.stock_data if d["ticker"] == ticker), None)
            priority = 6
            if data:
                priority = {
                    "Sell": 0,
                    "Consider Selling": 1,
                    "Buy": 2,            
                    "Consider Buying": 3,
                    "Hold": 4,
                }.get(data["recommendation"], 5)
            display_data.append((ticker, data, priority))

        display_data.sort(key=lambda x: x[2]) 

        char_widths = [6, 10, 30, 18, 10, 10, 12, 22, 25, 12, 8, 10, 8, 10]

        for ticker, data, _ in display_data:
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

            labels = []
            
            volume_str = ""
            if data and data['info'].get('volume'):
                volume_str = format_volume(data['info'].get('volume'))
            
            values = [
                "📊",
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
                f"{data['metrics'].get('MACD', ''):+.3f}" if data and data['metrics'].get('MACD') else "",
            ]

            for i, (val, width) in enumerate(zip(values, char_widths)):
                fg = self.theme["tree_fg"]
                
                if i in [5, 6] and isinstance(val, str) and "%" in val and val != "N/A":
                    try:
                        num = float(val.replace("%", "").replace("+", "").replace(" ", ""))
                        fg = "#66ff99" if num > 0 else "#ff6b6b" if num < 0 else "#cccccc"
                    except:
                        pass
                
                display_val = val
                if isinstance(val, str) and len(val) > width and i != 0:
                    display_val = val[:width-2] + ".."
                
                lbl = tk.Label(
                    frame,
                    text=display_val,
                    bg=bg_color,
                    fg=fg,
                    font=("Consolas", 12 if i == 0 else 10),
                    width=width,
                    anchor="center",
                    relief="flat"
                )
                lbl.grid(row=0, column=i, padx=(0, 1), sticky="ew")
                
                if i == 0 and data:
                    lbl.config(cursor="hand2")
                    chart_cmd = partial(self.open_chart, ticker, data.get("name", ticker))
                    lbl.bind("<Button-1>", lambda e, cmd=chart_cmd: cmd())
                
                labels.append(lbl)

            click_cmd = partial(self.show_details_popup, ticker)
            frame.bind("<Double-1>", lambda e: click_cmd())
            for idx, lbl in enumerate(labels):
                if idx != 0:
                    lbl.bind("<Double-1>", lambda e: click_cmd())

            self.rows.append({"frame": frame, "labels": labels, "data": data, "ticker": ticker})

        fetched = len(self.stock_data)
        total = len(self.current_tickers)
        self.status_lbl.config(text=f"Fetched {fetched}/{total} stock(s).")

    def open_chart(self, ticker: str, stock_name: str = "") -> None:
        """Open historical chart window for a stock."""
        try:
            open_chart_window(self.root, self.theme, ticker, stock_name, period="1y")
        except Exception as e:
            messagebox.showerror("Chart Error", f"Failed to open chart for {ticker}: {str(e)}")

    def _normalize_ticker(self, t: str) -> str:
        """Normalize ticker symbol by applying exchange suffix mappings."""
        t = t.upper().strip()
        for suf, norm in TICKER_SUFFIX_MAP.items():
            if t.endswith(suf):
                return t.replace(suf, norm)
        return t

    def open_search_dialog(self) -> None:
        """Open the stock search dialog for adding new tickers."""
        def add_ticker_callback(ticker: str) -> bool:
            norm = self._normalize_ticker(ticker)
            if norm in self.current_tickers:
                messagebox.showinfo("Duplicate", f"'{norm}' is already in your list.")
                return False
            
            self.current_tickers.insert(0, norm)
            self._update_list_display()
            self.unsaved_changes = True
            self.status_lbl.config(text=f"Added {norm} - Click 'Fetch Data' to analyze")
            return True
        
        open_stock_search(self.root, self.theme, add_ticker_callback)

    def remove_selected(self) -> None:
        """Open dialog to remove one or more tickers from the list."""
        if not self.rows:
            messagebox.showinfo("Empty List", "No stocks to remove.")
            return
        
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
        
        list_frame = tk.Frame(remove_dialog, bg=self.theme["background"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,
            bg=self.theme["entry"],
            fg=self.theme["text"],
            font=("Consolas", 10),
            yscrollcommand=scrollbar.set
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for ticker in self.current_tickers:
            data = next((d for d in self.stock_data if d["ticker"] == ticker), None)
            display_text = ticker
            if data:
                display_text += f" - {data['name']}"
            listbox.insert(tk.END, display_text)
        
        btn_frame = tk.Frame(remove_dialog, bg=self.theme["background"])
        btn_frame.pack(pady=10)
        
        def do_remove():
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("No Selection", "Please select at least one stock to remove.")
                return
            
            tickers_to_remove = [self.current_tickers[i] for i in sorted(selected_indices, reverse=True)]
            
            for ticker in tickers_to_remove:
                self.current_tickers.remove(ticker)
                self.stock_data = [d for d in self.stock_data if d["ticker"] != ticker]
            
            self._update_list_display()
            self.unsaved_changes = True
            remove_dialog.destroy()
            messagebox.showinfo("Removed", f"Removed {len(tickers_to_remove)} stock(s).")
        
        def do_remove_all():
            if messagebox.askyesno("Remove All", "Are you sure you want to remove ALL stocks from the list?"):
                self.current_tickers.clear()
                self.stock_data.clear()
                self._update_list_display()
                self.unsaved_changes = True
                remove_dialog.destroy()
                messagebox.showinfo("Removed", "All stocks removed.")
        
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

    def set_custom_period(self) -> None:
        """Open dialog to set custom lookback period for historical analysis."""
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
            try:
                days = int(entry.get())
                if days <= 0:
                    raise ValueError
                CONFIG.custom_period_days = days
                
                self.header_labels[6].config(text=f"{days} Day %")
                self.unsaved_changes = True
                pop.destroy()
                
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
        """Save the current ticker list to file."""
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
            
            self.header_labels[6].config(text=f"{custom_period} Day %")
            
            style_name = get_aggression_label(self.trading_aggression)
            self.trading_style_var.set(style_name)
            self.rec_mode_var.set(rec_mode)
            self.current_list_name = fn
            self.stock_data.clear()
            self._update_list_display()
            self.list_name_lbl.config(text=os.path.basename(fn))
            self.unsaved_changes = False

    def _load_default_list(self, silent: bool = False) -> None:
        """Load the default ticker list if one is configured."""
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
            
            self.header_labels[6].config(text=f"{custom_period} Day %")
            
            style_name = get_aggression_label(self.trading_aggression)
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
        """Display a popup with info."""
        info_text = (
            "=== INTERFACE ===\n\n"
            "Double-click rows for detailed info\n\n"
            "Click 📊 icon for charts\n\n"
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
        """Display detailed information popup for a specific stock."""
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
        
        detail_text = f"Ticker: {ticker}\n"
        detail_text += f"Name: {data['name']}\n"
        detail_text += f"Recommendation: {data['recommendation']}"
        
        if self.recommendation_mode == "complex" and metrics.get("Score") is not None:
            detail_text += f" (Score: {metrics['Score']:+.1f})"
        
        detail_text += f"\nReasons: {', '.join(data['reasons']) if data['reasons'] else 'None'}\n"
        detail_text += f"Sector/Industry: {data['sector']} / {data['industry']}\n"
        detail_text += "--- Price & Performance ---\n"
        detail_text += f"Current Price: {info.get('regularMarketPrice', 'N/A'):.2f}\n"
        detail_text += f"Previous Close: {info.get('previousClose', 'N/A'):.2f}\n"
        
        volume = info.get('volume')
        if volume:
            vol_str = format_volume(volume)
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
        """Fetch and analyze data for a single stock ticker."""
        try:
            ticker = self._normalize_ticker(ticker)
            stock = yf.Ticker(ticker)
            info = stock.info
            
            hist_1d = stock.history(period="2d", interval="1h")
            period_str = f"{CONFIG.custom_period_days}d"
            hist_long = stock.history(period=period_str)

            name = (info.get("shortName") or "N/A")[:30]
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            price = info.get("regularMarketPrice")

            swing_1d: Optional[float] = None
            if not hist_1d.empty and len(hist_1d) >= 2:
                close = hist_1d["Close"]
                swing_1d = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

            swing_1m: Optional[float] = None
            if not hist_long.empty and len(hist_long) >= 2:
                close = hist_long["Close"]
                swing_1m = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

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
            log_and_show("Fetch Error", f"{ticker}: {e}", "fetch_stock_data", ticker)
            return {
                "ticker": ticker, "name": "Error", "sector": "N/A", "industry": "N/A",
                "info": {}, "recommendation": "Hold", "reasons": [], "metrics": {},
                "price_swing_1d": "N/A", "price_swing_1m": "N/A"
            }

    def _calculate_simple_recommendation(self, info: dict, price: float, hist_long: pd.DataFrame,
                                        metrics: Dict[str, Any], swing_1d: Optional[float]) -> Tuple[str, List[str]]:
        """Calculate recommendation using simple legacy logic."""
        rec = "Hold"
        reasons: List[str] = []
        thresholds = self._get_adjusted_thresholds()

        if swing_1d is not None and abs(swing_1d) >= CONFIG.price_swing_threshold:
            reasons.append(f"1-day {swing_1d:+.2f}%")

        if price and "fiftyDayAverage" in info:
            ma = info["fiftyDayAverage"]
            if price < ma * thresholds["buy_ma_ratio"]:
                rec = "Buy"
                reasons.append("Below 50-day MA")
            elif price < ma * thresholds["consider_buy_ma_ratio"]:
                rec = "Consider Buying"
                reasons.append("Near 50-day MA")

        if price and "fiftyTwoWeekHigh" in info:
            high = info["fiftyTwoWeekHigh"]
            if price > high * thresholds["sell_high_ratio"]:
                rec = "Sell"
                reasons.append("Near 52-week high")
            elif price > high * thresholds["consider_sell_high_ratio"] and rec == "Hold":
                rec = "Consider Selling"
                reasons.append("Approaching high")

        rsi = metrics.get("RSI")
        if rsi is not None:
            if rsi > thresholds["rsi_overbought"]:
                reasons.append("Overbought")
            elif rsi < thresholds["rsi_oversold"]:
                reasons.append("Oversold")
                
        return rec, reasons

    def _calculate_complex_recommendation(self, info: dict, price: float, hist_long: pd.DataFrame,
                                         metrics: Dict[str, Any], swing_1d: Optional[float]) -> Tuple[str, List[str], float]:
        """Calculate recommendation using complex multi-factor scoring system."""
        reasons: List[str] = []
        total_score = 0.0
        weights = CONFIG.score_weights
        thresholds = self._get_adjusted_thresholds()
        
        # Factor 1: 50-Day Moving Average Position
        if price and "fiftyDayAverage" in info:
            ma = info["fiftyDayAverage"]
            ma_ratio = (price / ma - 1) * 100
            ma_score = max(-100, min(100, -ma_ratio * 5))
            total_score += ma_score * weights["ma_position"]
            
            if ma_ratio < -3:
                reasons.append(f"Below MA ({ma_ratio:.1f}%)")
            elif ma_ratio > 2:
                reasons.append(f"Above MA (+{ma_ratio:.1f}%)")
        
        # Factor 2: 52-Week High Position
        if price and "fiftyTwoWeekHigh" in info and "fiftyTwoWeekLow" in info:
            high = info["fiftyTwoWeekHigh"]
            low = info["fiftyTwoWeekLow"]
            if high > low:
                range_position = ((price - low) / (high - low)) * 100
                high_score = 100 - (range_position * 2)
                total_score += high_score * weights["high_position"]
                
                if range_position > 90:
                    reasons.append(f"Near 52W high ({range_position:.0f}%)")
                elif range_position < 20:
                    reasons.append(f"Near 52W low ({range_position:.0f}%)")
        
        # Factor 3: RSI Indicator
        rsi = metrics.get("RSI")
        if rsi is not None:
            if rsi > 50:
                rsi_score = -(rsi - 50) * 2
            else:
                rsi_score = (50 - rsi) * 2
            total_score += rsi_score * weights["rsi"]
            
            if rsi > thresholds["rsi_overbought"]:
                reasons.append(f"Overbought (RSI {rsi:.0f})")
            elif rsi < thresholds["rsi_oversold"]:
                reasons.append(f"Oversold (RSI {rsi:.0f})")
        
        # Factor 4: MACD Momentum
        macd = metrics.get("MACD")
        if macd is not None:
            macd_score = max(-100, min(100, macd * 20))
            total_score += macd_score * weights["macd"]
            
            if macd > 0.5:
                reasons.append(f"Positive momentum (MACD {macd:+.2f})")
            elif macd < -0.5:
                reasons.append(f"Negative momentum (MACD {macd:+.2f})")
        
        # Factor 5: P/E Ratio Valuation
        pe = metrics.get("P/E")
        if pe and pe > 0:
            if pe < 15:
                pe_score = 50
            elif pe > 30:
                pe_score = -50
            else:
                pe_score = 50 - ((pe - 15) / 15 * 100)
            total_score += pe_score * weights["pe_ratio"]
            
            if pe > 30:
                reasons.append(f"High P/E ({pe:.1f})")
            elif pe < 10:
                reasons.append(f"Low P/E ({pe:.1f})")
        
        # Factor 6: Analyst Target Price
        target_pct = metrics.get("Target %")
        if target_pct is not None:
            target_score = max(-100, min(100, target_pct * 2))
            total_score += target_score * weights["analyst_target"]
            
            if target_pct > 10:
                reasons.append(f"Upside potential ({target_pct:+.0f}%)")
            elif target_pct < -10:
                reasons.append(f"Downside risk ({target_pct:+.0f}%)")
        
        if swing_1d is not None and abs(swing_1d) >= CONFIG.price_swing_threshold:
            reasons.append(f"1-day {swing_1d:+.2f}%")
        
        # Adjust score based on trading aggression
        aggression_multiplier = 0.5 + (self.trading_aggression * 0.5)
        adjusted_score = total_score * aggression_multiplier
        
        # Convert score to recommendation
        buy_threshold = 30 - (self.trading_aggression * 20)
        consider_buy_threshold = 15 - (self.trading_aggression * 10)
        sell_threshold = -30 + (self.trading_aggression * 20)
        consider_sell_threshold = -15 + (self.trading_aggression * 10)
        
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
        """Fetch data for all tickers and update the display."""
        if not self.current_tickers:
            messagebox.showinfo("No Tickers", "Add tickers to track using the Search & Add button.")
            return

        self.status_lbl.config(text="Fetching data...")
        self.root.update_idletasks()
        for btn in self.button_refs.values():
            btn.config(state=tk.DISABLED)

        new_data: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=CONFIG.max_threads) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_stock_data, t): t
                for t in self.current_tickers
            }
            for i, future in enumerate(as_completed(future_to_ticker)):
                try:
                    data = future.result()
                    new_data.append(data)
                    self.status_lbl.config(text=f"Fetched data for {data['ticker']} ({i+1}/{len(self.current_tickers)})...")
                    self.root.update_idletasks()
                except Exception as e:
                    log_and_show("Thread Error", f"Error fetching data for a ticker: {e}", "fetch_and_display")

        self.stock_data = new_data
        self._update_list_display()

        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated_lbl.config(text=f"Last updated: {self.last_updated}")

        for btn in self.button_refs.values():
            btn.config(state=tk.NORMAL)

    def _on_closing(self) -> None:
        """Handle window close event, prompting to save unsaved changes."""
        if self.unsaved_changes:
            if messagebox.askyesno("Exit", "Unsaved changes! Save before exiting?"):
                self.save_current_list()
        self.root.destroy()