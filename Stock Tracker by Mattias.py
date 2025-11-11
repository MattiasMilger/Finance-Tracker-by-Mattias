"""Stock Tracker Application using Tkinter and yFinance for real-time stock data analysis."""
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
    min_window_width: int = 900
    min_window_height: int = 600
    lists_dir: str = "ticker_lists"
    default_list_file: str = "default_list.txt"
    max_threads: int = 3
    price_swing_threshold: float = 5.0
    custom_period_days: int = 30

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
        "tag_sell": "#663333",
        "tag_consider_sell": "#666633",
        "tag_buy": "#336633",
        "tag_consider_buy": "#336666",
        "tag_hold": "#2d2d2d",
    }
}

TICKER_SUFFIX_MAP = {
    ".ST": ".ST", ".STO": "", ".MI": ".MI", ".DE": ".DE",
    ".L": ".L", ".PA": ".PA", ".T": ".T", ".HK": ".HK",
    ".SS": ".SS", ".SZ": ".SZ", ".TO": ".TO", ".AX": ".AX",
    ".NS": ".NS", ".BO": ".BO",
}

logging.basicConfig(
    level=logging.INFO,
    filename="stock_tracker.log",
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def log_and_show(title: str, message: str, func_name: str, ticker: Optional[str] = None, msg_type: str = "error") -> None:
    log_msg = f"{title}: {message}"
    if ticker:
        log_msg += f" (Ticker: {ticker})"
    log_msg += f" in {func_name}"
    logging.error(log_msg)
    getattr(messagebox, f"show{msg_type}")(title, message)


def load_ticker_list(filename: str) -> List[str]:
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(t).strip().upper() for t in data if str(t).strip()]
    except Exception as e:
        log_and_show("Load Error", f"Failed to load ticker list: {e}", "load_ticker_list", msg_type="warning")
        return []


def save_ticker_list(filename: str, tickers: List[str]) -> None:
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(tickers, f, indent=2)
    except Exception as e:
        log_and_show("Save Error", f"Failed to save ticker list: {e}", "save_ticker_list")


def export_to_csv(stock_data: List[Dict[str, Any]]) -> None:
    """Export stock data to CSV with Save As dialog."""
    if not stock_data:
        messagebox.showwarning("Export Error", "No stock data to export.")
        return

    # Open Save As dialog
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


# --------------------------------------------------------------------------- #
# Metric Functions
# --------------------------------------------------------------------------- #
def get_pe_ratio(info: dict) -> Tuple[Optional[float], None]:
    return info.get("trailingPE"), None


def get_analyst_target(info: dict, price: float) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    target = info.get("targetMeanPrice")
    if target and price:
        diff = (target - price) / price * 100
        return target, diff, "Potential Upside" if price < target else None
    return None, None, None


def get_rsi(hist: pd.DataFrame, period: int = 14) -> Tuple[Optional[float], Optional[str]]:
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
    if hist.empty or "Close" not in hist:
        return None, None
    close = hist["Close"]
    ema_s = close.ewm(span=short, adjust=False).mean()
    ema_l = close.ewm(span=long, adjust=False).mean()
    macd_line = ema_s - ema_l
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.iloc[-1], sig_line.iloc[-1]


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        self._metrics[name] = func

    def compute(self, name: str, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
        return self._metrics.get(name, lambda *_, **__: (None, None))(*args, **kwargs)


registry = MetricRegistry()
registry.register("pe_ratio", get_pe_ratio)
registry.register("analyst_target", get_analyst_target)
registry.register("rsi", get_rsi)
registry.register("macd", get_macd)


# --------------------------------------------------------------------------- #
# Main Application
# --------------------------------------------------------------------------- #
class StockTrackerApp:
    def __init__(self, root: tk.Tk, theme: str = "dark") -> None:
        self.root = root
        self.root.title("Stock Tracker by Mattias")
        self.root.minsize(CONFIG.min_window_width, CONFIG.min_window_height)
        self.theme = THEMES[theme]
        self.root.configure(bg=self.theme["background"])

        self.button_refs: Dict[str, tk.Button] = {}
        self.stock_data: List[Dict[str, Any]] = []
        self.current_tickers: List[str] = []
        self.current_list_name: str = ""
        self.unsaved_changes: bool = False
        self.filter_query: str = "" # New member for search state

        self._sort_column: str = "Recommendation"
        self._sort_reverse: bool = False

        self.rows = []
        self.header_labels = []
        self.search_entry: Optional[tk.Entry] = None # New member for search input

        self._setup_menu()
        self._setup_ui()
        self._load_default_list(silent=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_list)
        file_menu.add_command(label="Open", command=self.open_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_current_list)
        file_menu.add_command(label="Save as", command=self.save_list_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data As CSV", command=lambda: export_to_csv(self.stock_data))
        file_menu.add_separator()
        file_menu.add_command(label="Set as Default", command=self.set_as_default)
        file_menu.add_command(label="Remove Default", command=self.remove_default)

        period_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Period", menu=period_menu)
        period_menu.add_command(label="Set Custom Period (Days)...", command=self.set_custom_period)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Info – Metric Explanations", command=self.show_info_popup)

    def _setup_ui(self) -> None:
        mgmt = tk.LabelFrame(
            self.root, text="Ticker List & Analysis",
            bg=self.theme["background"], fg=self.theme["text"], padx=10, pady=5
        )
        mgmt.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        top_frame = tk.Frame(mgmt, bg=self.theme["background"])
        top_frame.pack(fill=tk.X, pady=(0, 5))

        # --- List Info ---
        name_frame = tk.Frame(top_frame, bg=self.theme["background"])
        name_frame.pack(side=tk.LEFT)
        tk.Label(name_frame, text="Current List:", bg=self.theme["background"], fg=self.theme["text"]).pack(side=tk.LEFT)
        self.list_name_lbl = tk.Label(
            name_frame, text="(none)", bg=self.theme["background"],
            fg=self.theme["text"], font=("Arial", 9, "italic")
        )
        self.list_name_lbl.pack(side=tk.LEFT, padx=(5, 0))

        # --- Add Ticker ---
        add_frame = tk.Frame(top_frame, bg=self.theme["background"])
        add_frame.pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(add_frame, text="Add Ticker:", bg=self.theme["background"], fg=self.theme["text"]).pack(side=tk.LEFT)
        self.add_entry = tk.Entry(
            add_frame, width=15, bg=self.theme["entry"], fg=self.theme["text"],
            insertbackground=self.theme["text"]
        )
        self.add_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.add_entry.bind("<Return>", lambda e: self.add_ticker_from_entry())
        tk.Button(
            add_frame, text="Add", command=self.add_ticker_from_entry,
            bg=self.theme["button"], fg=self.theme["text"]
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            add_frame, text="Add Stock", command=self.open_search_dialog,
            bg=self.theme["button"], fg=self.theme["text"], width=20
        ).pack(side=tk.LEFT, padx=2)
        
        # --- Search Functionality ---
        search_frame = tk.Frame(top_frame, bg=self.theme["background"])
        search_frame.pack(side=tk.LEFT, padx=(20, 0))
        tk.Label(search_frame, text="Search in list:", bg=self.theme["background"], fg=self.theme["text"]).pack(side=tk.LEFT)
        self.search_entry = tk.Entry(
            search_frame, width=20, bg=self.theme["entry"], fg=self.theme["text"],
            insertbackground=self.theme["text"]
        )
        self.search_entry.pack(side=tk.LEFT, padx=(5, 0))
        self.search_entry.bind("<Return>", lambda e: self.filter_list(self.search_entry.get()))
        tk.Button(
            search_frame, text="Search", command=lambda: self.filter_list(self.search_entry.get()),
            bg=self.theme["button"], fg=self.theme["text"]
        ).pack(side=tk.LEFT, padx=5)
        # The 'Clear Filter' button has been removed as requested.


        action_frame = tk.Frame(top_frame, bg=self.theme["background"])
        action_frame.pack(expand=True)

        left_actions = tk.Frame(action_frame, bg=self.theme["background"])
        left_actions.pack(side=tk.LEFT, padx=20)
        for txt, cmd in [("Remove", self.remove_selected), ("Clear All", self.clear_all)]:
            btn = tk.Button(left_actions, text=txt, command=cmd, width=12,
                            bg=self.theme["button"], fg=self.theme["text"])
            btn.pack(side=tk.LEFT, padx=2)
            self.button_refs[txt] = btn

        fetch_btn = tk.Button(
            action_frame, text="Fetch Data", command=self.fetch_and_display,
            bg=self.theme["button"], fg=self.theme["text"], font=("Arial", 11, "bold"), width=15, height=1
        )
        fetch_btn.pack(side=tk.LEFT, padx=40)
        self.button_refs["Fetch Data"] = fetch_btn

        # === CANVAS TABLE ===
        table_container = tk.Frame(mgmt, bg=self.theme["background"])
        table_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(table_container, bg=self.theme["tree_bg"], highlightthickness=0)
        v_scroll = tk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scroll = tk.Scrollbar(mgmt, orient=tk.HORIZONTAL, command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.table_frame = tk.Frame(self.canvas, bg=self.theme["tree_bg"])
        self.canvas.create_window((0, 0), window=self.table_frame, anchor="nw")

        self.table_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        def _on_mousewheel(event):
            self.canvas.yview_scroll(-1*(event.delta//120), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

        # === HEADER ===
        self.header_frame = tk.Frame(self.table_frame, bg=self.theme["tree_heading_bg"])
        self.header_frame.pack(fill=tk.X)

        headers = [
            "Ticker", "Name", "Recommendation", "Price",
            "1 Day %", f"{CONFIG.custom_period_days} Day %",
            "Sector", "Industry", "P/E", "Target %", "RSI", "MACD"
        ]
        char_widths = [10, 25, 16, 10, 10, 12, 14, 16, 8, 10, 8, 10]

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
            lbl.grid(row=0, column=i, padx=(0, 1), pady=1, sticky="ew")
            lbl.bind("<Button-1>", lambda e, col=text: self._sort_by_column(col))
            self.header_labels.append(lbl)

        # === BOTTOM ===
        bottom = tk.Frame(self.root, bg=self.theme["background"])
        bottom.pack(pady=10)
        exit_btn = tk.Button(bottom, text="Exit", command=self._on_closing, width=15,
                             bg=self.theme["button"], fg=self.theme["text"])
        exit_btn.pack()
        self.button_refs["Exit"] = exit_btn

        self.status_lbl = tk.Label(
            self.root, text="Ready", bg=self.theme["background"],
            fg=self.theme["text"], font=("Arial", 10)
        )
        self.status_lbl.pack(pady=(0, 5))
        
    def filter_list(self, query: str) -> None:
        """Filter the displayed stock rows based on the ticker or name."""
        self.filter_query = query.strip().upper()
        if self.search_entry:
            if not self.filter_query:
                self.search_entry.delete(0, tk.END)
                self.status_lbl.config(text="Filter cleared. Showing all stocks.")
            else:
                self.status_lbl.config(text=f"Filtered by: '{query}'")

        # Sort the rows first, so that the displayed order is correct
        self._sort_by_column(self._sort_column) 

        filtered_count = 0
        
        # Iterate over the already sorted rows and show/hide the frame
        for row in self.rows:
            ticker = row['ticker']
            name = row['data'].get('name', 'N/A') if row['data'] else 'N/A'
            
            # Check if the query is in the ticker or the name (case-insensitive)
            if not self.filter_query or self.filter_query in ticker or self.filter_query in name.upper():
                row["frame"].pack(fill=tk.X, pady=1)
                filtered_count += 1
            else:
                row["frame"].pack_forget()

        if self.filter_query:
            self.status_lbl.config(text=f"Showing {filtered_count}/{len(self.current_tickers)} stock(s) matching '{query}'.")

    def _update_sort_indicator(self):
        for lbl in self.header_labels:
            base = lbl.cget("text").split(" ")[0]
            if base == self._sort_column or (base.endswith("Day") and self._sort_column in ["1 Day %", f"{CONFIG.custom_period_days} Day %"]):
                arrow = "Down" if self._sort_reverse else "Up"
                lbl.config(text=base + " " + arrow)
            else:
                lbl.config(text=base)

    def _sort_by_column(self, col_text: str):
        col_text = col_text.split(" ")[0]
        if col_text.endswith("Day"):
            col_text = "1 Day %" if "1" in col_text else f"{CONFIG.custom_period_days} Day %"

        if self._sort_column == col_text:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = col_text
            self._sort_reverse = False

        self._update_sort_indicator()

        col_idx = next((i for i, h in enumerate(self.header_labels) if h.cget("text").split(" ")[0] == col_text), None)
        if col_idx is None:
            return
            
        # Special handling for "Recommendation" column sorting
        if self._sort_column == "Recommendation":
            # Use the priority map for Recommendation sorting
            priority_map = {
                "Sell": 0,
                "Consider Selling": 1, 
                "Buy": 2,
                "Consider Buying": 3,
                "Hold": 4, 
            }
            def rec_key_func(row):
                data = row["data"]
                if not data: return (5, "Z") # Unknown data goes last
                rec = data.get("recommendation", "Unknown")
                return (priority_map.get(rec, 5), rec)
                
            self.rows.sort(key=rec_key_func, reverse=self._sort_reverse)
        else:
            def key_func(row):
                val = row["labels"][col_idx].cget("text")
                if val in ("N/A", "", "Error"):
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
        
        # Re-pack the sorted rows, respecting the current filter
        for row in self.rows:
            row["frame"].pack_forget()

        for row in self.rows:
            ticker = row['ticker']
            name = row['data'].get('name', 'N/A') if row['data'] else 'N/A'
            if not self.filter_query or self.filter_query in ticker or self.filter_query in name.upper():
                row["frame"].pack(fill=tk.X, pady=1)


    def _update_list_display(self) -> None:
        for row in self.rows:
            row["frame"].destroy()
        self.rows.clear()
        self.filter_query = "" # Clear filter on full list update
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
                # UPDATED SORT PRIORITY LOGIC
                # Lower number = higher priority (comes first/on top when sorted by ascending)
                priority = {
                    "Sell": 0,           # Highest Priority
                    "Consider Selling": 1,
                    "Buy": 2,            
                    "Consider Buying": 3,
                    "Hold": 4,           # Lowest Actionable Priority
                }.get(data["recommendation"], 5)
            display_data.append((ticker, data, priority))

        # We primarily use the sort function now, but this initial sort 
        # is still useful for initial list display when no explicit sort
        # has been applied, and it will be overridden by _sort_by_column 
        # later if a column is clicked.
        display_data.sort(key=lambda x: x[2]) 

        char_widths = [10, 25, 16, 10, 10, 12, 14, 16, 8, 10, 8, 10]

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
            values = [
                ticker,
                data["name"] if data else "",
                data["recommendation"] if data else "",
                f"{data['info'].get('regularMarketPrice', ''):.2f}" if data and data['info'].get('regularMarketPrice') is not None else "",
                data.get("price_swing_1d", "N/A") if data else "N/A",
                data.get("price_swing_1m", "N/A") if data else "N/A",
                data["sector"] if data else "",
                data["industry"] if data else "",
                f"{data['metrics'].get('P/E', ''):.1f}" if data and data['metrics'].get('P/E') else "",
                f"{data['metrics'].get('Target %', ''):+.1f}%" if data and data['metrics'].get('Target %') else "",
                f"{data['metrics'].get('RSI', ''):.1f}" if data and data['metrics'].get('RSI') else "",
                f"{data['metrics'].get('MACD', ''):+.3f}" if data and data['metrics'].get('MACD') else ""
            ]

            for i, (val, width) in enumerate(zip(values, char_widths)):
                fg = self.theme["tree_fg"]
                if i in [4, 5] and isinstance(val, str) and "%" in val and val != "N/A":
                    try:
                        num = float(val.replace("%", "").replace("+", "").replace(" ", ""))
                        fg = "#66ff99" if num > 0 else "#ff6b6b" if num < 0 else "#cccccc"
                    except:
                        pass
                lbl = tk.Label(
                    frame,
                    text=val,
                    bg=bg_color,
                    fg=fg,
                    font=("Consolas", 10),
                    width=width,
                    anchor="center",
                    relief="flat"
                )
                lbl.grid(row=0, column=i, padx=(0, 1), sticky="ew")
                labels.append(lbl)

            click_cmd = partial(self.show_details_popup, ticker)
            frame.bind("<Double-1>", lambda e: click_cmd())
            for lbl in labels:
                lbl.bind("<Double-1>", lambda e: click_cmd())

            self.rows.append({"frame": frame, "labels": labels, "data": data, "ticker": ticker})

        fetched = len(self.stock_data)
        total = len(self.current_tickers)
        self.status_lbl.config(text=f"Fetched {fetched}/{total} stock(s).")

    # === ALL OTHER METHODS (add_ticker, fetch, etc.) remain unchanged ===
    def _normalize_ticker(self, t: str) -> str:
        t = t.upper().strip()
        for suf, norm in TICKER_SUFFIX_MAP.items():
            if t.endswith(suf):
                return t.replace(suf, norm)
        return t

    def _validate_ticker(self, t: str) -> bool:
        t = t.strip()
        if not t or len(t) > 10 or not all(c.isalnum() or c in ".-" for c in t):
            return False
        return True

    def _ticker_exists(self, t: str) -> bool:
        try:
            info = yf.Ticker(t).info
            return bool(info.get("regularMarketPrice") or info.get("shortName"))
        except Exception:
            return False

    def open_search_dialog(self) -> None:
        """Open the stock search dialog."""
        def add_ticker_callback(ticker: str) -> None:
            """Callback to add ticker from search dialog."""
            norm = self._normalize_ticker(ticker)
            if norm in self.current_tickers:
                messagebox.showinfo("Duplicate", f"'{norm}' is already in your list.")
                return
            
            self.current_tickers.insert(0, norm)
            self._update_list_display()
            self.unsaved_changes = True
            self.status_lbl.config(text=f"Added {norm} - Click 'Fetch Data' to analyze")
        
        open_stock_search(self.root, self.theme, add_ticker_callback)

    def add_ticker_from_entry(self) -> None:
        raw = self.add_entry.get().strip().upper()
        self.add_entry.delete(0, tk.END)
        if not raw or not self._validate_ticker(raw):
            messagebox.showwarning("Invalid", f"'{raw}' is invalid or too long.")
            return
        norm = self._normalize_ticker(raw)
        if norm in self.current_tickers:
            messagebox.showinfo("Duplicate", f"'{norm}' already in list.")
            return

        self.status_lbl.config(text=f"Checking {norm}...")
        self.root.update_idletasks()
        if not self._ticker_exists(norm):
            messagebox.showerror("Invalid Ticker", f"'{norm}' does not exist.")
            self.status_lbl.config(text="Ready")
            return

        self.current_tickers.insert(0, norm)
        self.stock_data.clear() # Clear data so it's re-fetched
        self._update_list_display()
        self.unsaved_changes = True
        self.status_lbl.config(text=f"Added {norm}")

    def remove_selected(self) -> None:
        self._update_list_display()
        self.unsaved_changes = True

    def clear_all(self) -> None:
        if not self.current_tickers:
            return
        if messagebox.askyesno("Clear All", "Remove all tickers?"):
            self.current_tickers.clear()
            self.stock_data.clear()
            self._update_list_display()
            self.unsaved_changes = True

    def set_custom_period(self) -> None:
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
                
                # Update header label for 1-month swing
                self.header_labels[5].config(text=f"{days} Day %")
                
                pop.destroy()
                messagebox.showinfo("Updated", f"Custom period set to {days} days.")
                if self.stock_data:
                    self.fetch_and_display()
                else:
                    self._update_list_display()
            except ValueError:
                messagebox.showerror("Invalid", "Please enter a positive integer.")

        tk.Button(pop, text="OK", command=save, bg=self.theme["button"], fg=self.theme["text"]).pack(pady=5)

    def new_list(self) -> None:
        if self.unsaved_changes and messagebox.askyesnocancel("Unsaved Changes", "Save current list before creating new?"):
            self.save_current_list()
        self.current_tickers.clear()
        self.stock_data.clear()
        self.current_list_name = ""
        self._update_list_display()
        self.list_name_lbl.config(text="(none)")
        self.unsaved_changes = False

    def save_current_list(self) -> None:
        if not self.current_list_name:
            self.save_list_as()
            return
        save_ticker_list(self.current_list_name, self.current_tickers)
        self.unsaved_changes = False
        messagebox.showinfo("Saved", f"List saved as '{os.path.basename(self.current_list_name)}'")

    def save_list_as(self) -> None:
        fn = filedialog.asksaveasfilename(
            initialdir=CONFIG.lists_dir, title="Save Ticker List",
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if fn:
            save_ticker_list(fn, self.current_tickers)
            self.current_list_name = fn
            self.list_name_lbl.config(text=os.path.basename(fn))
            self.unsaved_changes = False

    def open_dialog(self) -> None:
        if self.unsaved_changes and messagebox.askyesnocancel("Unsaved Changes", "Save current list before loading?"):
            self.save_current_list()
        fn = filedialog.askopenfilename(
            initialdir=CONFIG.lists_dir, title="Load Ticker List",
            filetypes=[("JSON files", "*.json")]
        )
        if fn:
            self.current_tickers = load_ticker_list(fn)
            self.current_list_name = fn
            self.stock_data.clear()
            self._update_list_display()
            self.list_name_lbl.config(text=os.path.basename(fn))
            self.unsaved_changes = False

    def _load_default_list(self, silent: bool = False) -> None:
        path = os.path.join(CONFIG.lists_dir, CONFIG.default_list_file)
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            name = f.read().strip()
        full = os.path.join(CONFIG.lists_dir, name)
        if os.path.exists(full):
            self.current_tickers = load_ticker_list(full)
            self.current_list_name = full
            self.stock_data.clear() # Clear data to force a re-fetch
            self._update_list_display()
            self.list_name_lbl.config(text=name)
            self.unsaved_changes = False
            if not silent:
                messagebox.showinfo("Default Loaded", f"Loaded default list: {name}")

    def set_as_default(self) -> None:
        if not self.current_list_name:
            messagebox.showwarning("No List", "Save the current list first.")
            return
        path = os.path.join(CONFIG.lists_dir, CONFIG.default_list_file)
        with open(path, "w", encoding="utf-8") as f:
            f.write(os.path.basename(self.current_list_name))
        messagebox.showinfo("Default Set", f"'{os.path.basename(self.current_list_name)}' is now default.")

    def remove_default(self) -> None:
        path = os.path.join(CONFIG.lists_dir, CONFIG.default_list_file)
        if os.path.exists(path):
            os.remove(path)
            messagebox.showinfo("Default Removed", "Default list removed.")
        else:
            messagebox.showinfo("No Default", "No default list is set.")
            
    def show_info_popup(self, event=None) -> None:
        """Shows a popup with explanations of the metrics."""
        info_text = (
            "Stock Tracker Metrics Explained:\n\n"
            "Recommendation: Derived from comparing the current price to the 50-day moving average (MA) "
            "and the 52-week high, with configurable thresholds.\n"
            "P/E: Trailing Price-to-Earnings Ratio. High P/E (e.g., >30) may suggest overvaluation.\n"
            "Target %: Percentage difference between the Analyst Mean Target Price and the current price.\n"
            "RSI: Relative Strength Index (14-day). >70 is Overbought (potential sell), <30 is Oversold (potential buy).\n"
            "MACD: Moving Average Convergence Divergence (12/26 periods). Helps identify momentum and trend direction.\n"
            "Day %: Price swing percentage for 1 day and the custom period (currently set to "
            f"{CONFIG.custom_period_days} days)."
        )

        pop = tk.Toplevel(self.root)
        pop.title("Stock Tracker Metric Info")
        pop.configure(bg=self.theme["background"])
        pop.transient(self.root)
        pop.grab_set()

        tk.Label(pop, text=info_text, justify=tk.LEFT, padx=10, pady=10,
                 bg=self.theme["background"], fg=self.theme["text"]).pack()
        tk.Button(pop, text="Close", command=pop.destroy, bg=self.theme["button"], fg=self.theme["text"]).pack(pady=5)
        
    def show_details_popup(self, ticker: str) -> None:
        """Shows a detailed popup for the selected stock."""
        data = next((d for d in self.stock_data if d["ticker"] == ticker), None)
        if not data:
            messagebox.showerror("Error", f"No data found for {ticker}.")
            return

        pop = tk.Toplevel(self.root)
        pop.title(f"Details: {ticker} ({data['name']})")
        pop.configure(bg=self.theme["background"])
        pop.transient(self.root)
        pop.grab_set()

        # Build detailed information string
        info = data['info']
        metrics = data['metrics']
        
        detail_text = f"Ticker: {ticker}\n"
        detail_text += f"Name: {data['name']}\n"
        detail_text += f"Recommendation: {data['recommendation']} ({', '.join(data['reasons'])})\n"
        detail_text += f"Sector/Industry: {data['sector']} / {data['industry']}\n"
        detail_text += "--- Price & Performance ---\n"
        detail_text += f"Current Price: {info.get('regularMarketPrice', 'N/A'):.2f}\n"
        detail_text += f"Previous Close: {info.get('previousClose', 'N/A'):.2f}\n"
        detail_text += f"50-Day Avg: {info.get('fiftyDayAverage', 'N/A'):.2f}\n"
        detail_text += f"52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A'):.2f}\n"
        detail_text += f"52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A'):.2f}\n"
        detail_text += f"1 Day Swing: {data.get('price_swing_1d', 'N/A')}\n"
        detail_text += f"{CONFIG.custom_period_days} Day Swing: {data.get('price_swing_1m', 'N/A')}\n"
        detail_text += "--- Key Metrics ---\n"
        detail_text += f"Trailing P/E: {metrics.get('P/E', 'N/A'):.1f}\n"
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

            rec = "Hold"
            reasons: List[str] = []

            swing_1d: Optional[float] = None
            if not hist_1d.empty and len(hist_1d) >= 2:
                close = hist_1d["Close"]
                swing_1d = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
                if swing_1d is not None and abs(swing_1d) >= CONFIG.price_swing_threshold:
                    reasons.append(f"1-day {swing_1d:+.2f}%")

            swing_1m: Optional[float] = None
            if not hist_long.empty and len(hist_long) >= 2:
                close = hist_long["Close"]
                swing_1m = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

            if price and "fiftyDayAverage" in info:
                ma = info["fiftyDayAverage"]
                if price < ma * CONFIG.recommendation_thresholds["buy_ma_ratio"]:
                    rec = "Buy"
                    reasons.append("Below 50-day MA")
                elif price < ma * CONFIG.recommendation_thresholds["consider_buy_ma_ratio"]:
                    rec = "Consider Buying"
                    reasons.append("Near 50-day MA")

            if price and "fiftyTwoWeekHigh" in info:
                high = info["fiftyTwoWeekHigh"]
                if price > high * CONFIG.recommendation_thresholds["sell_high_ratio"]:
                    rec = "Sell"
                    reasons.append("Near 52-week high")
                elif price > high * CONFIG.recommendation_thresholds["consider_sell_high_ratio"] and rec == "Hold":
                    rec = "Consider Selling"
                    reasons.append("Approaching high")

            metrics: Dict[str, Any] = {}
            if CONFIG.enable_metrics["pe_ratio"]:
                metrics["P/E"] = registry.compute("pe_ratio", info)[0]
            if CONFIG.enable_metrics["analyst_target"]:
                tgt, diff, _ = registry.compute("analyst_target", info, price)
                metrics["Target"] = tgt
                metrics["Target %"] = diff
            if CONFIG.enable_metrics["rsi"]:
                rsi, flag = registry.compute("rsi", hist_long)
                metrics["RSI"] = rsi
                if flag:
                    reasons.append(flag)
            if CONFIG.enable_metrics["macd"]:
                macd, _ = registry.compute("macd", hist_long)
                metrics["MACD"] = macd

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

    def fetch_and_display(self) -> None:
        if not self.current_tickers:
            messagebox.showinfo("No Tickers", "Add tickers to track using the field above.")
            return

        self.status_lbl.config(text="Fetching data...")
        self.root.update_idletasks()
        for btn in self.button_refs.values():
            btn.config(state=tk.DISABLED)

        # Use a ThreadPoolExecutor for concurrent fetching
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

        for btn in self.button_refs.values():
            btn.config(state=tk.NORMAL)

    def _on_closing(self) -> None:
        if self.unsaved_changes:
            if messagebox.askyesno("Exit", "Unsaved changes! Save before exiting?"):
                self.save_current_list()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StockTrackerApp(root)
    root.mainloop()
