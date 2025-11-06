"""Stock Tracker Application using Tkinter and yFinance for real-time stock data analysis."""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import yfinance as yf


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
    price_swing_threshold: float = 5.0  # % swing warning
    custom_period_days: int = 30  # Default period

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
        "tree_select_bg": "#4a4a4a",
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
def log_and_show(
    title: str,
    message: str,
    func_name: str,
    ticker: Optional[str] = None,
    msg_type: str = "error"
) -> None:
    """Log error and show messagebox."""
    log_msg = f"{title}: {message}"
    if ticker:
        log_msg += f" (Ticker: {ticker})"
    log_msg += f" in {func_name}"
    logging.error(log_msg)
    getattr(messagebox, f"show{msg_type}")(title, message)


def load_ticker_list(filename: str) -> List[str]:
    """Load ticker list from JSON file."""
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
    """Save ticker list to JSON file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(tickers, f, indent=2)
    except Exception as e:
        log_and_show("Save Error", f"Failed to save ticker list: {e}", "save_ticker_list")


def export_to_csv(stock_data: List[Dict[str, Any]]) -> None:
    """Export stock data to CSV file."""
    if not stock_data:
        log_and_show("Export Error", "No stock data to export.", "export_to_csv")
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
        fn = f"stock_data_{datetime.now():%Y%m%d_%H%M%S}.csv"
        df.to_csv(fn, index=False)
        messagebox.showinfo("Success", f"Data exported to {fn}")
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
    """Registry for metric computation functions."""
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
        self.root.title("Finance Tracker by Mattias")
        self.root.minsize(CONFIG.min_window_width, CONFIG.min_window_height)
        self.theme = THEMES[theme]
        self.root.configure(bg=self.theme["background"])

        self.button_refs: Dict[str, tk.Button] = {}
        self.stock_data: List[Dict[str, Any]] = []
        self.current_tickers: List[str] = []
        self.current_list_name: str = ""
        self.unsaved_changes: bool = False

        # Sorting state
        self._sort_column: Optional[str] = None
        self._sort_reverse: bool = False

        self._setup_menu()
        self._setup_ui()
        self._load_default_list(silent=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ------------------------------------------------------------------- #
    # UI Construction
    # ------------------------------------------------------------------- #
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

        name_frame = tk.Frame(top_frame, bg=self.theme["background"])
        name_frame.pack(side=tk.LEFT)
        tk.Label(name_frame, text="Current List:", bg=self.theme["background"], fg=self.theme["text"]).pack(side=tk.LEFT)
        self.list_name_lbl = tk.Label(
            name_frame, text="(none)", bg=self.theme["background"],
            fg=self.theme["text"], font=("Arial", 9, "italic")
        )
        self.list_name_lbl.pack(side=tk.LEFT, padx=(5, 0))

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

        # Central action buttons
        action_frame = tk.Frame(top_frame, bg=self.theme["background"])
        action_frame.pack(expand=True)

        # Left side: Remove + Clear All (closer to Add)
        left_actions = tk.Frame(action_frame, bg=self.theme["background"])
        left_actions.pack(side=tk.LEFT, padx=20)
        for txt, cmd in [("Remove", self.remove_selected), ("Clear All", self.clear_all)]:
            btn = tk.Button(left_actions, text=txt, command=cmd, width=12,
                            bg=self.theme["button"], fg=self.theme["text"])
            btn.pack(side=tk.LEFT, padx=2)
            self.button_refs[txt] = btn

        # Center: Fetch Data (prominent)
        fetch_btn = tk.Button(
            action_frame, text="Fetch Data", command=self.fetch_and_display,
            bg=self.theme["button"], fg=self.theme["text"], font=("Arial", 11, "bold"), width=15, height=1
        )
        fetch_btn.pack(side=tk.LEFT, padx=40)
        self.button_refs["Fetch Data"] = fetch_btn

        # Right side: Export to CSV
        right_actions = tk.Frame(action_frame, bg=self.theme["background"])
        right_actions.pack(side=tk.LEFT)
        export_btn = tk.Button(
            right_actions, text="Export to CSV", command=lambda: export_to_csv(self.stock_data),
            width=14, bg=self.theme["button"], fg=self.theme["text"]
        )
        export_btn.pack(side=tk.LEFT, padx=2)
        self.button_refs["Export to CSV"] = export_btn

        tree_container = tk.Frame(mgmt, bg=self.theme["background"])
        tree_container.pack(fill=tk.BOTH, expand=True)

        columns = (
            "ticker", "name", "recommendation", "price", "1_day_%", "1_month_%",
            "sector", "industry", "pe", "target_diff", "rsi", "macd"
        )
        self.tree = ttk.Treeview(
            tree_container, columns=columns, show="headings", selectmode="extended"
        )
        col_widths = {
            "ticker": 80, "name": 180, "recommendation": 100,
            "price": 80, "1_day_%": 80, "1_month_%": 80,
            "sector": 100, "industry": 100,
            "pe": 70, "target_diff": 80, "rsi": 60, "macd": 80
        }
        for col, width in col_widths.items():
            heading = col.replace("_", " ").title()
            if col == "1_month_%":
                heading = f"{CONFIG.custom_period_days} Day %"
            self.tree.heading(col, text=heading, anchor="center")
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scroll = tk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=v_scroll.set)

        h_scroll = tk.Scrollbar(mgmt, orient=tk.HORIZONTAL, command=self.tree.xview)
        h_scroll.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.tree.configure(xscrollcommand=h_scroll.set)

        self.tree.bind("<Double-1>", self.show_details_popup)
        self._setup_sorting()

        bottom = tk.Frame(self.root, bg=self.theme["background"])
        bottom.pack(pady=10)
        exit_btn = tk.Button(bottom, text="Exit", command=self._on_closing, width=15,
                             bg=self.theme["button"], fg=self.theme["text"])
        exit_btn.pack()
        self.button_refs["Exit"] = exit_btn

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background=self.theme["tree_bg"],
                        foreground=self.theme["tree_fg"],
                        fieldbackground=self.theme["tree_bg"],
                        rowheight=22)
        style.configure("Treeview.Heading",
                        background=self.theme["tree_heading_bg"],
                        foreground=self.theme["text"])
        style.map("Treeview", background=[("selected", self.theme["tree_select_bg"])])

        for tag, color in [
            ("Sell", self.theme["tag_sell"]),
            ("Consider Selling", self.theme["tag_consider_sell"]),
            ("Buy", self.theme["tag_buy"]),
            ("Consider Buying", self.theme["tag_consider_buy"]),
            ("Hold", self.theme["tag_hold"])
        ]:
            self.tree.tag_configure(tag, background=color)

        self.status_lbl = tk.Label(
            self.root, text="Ready", bg=self.theme["background"],
            fg=self.theme["text"], font=("Arial", 10)
        )
        self.status_lbl.pack(pady=(0, 5))

    # ------------------------------------------------------------------- #
    # Sorting
    # ------------------------------------------------------------------- #
    def _setup_sorting(self) -> None:
        for col in self.tree["columns"]:
            self.tree.heading(col, command=lambda c=col: self._sort_by_column(c))
        self._update_sort_indicator()

    def _update_sort_indicator(self) -> None:
        for col in self.tree["columns"]:
            base_text = col.replace("_", " ").title()
            if col == "1_month_%":
                base_text = f"{CONFIG.custom_period_days} Day %"
            if col == self._sort_column:
                arrow = "Down" if self._sort_reverse else "Up"
                self.tree.heading(col, text=base_text + " " + arrow)
            else:
                self.tree.heading(col, text=base_text)

    def _sort_by_column(self, col: str) -> None:
        if self._sort_column == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = col
            self._sort_reverse = False

        self._update_sort_indicator()

        items = [(self.tree.set(item, col), item) for item in self.tree.get_children()]

        def parse_percent(v: str) -> float:
            if v in ("", "N/A"):
                return float('-inf') if self._sort_reverse else float('inf')
            return float(v.replace("%", "").strip())

        def parse_float(v: str) -> float:
            if v in ("", "N/A"):
                return float('-inf') if self._sort_reverse else float('inf')
            return float(v)

        if col in ("1_day_%", "1_month_%"):
            items.sort(key=lambda x: parse_percent(x[0]), reverse=self._sort_reverse)
        elif col == "price":
            items.sort(key=lambda x: parse_float(x[0]), reverse=self._sort_reverse)
        elif col in ("pe", "rsi", "macd", "target_diff"):
            items.sort(key=lambda x: parse_float(x[0]), reverse=self._sort_reverse)
        else:
            items.sort(
                key=lambda x: "" if x[0] in ("", "N/A") else x[0].lower(),
                reverse=self._sort_reverse
            )

        for idx, (_, item) in enumerate(items):
            self.tree.move(item, "", idx)

        for item in self.tree.get_children():
            ticker = self.tree.item(item)["values"][0]
            data = next((d for d in self.stock_data if d["ticker"] == ticker), None)
            if data:
                self.tree.item(item, tags=(data["recommendation"],))

    # ------------------------------------------------------------------- #
    # Ticker & List Management
    # ------------------------------------------------------------------- #
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

    def _update_list_display(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.current_tickers:
            self.status_lbl.config(text="Ready")
            return

        display_data = []
        for ticker in self.current_tickers:
            data = next((d for d in self.stock_data if d["ticker"] == ticker), None)
            if data:
                display_data.append({
                    "values": (
                        data["ticker"], data["name"], data["recommendation"],
                        f"{data['info'].get('regularMarketPrice', ''):.2f}",
                        data.get("price_swing_1d", "N/A"),
                        data.get("price_swing_1m", "N/A"),
                        data["sector"], data["industry"],
                        f"{data['metrics'].get('P/E', ''):.1f}" if data['metrics'].get('P/E') else "",
                        f"{data['metrics'].get('Target %', ''):+.1f}%" if data['metrics'].get('Target %') else "",
                        f"{data['metrics'].get('RSI', ''):.1f}" if data['metrics'].get('RSI') else "",
                        f"{data['metrics'].get('MACD', ''):+.3f}" if data['metrics'].get('MACD') else ""
                    ),
                    "tags": (data["recommendation"],),
                    "priority": {"Sell": 0, "Consider Selling": 1, "Buy": 2,
                                 "Consider Buying": 3, "Hold": 4}.get(data["recommendation"], 5)
                })
            else:
                display_data.append({
                    "values": (ticker, "", "", "", "", "", "", "", "", "", "", ""),
                    "tags": (), "priority": 6
                })

        display_data.sort(key=lambda x: x["priority"])
        for item in display_data:
            self.tree.insert("", tk.END, values=item["values"], tags=item["tags"])

        fetched = len(self.stock_data)
        total = len(self.current_tickers)
        if fetched == total:
            self.status_lbl.config(text=f"Fetched {fetched} stock(s).")
        else:
            self.status_lbl.config(text=f"{fetched}/{total} fetched. {total - fetched} pending.")

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
        self._update_list_display()
        self.unsaved_changes = True
        self.status_lbl.config(text=f"Added {norm}")
        if self.tree.get_children():
            first = self.tree.get_children()[0]
            self.tree.see(first)
            self.tree.selection_set(first)

    def remove_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Select", "Select ticker(s) to remove.")
            return
        to_remove = [self.tree.item(i)["values"][0] for i in sel]
        self.current_tickers = [t for t in self.current_tickers if t not in to_remove]
        self.stock_data = [d for d in self.stock_data if d["ticker"] not in to_remove]
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
                old_days = CONFIG.custom_period_days
                CONFIG.custom_period_days = days
                self.tree.heading("1_month_%", text=f"{days} Day %")
                self._update_sort_indicator()
                pop.destroy()
                messagebox.showinfo("Updated", f"Custom period set to {days} days.")
                if self.stock_data:
                    self.fetch_and_display()
                else:
                    self._update_list_display()
            except ValueError:
                messagebox.showerror("Invalid", "Please enter a positive integer.")

        tk.Button(pop, text="OK", command=save, bg=self.theme["button"], fg=self.theme["text"]).pack(pady=5)

    # ------------------------------------------------------------------- #
    # List Management
    # ------------------------------------------------------------------- #
    def new_list(self) -> None:
        if self.unsaved_changes and messagebox.askyesnocancel("Unsaved Changes", "Save current list before creating new?"):
            self.save_current_list()
        self.current_tickers.clear()
        self.stock_data.clear()
        self.current_list_name = ""
        self._update_list_display()
        self.update_list_name()
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
            self.update_list_name()
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
            self.update_list_name()
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
            self._update_list_display()
            self.update_list_name()
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

    def update_list_name(self) -> None:
        self.list_name_lbl.config(text=os.path.basename(self.current_list_name) if self.current_list_name else "(none)")

    # ------------------------------------------------------------------- #
    # Data Fetch & Display
    # ------------------------------------------------------------------- #
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
                if abs(swing_1d) >= CONFIG.price_swing_threshold:
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

            if rec == "Hold" and not any("swing" in r.lower() for r in reasons):
                reasons = []

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

        def fetch_all() -> None:
            with ThreadPoolExecutor(max_workers=min(len(self.current_tickers), CONFIG.max_threads)) as executor:
                futures = {executor.submit(self.fetch_stock_data, t): t for t in self.current_tickers}
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
            self.root.after(0, self._display_results, results)

        self.root.after(0, fetch_all)

    def _display_results(self, data: List[Dict[str, Any]]) -> None:
        self.stock_data = data
        self._update_list_display()
        for btn in self.button_refs.values():
            btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------- #
    # Pop-ups
    # ------------------------------------------------------------------- #
    def show_details_popup(self, event: Optional[tk.Event] = None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        ticker = self.tree.item(sel[0])["values"][0]
        data = next((d for d in self.stock_data if d["ticker"] == ticker), None)
        if not data:
            return

        pop = tk.Toplevel(self.root)
        pop.title(f"Details: {ticker}")
        pop.geometry("560x720")
        pop.configure(bg=self.theme["background"])

        txt = tk.Text(pop, wrap=tk.WORD, bg=self.theme["tree_bg"], fg=self.theme["text"], font=("Consolas", 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        lines = [
            f"Ticker: {data['ticker']}",
            f"Name: {data['name']}",
            f"Sector: {data['sector']}",
            f"Industry: {data['industry']}",
            f"Recommendation: {data['recommendation']}",
            "",
            "Reasons:" if data['reasons'] else "No active signals."
        ]
        lines.extend(f"  * {r}" for r in data['reasons'])
        lines.append("")
        lines.append("Market Data:")
        for k in ("regularMarketPrice", "previousClose", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "fiftyDayAverage", "trailingPE"):
            v = data["info"].get(k, "N/A")
            if isinstance(v, float):
                v = f"{v:.2f}"
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(f"1 Day %: {data['price_swing_1d']}")
        lines.append(f"{CONFIG.custom_period_days} Day %: {data['price_swing_1m']}")
        lines.append("")
        lines.append("Extra Metrics:")
        for k, v in data["metrics"].items():
            if v is None:
                v = "N/A"
            elif isinstance(v, float):
                v = f"{v:+.2f}%" if "Diff" in k or "Growth" in k else f"{v:.2f}"
            lines.append(f"  {k}: {v}")

        txt.insert(tk.END, "\n".join(lines))
        txt.config(state=tk.DISABLED)

    def show_info_popup(self) -> None:
        info_text = f"""
Metric Explanations

Price: Current market price
1 Day %: 24-hour price change %
{CONFIG.custom_period_days} Day %: Price change over custom period
P/E: Price-to-Earnings ratio (lower may indicate undervaluation)
Target %: Analyst target price vs current (% difference)
RSI: Relative Strength Index
  * < 30: Oversold (possible buy)
  * > 70: Overbought (possible sell)
MACD: Moving Average Convergence Divergence
  * Positive & rising: Bullish momentum
  * Negative & falling: Bearish momentum

Recommendation Logic:
* Buy: Price < 97% of 50-day MA
* Consider Buying: Price < 99% of 50-day MA
* Sell: Price > 98 lumine% of 52-week high
* Consider Selling: Price > 95% of 52-week high
* Hold: No strong signals

Large 1-day swings (>5%) also trigger alerts.
        """.strip()

        pop = tk.Toplevel(self.root)
        pop.title("Metric Explanations")
        pop.geometry("600x680")
        pop.configure(bg=self.theme["background"])

        txt = tk.Text(pop, wrap=tk.WORD, bg=self.theme["tree_bg"], fg=self.theme["text"], font=("Arial", 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        txt.insert(tk.END, info_text)
        txt.config(state=tk.DISABLED)

    # ------------------------------------------------------------------- #
    # Closing
    # ------------------------------------------------------------------- #
    def _on_closing(self) -> None:
        if self.unsaved_changes:
            resp = messagebox.askyesnocancel("Unsaved Changes", "Save current list before exiting?")
            if resp is True:
                self.save_current_list()
            elif resp is None:
                return
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = StockTrackerApp(root)
    root.mainloop()
