"""Stock Tracker Application using Tkinter and yFinance for real-time stock data analysis."""

import json
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Tuple, Optional
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf

# Configuration using dataclass
from dataclasses import dataclass

@dataclass
class AppConfig:
    min_window_width: int = 600
    min_window_height: int = 400
    preferred_file: str = "preferred_stocks.json"
    max_threads: int = 3
    recommendation_thresholds: dict = None
    enable_metrics: dict = None

    def __post_init__(self):
        self.recommendation_thresholds = {
            "buy_ma_ratio": 0.97,
            "consider_buy_ma_ratio": 0.99,
            "sell_high_ratio": 0.98,
            "consider_sell_high_ratio": 0.95,
            "pe_high": 30,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_buy": 0,
            "macd_sell": 0,
        }
        self.enable_metrics = {
            "pe_ratio": True,
            "revenue_growth": True,
            "analyst_target": True,
            "rsi": True,
            "macd": True,
        }

CONFIG = AppConfig()

# Theme configuration
THEMES = {
    "dark": {
        "background": "#2b2b2b",
        "text": "#ffffff",
        "entry": "#4a4a4a",
        "button": "#3a3a3a",
    }
}

# Extended ticker suffix map
TICKER_SUFFIX_MAP = {
    '.ST': '.ST',   # Stockholm exchange
    '.STO': '',     # TSX (e.g., LUG.STO -> LUG)
    '.MI': '.MI',   # Milan exchange
    '.DE': '.DE',   # German exchange
    '.L': '.L',     # London Stock Exchange
    '.PA': '.PA',   # Euronext Paris
    '.T': '.T',     # Tokyo Stock Exchange
    '.HK': '.HK',   # Hong Kong Stock Exchange
    '.SS': '.SS',   # Shanghai Stock Exchange
    '.SZ': '.SZ',   # Shenzhen Stock Exchange
    '.TO': '.TO',   # Toronto Stock Exchange
    '.AX': '.AX',   # Australian Securities Exchange
    '.NS': '.NS',   # National Stock Exchange of India
    '.BO': '.BO',   # Bombay Stock Exchange
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename="stock_tracker.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Utility functions
def handle_error(title: str, message: str, msg_type: str = "error") -> None:
    logging.error(f"{title}: {message}")
    getattr(messagebox, f"show{msg_type}")(title, message)

def load_preferred(preferred_file: str) -> list[str]:
    try:
        if os.path.exists(preferred_file):
            with open(preferred_file, 'r') as f:
                data = json.load(f)
                return [str(t).upper() for t in data] if isinstance(data, list) else []
    except Exception as e:
        handle_error("Load Error", f"Failed to load preferred tickers: {e}")
    return []

def save_preferred(preferred_file: str, tickers: list[str]) -> None:
    try:
        with open(preferred_file, 'w') as f:
            json.dump(tickers, f)
    except Exception as e:
        handle_error("Save Error", f"Failed to save preferred tickers: {e}")

def export_to_csv(stock_data: list[dict]) -> None:
    if not stock_data:
        handle_error("Export Error", "No stock data to export.")
        return
    try:
        flat_data = []
        for data in stock_data:
            flat_entry = {
                "ticker": data["ticker"],
                "name": data["name"],
                "sector": data["sector"],
                "industry": data["industry"],
                "recommendation": data["recommendation"],
                "reasons": ", ".join(data["reasons"]),
                **{k: v for k, v in data["info"].items() if k in [
                    "regularMarketPrice", "previousClose", "fiftyTwoWeekHigh",
                    "fiftyTwoWeekLow", "trailingPE", "fiftyDayAverage"
                ]},
                **data["metrics"]
            }
            flat_data.append(flat_entry)
        df = pd.DataFrame(flat_data)
        filename = f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        messagebox.showinfo("Success", f"Data exported to {filename}")
    except Exception as e:
        handle_error("Export Error", f"Failed to export to CSV: {e}")

# Data processing functions
def get_pe_ratio(info: dict) -> tuple[float | None, None]:
    return info.get("trailingPE"), None

def get_revenue_growth(info: dict) -> tuple[float | None, None]:
    try:
        revenue = info.get("totalRevenue")
        prev_revenue = info.get("revenuePrevious")
        if revenue and prev_revenue:
            return (revenue - prev_revenue) / prev_revenue * 100, None
    except Exception:
        return None, None
    return None, None

def get_analyst_target(info: dict, current_price: float) -> tuple[float | None, float | None, str | None]:
    target = info.get("targetMeanPrice")
    if target and current_price:
        diff = (target - current_price) / current_price * 100
        flag = "Potential Upside" if current_price < target else None
        return target, diff, flag
    return None, None, None

def get_rsi(history: pd.DataFrame, period: int = 14) -> tuple[float | None, str | None]:
    if history.empty or "Close" not in history:
        return None, None
    delta = history["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100, "Overbought"
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    flag = None
    if rsi > CONFIG.recommendation_thresholds["rsi_overbought"]:
        flag = "Overbought"
    elif rsi < CONFIG.recommendation_thresholds["rsi_oversold"]:
        flag = "Oversold"
    return rsi, flag

def get_macd(history: pd.DataFrame, short: int = 12, long: int = 26, signal: int = 9) -> tuple[float | None, float | None]:
    if history.empty or "Close" not in history:
        return None, None
    close = history["Close"]
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.iloc[-1], signal_line.iloc[-1]

class MetricRegistry:
    def __init__(self):
        self.metrics: dict[str, Callable] = {}
    def register(self, name: str, func: Callable):
        self.metrics[name] = func
    def compute(self, name: str, *args, **kwargs) -> tuple:
        return self.metrics.get(name, lambda *a, **k: (None, None))(*args, **kwargs)

registry = MetricRegistry()
registry.register("pe_ratio", get_pe_ratio)
registry.register("revenue_growth", get_revenue_growth)
registry.register("analyst_target", get_analyst_target)
registry.register("rsi", get_rsi)
registry.register("macd", get_macd)

# Main application class
class StockTrackerApp:
    def __init__(self, root: tk.Tk, theme: str = "dark") -> None:
        self.root = root
        self.root.title("Finance Tracker by Mattias")
        self.root.minsize(CONFIG.min_window_width, CONFIG.min_window_height)
        self.theme = THEMES[theme]
        self.root.configure(bg=self.theme["background"])
        self.button_refs: dict[str, tk.Button] = {}
        self.stock_data: list[dict] = []
        self.setup_ui()
        self.load_preferred_tickers(silent=True)

    def setup_ui(self) -> None:
        tk.Label(
            self.root,
            text="Enter Tickers (comma-separated):",
            bg=self.theme["background"],
            fg=self.theme["text"]
        ).pack(pady=5)
        self.ticker_entry = tk.Entry(
            self.root,
            width=50,
            bg=self.theme["entry"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"]
        )
        self.ticker_entry.pack(pady=5)
        self.status_label = tk.Label(
            self.root,
            text="",
            bg=self.theme["background"],
            fg=self.theme["text"]
        )
        self.status_label.pack(pady=5)
        self.progress_bar = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()
        self.text_frame = tk.Frame(self.root, bg=self.theme["background"])
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas = tk.Canvas(self.text_frame, bg=self.theme["background"], highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.theme["background"])
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        button_frame = tk.Frame(self.root, bg=self.theme["background"])
        button_frame.pack(pady=10)
        self.buttons = [
            ("Fetch Info", self.fetch_and_display),
            ("Save Preferred", self.save_current_as_preferred),
            ("Load Preferred", lambda: self.load_preferred_tickers(silent=False)),
            ("Export to CSV", lambda: export_to_csv(self.stock_data)),
            ("Exit", self.root.quit)
        ]
        for text, command in self.buttons:
            button = tk.Button(
                button_frame,
                text=text,
                command=command,
                bg=self.theme["button"],
                fg=self.theme["text"],
                width=15
            )
            button.pack(side=tk.LEFT, padx=5)
            self.button_refs[text] = button

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def normalize_ticker(self, ticker: str) -> str:
        ticker = ticker.upper()
        for suffix, normalized in TICKER_SUFFIX_MAP.items():
            if ticker.endswith(suffix):
                return ticker.replace(suffix, normalized)
        logging.warning(f"Unrecognized ticker suffix for {ticker}")
        return ticker

    def validate_tickers(self, tickers: list[str]) -> list[str]:
        valid_tickers = []
        invalid_tickers = []
        seen_tickers = set()
        duplicates = []
        for ticker in tickers:
            if not ticker:
                continue
            if ticker in seen_tickers:
                duplicates.append(ticker)
                continue
            seen_tickers.add(ticker)
            if len(ticker) > 10:
                invalid_tickers.append(f"{ticker}: Too long (max 10 characters)")
                continue
            if not all(c.isalnum() or c in ['.', '-'] for c in ticker):
                invalid_tickers.append(f"{ticker}: Contains invalid characters")
                continue
            valid_tickers.append(ticker)
        if duplicates:
            messagebox.showwarning("Duplicate Tickers", f"Duplicate tickers removed: {', '.join(duplicates)}")
            self.ticker_entry.delete(0, tk.END)
            self.ticker_entry.insert(0, ", ".join(valid_tickers))
        if invalid_tickers:
            messagebox.showwarning("Invalid Tickers", "The following tickers are invalid:\n" + "\n".join(invalid_tickers))
        return valid_tickers

    def load_preferred_tickers(self, silent: bool = False) -> None:
        tickers = load_preferred(CONFIG.preferred_file)
        if tickers:
            self.ticker_entry.delete(0, tk.END)
            self.ticker_entry.insert(0, ", ".join(tickers))
            if not silent:
                messagebox.showinfo("Loaded", "Preferred tickers loaded.")
        elif not silent:
            messagebox.showinfo("Info", "No preferred tickers found.")

    def save_current_as_preferred(self) -> None:
        tickers = [t.strip().upper() for t in self.ticker_entry.get().split(",") if t.strip()]
        if not tickers:
            handle_error("Save Error", "Please enter tickers to save.")
            return
        save_preferred(CONFIG.preferred_file, tickers)
        messagebox.showinfo("Saved", "Preferred tickers saved.")

    def fetch_stock_data(self, ticker: str) -> dict:
        try:
            ticker = self.normalize_ticker(ticker)
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="180d")
            name = info.get('shortName', 'N/A')
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            current_price = info.get('regularMarketPrice')
            recommendation = "Hold"
            reasons = []
            if current_price and "fiftyDayAverage" in info:
                if current_price < info["fiftyDayAverage"] * CONFIG.recommendation_thresholds["buy_ma_ratio"]:
                    recommendation = "Buy"
                    reasons.append("Price below 50-day MA")
                elif current_price < info["fiftyDayAverage"] * CONFIG.recommendation_thresholds["consider_buy_ma_ratio"]:
                    recommendation = "Consider Buying"
                    reasons.append("Price slightly below 50-day MA")
            if current_price and "fiftyTwoWeekHigh" in info:
                if current_price > info["fiftyTwoWeekHigh"] * CONFIG.recommendation_thresholds["sell_high_ratio"]:
                    recommendation = "Sell"
                    reasons.append("Price near 52-week high")
                elif (current_price > info["fiftyTwoWeekHigh"] * CONFIG.recommendation_thresholds["consider_sell_high_ratio"] and
                      recommendation == "Hold"):
                    recommendation = "Consider Selling"
                    reasons.append("Price approaching 52-week high")
            metrics = {}
            reasons_from_metrics = []
            if CONFIG.enable_metrics["pe_ratio"]:
                metrics["P/E Ratio"], _ = registry.compute("pe_ratio", info)
            if CONFIG.enable_metrics["revenue_growth"]:
                metrics["Revenue Growth YoY"], _ = registry.compute("revenue_growth", info)
            if CONFIG.enable_metrics["analyst_target"]:
                tgt, diff, flag = registry.compute("analyst_target", info, current_price)
                metrics["Analyst Target"] = tgt
                metrics["Target % Diff"] = diff
                if flag:
                    reasons_from_metrics.append(flag)
            if CONFIG.enable_metrics["rsi"]:
                rsi, flag = registry.compute("rsi", history)
                metrics["RSI (14)"] = rsi
                if flag:
                    reasons_from_metrics.append(flag)
            if CONFIG.enable_metrics["macd"]:
                macd, signal = registry.compute("macd", history)
                metrics["MACD"] = macd
                metrics["MACD Signal"] = signal
            reasons += reasons_from_metrics
            if recommendation == "Hold":
                reasons = []
            return {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": industry,
                "info": info,
                "recommendation": recommendation,
                "reasons": reasons,
                "metrics": metrics
            }
        except Exception as e:
            handle_error("Fetch Error", f"Error fetching data for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "name": "N/A",
                "sector": "N/A",
                "industry": "N/A",
                "info": {},
                "recommendation": "Hold",
                "reasons": [],
                "metrics": {}
            }

    def fetch_and_display(self) -> None:
        tickers = [t.strip().upper() for t in self.ticker_entry.get().split(",") if t.strip()]
        tickers = self.validate_tickers(tickers)
        if not tickers:
            handle_error("Validation Error", "No valid tickers entered.")
            return
        self.status_label.config(text="Fetching data...")
        self.progress_bar.pack()
        self.progress_bar.start()
        for button in self.button_refs.values():
            button.config(state=tk.DISABLED)
        def update_ui(stock_data: list[dict]) -> None:
            self.stock_data = stock_data
            priority = {"Sell": 0, "Consider Selling": 1, "Buy": 2, "Consider Buying": 3, "Hold": 4}
            stock_data.sort(key=lambda x: priority.get(x["recommendation"], 5))
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            for data in stock_data:
                block = tk.Frame(
                    self.scrollable_frame,
                    bg=self.theme["background"],
                    bd=1,
                    relief="solid",
                    padx=5,
                    pady=5
                )
                block.pack(fill=tk.X, pady=5)
                rec_color = (
                    "#ff0000" if data["recommendation"] == "Sell" else
                    "#FFFF00" if data["recommendation"] == "Consider Selling" else
                    "#00ff00" if data["recommendation"] == "Buy" else
                    "#87CEEB" if data["recommendation"] == "Consider Buying" else
                    self.theme["text"]
                )
                overview_text = f"{data['ticker']} - {data['name']}\n"
                overview_text += f"Recommendation: {data['recommendation']}\n"
                if data["reasons"]:
                    overview_text += f"Reasons: {', '.join(data['reasons'])}\n"
                overview_text += f"Sector: {data['sector']}, Industry: {data['industry']}"
                header = tk.Label(
                    block,
                    text=overview_text,
                    fg=rec_color,
                    bg=self.theme["background"],
                    justify="left",
                    anchor="w"
                )
                header.pack(fill=tk.X)
                details_frame = tk.Frame(block, bg=self.theme["background"])
                details_frame.pack(fill=tk.X, pady=5)
                details_frame.pack_forget()
                details_text = "Financial Indicators:\n"
                for key in [
                    "regularMarketPrice", "previousClose", "fiftyTwoWeekHigh",
                    "fiftyTwoWeekLow", "trailingPE", "fiftyDayAverage"
                ]:
                    details_text += f"{key}: {data['info'].get(key, 'N/A')}\n"
                details_text += "\nExtra Metrics:\n"
                for key, value in data["metrics"].items():
                    if value is None:
                        details_text += f"{key}: N/A\n"
                    elif "Diff" in key or "Growth" in key:
                        details_text += f"{key}: {value:.2f}%\n" if value is not None else f"{key}: N/A\n"
                    elif "RSI" in key or "MACD" in key:
                        details_text += f"{key}: {value:.2f}\n" if value is not None else f"{key}: N/A\n"
                    else:
                        details_text += f"{key}: {value}\n"
                details_label = tk.Label(
                    details_frame,
                    text=details_text,
                    fg=self.theme["text"],
                    bg=self.theme["background"],
                    justify="left",
                    anchor="w"
                )
                details_label.pack(fill=tk.X)
                def toggle_details(frame=details_frame) -> None:
                    if frame.winfo_ismapped():
                        frame.pack_forget()
                        toggle_btn.config(text="Show Details")
                    else:
                        frame.pack(fill=tk.X, pady=5)
                        toggle_btn.config(text="Hide Details")
                toggle_btn = tk.Button(
                    block,
                    text="Show Details",
                    command=toggle_details,
                    bg=self.theme["button"],
                    fg=self.theme["text"]
                )
                toggle_btn.pack()
            self.status_label.config(text="Data fetched successfully!")
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            for button in self.button_refs.values():
                button.config(state=tk.NORMAL)
        def fetch_all() -> None:
            max_workers = min(len(tickers), CONFIG.max_threads)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                stock_data = list(executor.map(self.fetch_stock_data, tickers))
            self.root.after(0, update_ui, stock_data)
        self.root.after(0, fetch_all)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockTrackerApp(root)
    root.mainloop()
