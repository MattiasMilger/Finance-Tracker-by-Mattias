import tkinter as tk
from tkinter import messagebox, ttk
import yfinance as yf
import json
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime
import logging
import numpy as np

# === CONFIGURATION CONSTANTS ===
CONFIG = {
    "MIN_WINDOW_WIDTH": 600,
    "MIN_WINDOW_HEIGHT": 400,
    "BACKGROUND_COLOR": "#2b2b2b",
    "TEXT_COLOR": "#ffffff",
    "ENTRY_COLOR": "#4a4a4a",
    "BUTTON_COLOR": "#3a3a3a",
    "PREFERRED_FILE": "preferred_stocks.json",
    "MAX_THREADS": 3,
    "RECOMMENDATION_THRESHOLDS": {
        "buy_ma_ratio": 0.97,
        "sell_high_ratio": 0.98,
        "pe_high": 30,
        "rsi_overbought": 70,
        "macd_buy": 0,
        "macd_sell": 0,
        "bollinger_buy": -2,
        "bollinger_sell": 2,
        "volume_spike": 2.0,
    }
}

# Ticker normalization for common exchanges
TICKER_SUFFIX_MAP = {
    '.ST': '.STO',
    '.MI': '.MI',
    '.DE': '.DE',
}

# Setup logging
logging.basicConfig(level=logging.INFO, filename="stock_tracker.log", 
                    format="%(asctime)s - %(levelname)s - %(message)s")

class StockTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Finance Tracker by Mattias")
        self.root.minsize(CONFIG["MIN_WINDOW_WIDTH"], CONFIG["MIN_WINDOW_HEIGHT"])
        self.root.configure(bg=CONFIG["BACKGROUND_COLOR"])
        self.button_refs = {}
        self.setup_ui()
        self.load_preferred_tickers(silent=True)

    def setup_ui(self):
        tk.Label(self.root, text="Enter Tickers (comma-separated):", 
                bg=CONFIG["BACKGROUND_COLOR"], fg=CONFIG["TEXT_COLOR"]).pack(pady=5)
        self.ticker_entry = tk.Entry(self.root, width=50, bg=CONFIG["ENTRY_COLOR"], 
                                   fg=CONFIG["TEXT_COLOR"], insertbackground=CONFIG["TEXT_COLOR"])
        self.ticker_entry.pack(pady=5)

        self.status_label = tk.Label(self.root, text="", bg=CONFIG["BACKGROUND_COLOR"], fg=CONFIG["TEXT_COLOR"])
        self.status_label.pack(pady=5)

        text_frame = tk.Frame(self.root, bg=CONFIG["BACKGROUND_COLOR"])
        text_frame.pack(pady=10)
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box = tk.Text(text_frame, height=15, width=70, bg=CONFIG["ENTRY_COLOR"], 
                               fg=CONFIG["TEXT_COLOR"], state=tk.DISABLED, yscrollcommand=scrollbar.set)
        self.text_box.pack(side=tk.LEFT)
        scrollbar.config(command=self.text_box.yview)
        self.text_box.tag_configure("green", foreground="#00ff00")
        self.text_box.tag_configure("red", foreground="#ff0000")
        self.text_box.tag_configure("blue", foreground="#00aaff")  # NEW: sector/industry blue

        button_frame = tk.Frame(self.root, bg=CONFIG["BACKGROUND_COLOR"])
        button_frame.pack(pady=10)
        self.buttons = [
            ("Fetch Info", self.fetch_and_display),
            ("Save Preferred", self.save_current_as_preferred),
            ("Load Preferred", lambda: self.load_preferred_tickers(silent=False)),
            ("Export to CSV", self.export_to_csv),
            ("Copy to Clipboard", self.copy_to_clipboard),
            ("Exit", self.root.quit)
        ]
        for text, command in self.buttons:
            button = tk.Button(button_frame, text=text, command=command, bg=CONFIG["BUTTON_COLOR"], 
                              fg=CONFIG["TEXT_COLOR"], width=15)
            button.pack(side=tk.LEFT, padx=5)
            self.button_refs[text] = button

    def normalize_ticker(self, ticker):
        for suffix, normalized in TICKER_SUFFIX_MAP.items():
            if ticker.endswith(suffix):
                ticker = ticker.replace(suffix, normalized)
        return ticker

    def load_preferred(self):
        try:
            if os.path.exists(CONFIG["PREFERRED_FILE"]):
                with open(CONFIG["PREFERRED_FILE"], 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError("Invalid preferred stocks file format")
                    return [str(t).upper() for t in data]
        except (json.JSONDecodeError, ValueError) as e:
            self.show_message("Error", f"Failed to load preferred tickers: {e}", "error")
        return []

    def save_preferred(self, tickers):
        try:
            with open(CONFIG["PREFERRED_FILE"], 'w') as f:
                json.dump(tickers, f)
        except Exception as e:
            self.show_message("Error", f"Failed to save preferred tickers: {e}", "error")

    def show_message(self, title, message, msg_type="info"):
        getattr(messagebox, f"show{msg_type}")(title, message)

    def copy_to_clipboard(self):
        try:
            content = self.text_box.get("1.0", tk.END).strip()
            if content:
                self.root.clipboard_clear()
                self.root.clipboard_append(content)
                self.show_message("Success", "Text copied to clipboard!")
            else:
                self.show_message("Info", "No text to copy.")
        except Exception as e:
            self.show_message("Error", f"Failed to copy to clipboard: {e}", "error")

    def export_to_csv(self):
        if not hasattr(self, 'stock_data') or not self.stock_data:
            self.show_message("Error", "No stock data to export.", "error")
            return
        try:
            df = pd.DataFrame(self.stock_data)
            filename = f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            self.show_message("Success", f"Data exported to {filename}")
        except Exception as e:
            self.show_message("Error", f"Failed to export to CSV: {e}", "error")

    def fetch_stock_data(self, ticker):
        try:
            ticker = self.normalize_ticker(ticker)
            logging.info(f"Fetching data for ticker: {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="60d")

            name = info.get('shortName', 'N/A')
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            current_price = info.get('regularMarketPrice', 
                                   info.get('currentPrice', 
                                           history['Close'].iloc[-1] if not history.empty else None))
            previous_close = info.get('previousClose', None)
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', None)
            fifty_two_week_low = info.get('fiftyTwoWeekLow', None)
            pe_ratio = info.get('trailingPE', None)
            fifty_day_avg = info.get('fiftyDayAverage', None)

            # RSI
            rsi = None
            if not history.empty and len(history) >= 14:
                delta = history['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                if not loss.empty and loss.mean() != 0:
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs.mean()))

            # MACD
            macd = None
            macd_signal = None
            if not history.empty and len(history) >= 26:
                ema12 = history['Close'].ewm(span=12, adjust=False).mean()
                ema26 = history['Close'].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                macd = macd.iloc[-1]
                macd_signal = macd_signal.iloc[-1]

            # Bollinger Bands
            bollinger_z = None
            if not history.empty and len(history) >= 20:
                sma20 = history['Close'].rolling(window=20).mean()
                std20 = history['Close'].rolling(window=20).std()
                if current_price and not pd.isna(sma20.iloc[-1]) and not pd.isna(std20.iloc[-1]):
                    bollinger_z = (current_price - sma20.iloc[-1]) / std20.iloc[-1]

            # Volume
            avg_volume = None
            volume_ratio = None
            if not history.empty and len(history) >= 20:
                avg_volume = history['Volume'].rolling(window=20).mean().iloc[-1]
                recent_volume = history['Volume'].iloc[-1]
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume

            # Recommendation
            recommendation = "Hold"
            reasons = []
            if current_price and fifty_day_avg:
                if current_price < fifty_day_avg * CONFIG["RECOMMENDATION_THRESHOLDS"]["buy_ma_ratio"]:
                    recommendation = "Consider Buying"
                    reasons.append("Price below 50-day MA")
                elif current_price > fifty_two_week_high * CONFIG["RECOMMENDATION_THRESHOLDS"]["sell_high_ratio"]:
                    recommendation = "Consider Selling"
                    reasons.append("Price near 52-week high")
            if pe_ratio and pe_ratio > CONFIG["RECOMMENDATION_THRESHOLDS"]["pe_high"]:
                recommendation = "Consider Selling"
                reasons.append("High P/E ratio")
            if rsi and rsi > CONFIG["RECOMMENDATION_THRESHOLDS"]["rsi_overbought"]:
                recommendation = "Consider Selling"
                reasons.append("Overbought (RSI)")
            if macd is not None and macd_signal is not None:
                if macd > macd_signal and macd > CONFIG["RECOMMENDATION_THRESHOLDS"]["macd_buy"]:
                    recommendation = "Consider Buying"
                    reasons.append("Bullish MACD crossover")
                elif macd < macd_signal and macd < CONFIG["RECOMMENDATION_THRESHOLDS"]["macd_sell"]:
                    recommendation = "Consider Selling"
                    reasons.append("Bearish MACD crossover")
            if bollinger_z is not None:
                if bollinger_z < CONFIG["RECOMMENDATION_THRESHOLDS"]["bollinger_buy"]:
                    recommendation = "Consider Buying"
                    reasons.append("Price below lower Bollinger Band")
                elif bollinger_z > CONFIG["RECOMMENDATION_THRESHOLDS"]["bollinger_sell"]:
                    recommendation = "Consider Selling"
                    reasons.append("Price above upper Bollinger Band")
            if volume_ratio is not None and volume_ratio > CONFIG["RECOMMENDATION_THRESHOLDS"]["volume_spike"]:
                reasons.append(f"High volume ({volume_ratio:.2f}x average)")

            return {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": industry,
                "current_price": round(current_price, 2) if current_price else None,
                "previous_close": round(previous_close, 2) if previous_close else None,
                "fifty_two_week_high": round(fifty_two_week_high, 2) if fifty_two_week_high else None,
                "fifty_two_week_low": round(fifty_two_week_low, 2) if fifty_two_week_low else None,
                "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                "rsi": round(rsi, 2) if rsi else None,
                "macd": round(macd, 2) if macd is not None else None,
                "macd_signal": round(macd_signal, 2) if macd_signal is not None else None,
                "bollinger_z": round(bollinger_z, 2) if bollinger_z is not None else None,
                "avg_volume": round(avg_volume, 0) if avg_volume else None,
                "volume_ratio": round(volume_ratio, 2) if volume_ratio else None,
                "recommendation": recommendation,
                "reasons": reasons or ["No specific reason"]
            }
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "name": "N/A",
                "sector": "N/A",
                "industry": "N/A",
                "current_price": None,
                "previous_close": None,
                "fifty_two_week_high": None,
                "fifty_two_week_low": None,
                "pe_ratio": None,
                "rsi": None,
                "macd": None,
                "macd_signal": None,
                "bollinger_z": None,
                "avg_volume": None,
                "volume_ratio": None,
                "recommendation": f"Failed to fetch data ({str(e)})",
                "reasons": []
            }

    def fetch_and_display(self):
        tickers_raw = self.ticker_entry.get().strip()
        if not tickers_raw:
            self.show_message("Error", "Please enter at least one ticker.", "error")
            return
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        self.status_label.config(text="Fetching data...")
        for button in self.button_refs.values():
            button.config(state=tk.DISABLED)

        def update_ui(stock_data):
            self.stock_data = stock_data
            priority = {"Consider Selling": 0, "Consider Buying": 1, "Hold": 2}
            stock_data.sort(key=lambda x: priority.get(x["recommendation"], 3))
            output = []
            for data in stock_data:
                if "Failed" in data["recommendation"]:
                    output.append((f"Ticker: {data['ticker']} - {data['recommendation']}\n\n", ""))
                else:
                    block = (
                        f"Ticker: {data['ticker']}\n"
                        f"Name: {data['name']}\n"
                        f"Current Price: {data['current_price']}\n"
                        f"Previous Close: {data['previous_close']}\n"
                        f"52-Week High: {data['fifty_two_week_high']}\n"
                        f"52-Week Low: {data['fifty_two_week_low']}\n"
                        f"P/E Ratio: {data['pe_ratio']}\n"
                        f"RSI (14-day): {data['rsi']}\n"
                        f"MACD: {data['macd']}\n"
                        f"MACD Signal: {data['macd_signal']}\n"
                        f"Bollinger Z-Score: {data['bollinger_z']}\n"
                        f"Avg Volume (20-day): {data['avg_volume']}\n"
                        f"Volume Ratio: {data['volume_ratio']}\n"
                    )
                    output.append((block, ""))
                    recommendation = (
                        f"Recommendation: {data['recommendation']} ({'; '.join(data['reasons'])})\n"
                    )
                    tag = "green" if data["recommendation"] == "Consider Buying" else "red" if data["recommendation"] == "Consider Selling" else ""
                    output.append((recommendation, tag))

                    # NEW: Sector & Industry always blue
                    sector_industry = f"Sector = {data['sector']}, Industry = {data['industry']}.\n\n"
                    output.append((sector_industry, "blue"))

            self.text_box.config(state=tk.NORMAL)
            self.text_box.delete(1.0, tk.END)
            for line, tag in output:
                self.text_box.insert(tk.END, line, tag)
            self.text_box.config(state=tk.DISABLED)
            self.status_label.config(text="Data fetched successfully!")
            for button in self.button_refs.values():
                button.config(state=tk.NORMAL)

        def fetch_all():
            with ThreadPoolExecutor(max_workers=CONFIG["MAX_THREADS"]) as executor:
                stock_data = list(executor.map(self.fetch_stock_data, tickers))
            self.root.after(0, update_ui, stock_data)

        self.root.after(0, fetch_all)

    def save_current_as_preferred(self):
        tickers_raw = self.ticker_entry.get().strip()
        if not tickers_raw:
            self.show_message("Error", "Please enter tickers to save.", "error")
            return
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        self.save_preferred(tickers)
        self.show_message("Saved", "Preferred tickers saved.")

    def load_preferred_tickers(self, silent=False):
        tickers = self.load_preferred()
        if tickers:
            self.ticker_entry.delete(0, tk.END)
            self.ticker_entry.insert(0, ", ".join(tickers))
            if not silent:
                self.show_message("Loaded", "Preferred tickers loaded.")
        else:
            if not silent:
                self.show_message("Info", "No preferred tickers found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockTrackerApp(root)
    root.mainloop()
