import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import json
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime
import logging

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

# Ticker normalization
TICKER_SUFFIX_MAP = {
    '.ST': '.STO',
    '.MI': '.MI',
    '.DE': '.DE',
}

# Logging
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

        # Scrollable output frame
        self.text_frame = tk.Frame(self.root, bg=CONFIG["BACKGROUND_COLOR"])
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.text_frame, bg=CONFIG["BACKGROUND_COLOR"], highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=CONFIG["BACKGROUND_COLOR"])

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        button_frame = tk.Frame(self.root, bg=CONFIG["BACKGROUND_COLOR"])
        button_frame.pack(pady=10)
        self.buttons = [
            ("Fetch Info", self.fetch_and_display),
            ("Save Preferred", self.save_current_as_preferred),
            ("Load Preferred", lambda: self.load_preferred_tickers(silent=False)),
            ("Export to CSV", self.export_to_csv),
            ("Exit", self.root.quit)
        ]
        for text, command in self.buttons:
            button = tk.Button(button_frame, text=text, command=command,
                               bg=CONFIG["BUTTON_COLOR"], fg=CONFIG["TEXT_COLOR"], width=15)
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
                    return [str(t).upper() for t in data] if isinstance(data, list) else []
        except Exception as e:
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
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="60d")

            name = info.get('shortName', 'N/A')
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            current_price = info.get('regularMarketPrice', None)

            # Recommendation
            recommendation = "Hold"
            reasons = []
            if current_price and "fiftyDayAverage" in info and current_price < info["fiftyDayAverage"] * CONFIG["RECOMMENDATION_THRESHOLDS"]["buy_ma_ratio"]:
                recommendation = "Buy"
                reasons.append("Price below 50-day MA")
            elif current_price and "fiftyTwoWeekHigh" in info and current_price > info["fiftyTwoWeekHigh"] * CONFIG["RECOMMENDATION_THRESHOLDS"]["sell_high_ratio"]:
                recommendation = "Sell"
                reasons.append("Price near 52-week high")

            return {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": industry,
                "info": info,
                "recommendation": recommendation,
                "reasons": reasons or ["No specific reason"]
            }
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {str(e)}")
            return {"ticker": ticker, "name": "N/A", "sector": "N/A", "industry": "N/A",
                    "info": {}, "recommendation": "Hold", "reasons": [f"Error: {e}"]}

    def fetch_and_display(self):
        tickers = [t.strip().upper() for t in self.ticker_entry.get().split(",") if t.strip()]
        if not tickers:
            self.show_message("Error", "Please enter at least one ticker.", "error")
            return

        self.status_label.config(text="Fetching data...")
        for button in self.button_refs.values():
            button.config(state=tk.DISABLED)

        def update_ui(stock_data):
            self.stock_data = stock_data
            priority = {"Sell": 0, "Buy": 1, "Hold": 2}
            stock_data.sort(key=lambda x: priority.get(x["recommendation"], 3))

            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()

            for data in stock_data:
                block = tk.Frame(self.scrollable_frame, bg=CONFIG["BACKGROUND_COLOR"], bd=1, relief="solid", padx=5, pady=5)
                block.pack(fill=tk.X, pady=5)

                rec_color = "#ff0000" if data["recommendation"] == "Sell" else "#00ff00" if data["recommendation"] == "Buy" else CONFIG["TEXT_COLOR"]

                header = tk.Label(block, text=f"{data['ticker']} - {data['name']}\n"
                                              f"Recommendation: {data['recommendation']}\n"
                                              f"Sector: {data['sector']}, Industry: {data['industry']}",
                                  fg=rec_color, bg=CONFIG["BACKGROUND_COLOR"], justify="left", anchor="w")
                header.pack(fill=tk.X)

                details_frame = tk.Frame(block, bg=CONFIG["BACKGROUND_COLOR"])
                details_frame.pack(fill=tk.X, pady=5)
                details_frame.pack_forget()

                details_text = "Financial Indicators:\n"
                for key in ["regularMarketPrice", "previousClose", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                            "trailingPE", "fiftyDayAverage"]:
                    details_text += f"{key}: {data['info'].get(key, 'N/A')}\n"
                details_text += f"\nReasons: {', '.join(data['reasons'])}\n"

                details_label = tk.Label(details_frame, text=details_text, fg=CONFIG["TEXT_COLOR"],
                                         bg=CONFIG["BACKGROUND_COLOR"], justify="left", anchor="w")
                details_label.pack(fill=tk.X)

                def toggle_details(frame=details_frame, btn_text="Show Details"):
                    if frame.winfo_ismapped():
                        frame.pack_forget()
                        toggle_btn.config(text="Show Details")
                    else:
                        frame.pack(fill=tk.X, pady=5)
                        toggle_btn.config(text="Hide Details")

                toggle_btn = tk.Button(block, text="Show Details", command=toggle_details,
                                       bg=CONFIG["BUTTON_COLOR"], fg=CONFIG["TEXT_COLOR"])
                toggle_btn.pack()

            self.status_label.config(text="Data fetched successfully!")
            for button in self.button_refs.values():
                button.config(state=tk.NORMAL)

        def fetch_all():
            with ThreadPoolExecutor(max_workers=CONFIG["MAX_THREADS"]) as executor:
                stock_data = list(executor.map(self.fetch_stock_data, tickers))
            self.root.after(0, update_ui, stock_data)

        self.root.after(0, fetch_all)

    def save_current_as_preferred(self):
        tickers = [t.strip().upper() for t in self.ticker_entry.get().split(",") if t.strip()]
        if not tickers:
            self.show_message("Error", "Please enter tickers to save.", "error")
            return
        self.save_preferred(tickers)
        self.show_message("Saved", "Preferred tickers saved.")

    def load_preferred_tickers(self, silent=False):
        tickers = self.load_preferred()
        if tickers:
            self.ticker_entry.delete(0, tk.END)
            self.ticker_entry.insert(0, ", ".join(tickers))
            if not silent:
                self.show_message("Loaded", "Preferred tickers loaded.")
        elif not silent:
            self.show_message("Info", "No preferred tickers found.")


if __name__ == "__main__":
    root = tk.Tk()
    app = StockTrackerApp(root)
    root.mainloop()
