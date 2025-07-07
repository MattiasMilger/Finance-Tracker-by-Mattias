import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import json
import os

# === CONFIGURATION CONSTANTS ===
MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT = 600, 400
BACKGROUND_COLOR = "#2b2b2b"
TEXT_COLOR = "#ffffff"
ENTRY_COLOR = "#4a4a4a"
BUTTON_COLOR = "#3a3a3a"
PREFERRED_FILE = "preferred_stocks.json"

# === HELPER FUNCTIONS ===
def load_preferred():
    if os.path.exists(PREFERRED_FILE):
        with open(PREFERRED_FILE, 'r') as f:
            return json.load(f)
    return []

def save_preferred(tickers):
    with open(PREFERRED_FILE, 'w') as f:
        json.dump(tickers, f)

def show_message(title, message, msg_type="info"):
    getattr(messagebox, f"show{msg_type}")(title, message)

# === MAIN APP LOGIC ===
def run_app():
    root = tk.Tk()
    root.title("Finance Tracker by Mattias")
    root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
    root.configure(bg=BACKGROUND_COLOR)

    tk.Label(root, text="Enter Tickers (comma-separated):", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).pack(pady=5)
    ticker_entry = tk.Entry(root, width=50, bg=ENTRY_COLOR, fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
    ticker_entry.pack(pady=5)

    text_box = tk.Text(root, height=15, width=70, bg=ENTRY_COLOR, fg=TEXT_COLOR, state=tk.DISABLED)
    text_box.pack(pady=10)

    def update_text_box(content):
        text_box.config(state=tk.NORMAL)
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, content)
        text_box.config(state=tk.DISABLED)

    def fetch_and_display():
        tickers_raw = ticker_entry.get().strip()
        if not tickers_raw:
            show_message("Error", "Please enter at least one ticker.", "error")
            return
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        output = ""
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                name = info.get('shortName', 'N/A')
                current_price = info.get('regularMarketPrice')
                previous_close = info.get('previousClose')
                fifty_two_week_high = info.get('fiftyTwoWeekHigh')
                fifty_two_week_low = info.get('fiftyTwoWeekLow')
                recommendation = "Hold"
                if current_price and info.get('fiftyDayAverage'):
                    if current_price < info['fiftyDayAverage'] * 0.97:
                        recommendation = "Consider Buying"
                    elif current_price > fifty_two_week_high * 0.98:
                        recommendation = "Consider Holding/Selling"
                output += (
                    f"Ticker: {ticker}\n"
                    f"Name: {name}\n"
                    f"Current Price: {current_price}\n"
                    f"Previous Close: {previous_close}\n"
                    f"52-Week High: {fifty_two_week_high}\n"
                    f"52-Week Low: {fifty_two_week_low}\n"
                    f"Recommendation: {recommendation}\n\n"
                )
            except Exception as e:
                output += f"Ticker: {ticker} - Failed to fetch data ({e})\n\n"

        update_text_box(output)

    def save_current_as_preferred():
        tickers_raw = ticker_entry.get().strip()
        if not tickers_raw:
            show_message("Error", "Please enter tickers to save.", "error")
            return
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        save_preferred(tickers)
        show_message("Saved", "Preferred tickers saved.")

    def load_preferred_tickers():
        tickers = load_preferred()
        if tickers:
            ticker_entry.delete(0, tk.END)
            ticker_entry.insert(0, ", ".join(tickers))
            show_message("Loaded", "Preferred tickers loaded.")
        else:
            show_message("Info", "No preferred tickers found.")

    button_frame = tk.Frame(root, bg=BACKGROUND_COLOR)
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="Fetch Info", command=fetch_and_display, bg=BUTTON_COLOR, fg=TEXT_COLOR, width=15).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Save Preferred", command=save_current_as_preferred, bg=BUTTON_COLOR, fg=TEXT_COLOR, width=15).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Load Preferred", command=load_preferred_tickers, bg=BUTTON_COLOR, fg=TEXT_COLOR, width=15).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Exit", command=root.quit, bg=BUTTON_COLOR, fg=TEXT_COLOR, width=10).pack(side=tk.LEFT, padx=5)

    load_preferred_tickers()  # Auto-load on startup
    root.mainloop()

# === RUN ===
if __name__ == "__main__":
    run_app()
