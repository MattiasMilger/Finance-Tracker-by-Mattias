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

def copy_to_clipboard(text_widget):
    try:
        content = text_widget.get("1.0", tk.END).strip()
        if content:
            root.clipboard_clear()
            root.clipboard_append(content)
            show_message("Success", "Text copied to clipboard!")
        else:
            show_message("Info", "No text to copy.")
    except Exception as e:
        show_message("Error", f"Failed to copy to clipboard: {e}", "error")

# === MAIN APP LOGIC ===
def run_app():
    global root
    root = tk.Tk()
    root.title("Finance Tracker by Mattias")
    root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
    root.configure(bg=BACKGROUND_COLOR)

    tk.Label(root, text="Enter Tickers (comma-separated):", bg=BACKGROUND_COLOR, fg=TEXT_COLOR).pack(pady=5)
    ticker_entry = tk.Entry(root, width=50, bg=ENTRY_COLOR, fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
    ticker_entry.pack(pady=5)

    # Create a frame for the text box and scrollbar
    text_frame = tk.Frame(root, bg=BACKGROUND_COLOR)
    text_frame.pack(pady=10)

    # Add scrollbar
    scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configure text box with scrollbar
    text_box = tk.Text(text_frame, height=15, width=70, bg=ENTRY_COLOR, fg=TEXT_COLOR, 
                      state=tk.DISABLED, yscrollcommand=scrollbar.set)
    text_box.pack(side=tk.LEFT)
    scrollbar.config(command=text_box.yview)

    # Configure tags for colored text
    text_box.tag_configure("green", foreground="#00ff00")  # Green for Consider Buying
    text_box.tag_configure("red", foreground="#ff0000")   # Red for Consider Selling

    def update_text_box(content):
        text_box.config(state=tk.NORMAL)
        text_box.delete(1.0, tk.END)
        for line, tag in content:
            text_box.insert(tk.END, line, tag)
        text_box.config(state=tk.DISABLED)

    def fetch_and_display():
        tickers_raw = ticker_entry.get().strip()
        if not tickers_raw:
            show_message("Error", "Please enter at least one ticker.", "error")
            return
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        stock_data = []  # List to store stock info dictionaries

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
                        recommendation = "Consider Selling"
                # Store data in a dictionary
                stock_data.append({
                    "ticker": ticker,
                    "name": name,
                    "current_price": current_price,
                    "previous_close": previous_close,
                    "fifty_two_week_high": fifty_two_week_high,
                    "fifty_two_week_low": fifty_two_week_low,
                    "recommendation": recommendation
                })
            except Exception as e:
                stock_data.append({
                    "ticker": ticker,
                    "name": "N/A",
                    "current_price": None,
                    "previous_close": None,
                    "fifty_two_week_high": None,
                    "fifty_two_week_low": None,
                    "recommendation": f"Failed to fetch data ({e})"
                })

        # Define recommendation priority
        priority = {"Consider Selling": 0, "Consider Buying": 1, "Hold": 2}
        # Sort by recommendation, using priority.get() to handle errors
        stock_data.sort(key=lambda x: priority.get(x["recommendation"], 3))

        # Format output with tags for recommendations
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
                )
                output.append((block, ""))  # Regular text, no tag
                # Add recommendation with color tag
                recommendation = f"Recommendation: {data['recommendation']}\n\n"
                tag = "green" if data["recommendation"] == "Consider Buying" else "red" if data["recommendation"] == "Consider Selling" else ""
                output.append((recommendation, tag))

        update_text_box(output)  # Pass the output list directly

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
    tk.Button(button_frame, text="Copy to Clipboard", command=lambda: copy_to_clipboard(text_box), bg=BUTTON_COLOR, fg=TEXT_COLOR, width=15).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Exit", command=root.quit, bg=BUTTON_COLOR, fg=TEXT_COLOR, width=10).pack(side=tk.LEFT, padx=5)

    load_preferred_tickers()  # Auto-load on startup
    root.mainloop()

# === RUN ===
if __name__ == "__main__":
    run_app()
