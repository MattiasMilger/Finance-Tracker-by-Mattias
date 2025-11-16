"""Historical Price Charts Module for Stock Tracker Application.

Provides interactive charts with multiple timeframes, technical indicators,
and volume analysis using matplotlib.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.dates as mdates


class HistoricalChartWindow:
    """Window for displaying interactive historical price charts."""
    
    def __init__(self, parent: tk.Tk, theme: Dict[str, str], ticker: str, 
                 stock_name: str = "", initial_period: str = "1y"):
        """Initialize the chart window.
        
        Args:
            parent: Parent Tkinter window
            theme: Color theme dictionary
            ticker: Stock ticker symbol
            stock_name: Company name for display
            initial_period: Initial time period to display (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        """
        self.parent = parent
        self.theme = theme
        self.ticker = ticker
        self.stock_name = stock_name or ticker
        self.current_period = initial_period
        
        # Chart display options
        self.show_ma50 = tk.BooleanVar(value=True)
        self.show_ma200 = tk.BooleanVar(value=True)
        self.show_volume = tk.BooleanVar(value=True)
        self.show_bollinger = tk.BooleanVar(value=False)
        self.chart_type = tk.StringVar(value="line")  # line, candlestick, area
        
        # Data storage
        self.hist_data: Optional[pd.DataFrame] = None
        
        # Create window with larger size
        self.window = tk.Toplevel(parent)
        self.window.title(f"Historical Chart: {ticker} - {self.stock_name}")
        self.window.geometry("1500x950")  # Increased from 1400x900
        self.window.configure(bg=theme["background"])
        
        # Setup UI
        self._setup_ui()
        
        # Load initial data
        self.load_data()
    
    def _setup_ui(self):
        """Create and layout all UI components."""
        # Top control panel
        control_frame = tk.Frame(self.window, bg=self.theme["background"])
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Left side - period selection
        period_frame = tk.LabelFrame(
            control_frame, 
            text="Time Period", 
            bg=self.theme["background"],
            fg=self.theme["text"],
            padx=10,
            pady=5
        )
        period_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        periods = [
            ("1 Month", "1mo"),
            ("3 Months", "3mo"),
            ("6 Months", "6mo"),
            ("1 Year", "1y"),
            ("2 Years", "2y"),
            ("5 Years", "5y"),
            ("Max", "max")
        ]
        
        for i, (label, period) in enumerate(periods):
            btn = tk.Button(
                period_frame,
                text=label,
                command=lambda p=period: self.change_period(p),
                bg=self.theme["button"],
                fg=self.theme["text"],
                width=10,
                relief=tk.RAISED if period != self.current_period else tk.SUNKEN
            )
            btn.grid(row=0, column=i, padx=2)
            
        # Middle - chart type
        chart_frame = tk.LabelFrame(
            control_frame,
            text="Chart Type",
            bg=self.theme["background"],
            fg=self.theme["text"],
            padx=10,
            pady=5
        )
        chart_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        chart_types = [
            ("Line", "line"),
            ("Candlestick", "candlestick"),
            ("Area", "area")
        ]
        
        for i, (label, ctype) in enumerate(chart_types):
            rb = tk.Radiobutton(
                chart_frame,
                text=label,
                variable=self.chart_type,
                value=ctype,
                command=self.update_chart,
                bg=self.theme["background"],
                fg=self.theme["text"],
                selectcolor=self.theme["button"],
                activebackground=self.theme["background"],
                activeforeground=self.theme["text"]
            )
            rb.grid(row=0, column=i, padx=5)
        
        # Right side - indicators
        indicator_frame = tk.LabelFrame(
            control_frame,
            text="Indicators",
            bg=self.theme["background"],
            fg=self.theme["text"],
            padx=10,
            pady=5
        )
        indicator_frame.pack(side=tk.LEFT)
        
        indicators = [
            ("50-Day MA", self.show_ma50),
            ("200-Day MA", self.show_ma200),
            ("Volume", self.show_volume),
            ("Bollinger Bands", self.show_bollinger)
        ]
        
        for i, (label, var) in enumerate(indicators):
            cb = tk.Checkbutton(
                indicator_frame,
                text=label,
                variable=var,
                command=self.update_chart,
                bg=self.theme["background"],
                fg=self.theme["text"],
                selectcolor=self.theme["button"],
                activebackground=self.theme["background"],
                activeforeground=self.theme["text"]
            )
            cb.grid(row=0, column=i, padx=5)
        
        # Refresh button - removed emoji
        tk.Button(
            control_frame,
            text="Refresh",
            command=self.load_data,
            bg=self.theme["button"],
            fg=self.theme["text"],
            font=("Arial", 10, "bold"),
            width=10
        ).pack(side=tk.RIGHT, padx=10)
        
        # Chart container with matplotlib
        chart_container = tk.Frame(self.window, bg=self.theme["background"])
        chart_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create matplotlib figure with larger size
        self.fig = Figure(figsize=(15, 9), facecolor=self.theme["background"])  # Increased from 14x8
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = tk.Frame(chart_container, bg=self.theme["background"])
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Status bar
        self.status_label = tk.Label(
            self.window,
            text="Ready",
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 9),
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 5))
    
    def change_period(self, period: str):
        """Change the time period and reload data.
        
        Args:
            period: New period string (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        """
        self.current_period = period
        self.load_data()
        
        # Update button states
        for widget in self.window.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.LabelFrame) and child.cget("text") == "Time Period":
                        for btn in child.winfo_children():
                            if isinstance(btn, tk.Button):
                                btn.config(relief=tk.RAISED)
                                if period in btn.cget("text").lower().replace(" ", ""):
                                    btn.config(relief=tk.SUNKEN)
    
    def load_data(self):
        """Fetch historical data from yfinance and update chart."""
        self.status_label.config(text=f"Loading data for {self.ticker}...")
        self.window.update_idletasks()
        
        try:
            stock = yf.Ticker(self.ticker)
            
            # Determine interval based on period
            interval = "1d"
            if self.current_period in ["1mo", "3mo"]:
                interval = "1d"
            elif self.current_period in ["6mo", "1y"]:
                interval = "1d"
            else:
                interval = "1wk"  # Use weekly for longer periods
            
            # Fetch data
            self.hist_data = stock.history(period=self.current_period, interval=interval)
            
            if self.hist_data.empty:
                messagebox.showerror("Error", f"No data available for {self.ticker}")
                self.status_label.config(text="Error: No data available")
                return
            
            # Calculate indicators
            self._calculate_indicators()
            
            # Update chart
            self.update_chart()
            
            # Update status with date range
            start_date = self.hist_data.index[0].strftime("%Y-%m-%d")
            end_date = self.hist_data.index[-1].strftime("%Y-%m-%d")
            
            # Calculate the actual time span for display
            date_diff = self.hist_data.index[-1] - self.hist_data.index[0]
            years = date_diff.days / 365.25
            
            if self.current_period == "max":
                if years >= 1:
                    period_display = f"Max ({years:.1f} years)"
                else:
                    period_display = f"Max ({date_diff.days} days)"
            else:
                period_display = self.current_period.upper()
            
            self.status_label.config(
                text=f"Showing data from {start_date} to {end_date} ({len(self.hist_data)} data points) - Period: {period_display}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")
    
    def _calculate_indicators(self):
        """Calculate technical indicators on historical data."""
        if self.hist_data is None or self.hist_data.empty:
            return
        
        # Moving averages
        self.hist_data["MA50"] = self.hist_data["Close"].rolling(window=50).mean()
        self.hist_data["MA200"] = self.hist_data["Close"].rolling(window=200).mean()
        
        # Bollinger Bands (20-day, 2 std dev)
        self.hist_data["BB_Middle"] = self.hist_data["Close"].rolling(window=20).mean()
        bb_std = self.hist_data["Close"].rolling(window=20).std()
        self.hist_data["BB_Upper"] = self.hist_data["BB_Middle"] + (bb_std * 2)
        self.hist_data["BB_Lower"] = self.hist_data["BB_Middle"] - (bb_std * 2)
    
    def update_chart(self):
        """Redraw the chart with current settings."""
        if self.hist_data is None or self.hist_data.empty:
            return
        
        # Clear previous chart
        self.fig.clear()
        
        # Determine layout
        if self.show_volume.get():
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax_price = self.fig.add_subplot(gs[0])
            ax_volume = self.fig.add_subplot(gs[1], sharex=ax_price)
        else:
            ax_price = self.fig.add_subplot(111)
            ax_volume = None
        
        # Configure colors for dark theme
        text_color = "#e0e0e0"
        grid_color = "#3a3a3a"
        
        # Draw price chart
        if self.chart_type.get() == "line":
            self._draw_line_chart(ax_price)
        elif self.chart_type.get() == "candlestick":
            self._draw_candlestick_chart(ax_price)
        elif self.chart_type.get() == "area":
            self._draw_area_chart(ax_price)
        
        # Add moving averages
        if self.show_ma50.get() and "MA50" in self.hist_data.columns:
            ax_price.plot(
                self.hist_data.index,
                self.hist_data["MA50"],
                label="50-Day MA",
                color="#FFB74D",
                linewidth=1.5,
                alpha=0.8
            )
        
        if self.show_ma200.get() and "MA200" in self.hist_data.columns:
            ax_price.plot(
                self.hist_data.index,
                self.hist_data["MA200"],
                label="200-Day MA",
                color="#E57373",
                linewidth=1.5,
                alpha=0.8
            )
        
        # Add Bollinger Bands
        if self.show_bollinger.get():
            ax_price.plot(
                self.hist_data.index,
                self.hist_data["BB_Upper"],
                label="Upper BB",
                color="#9575CD",
                linewidth=1,
                linestyle="--",
                alpha=0.6
            )
            ax_price.plot(
                self.hist_data.index,
                self.hist_data["BB_Lower"],
                label="Lower BB",
                color="#9575CD",
                linewidth=1,
                linestyle="--",
                alpha=0.6
            )
            ax_price.fill_between(
                self.hist_data.index,
                self.hist_data["BB_Upper"],
                self.hist_data["BB_Lower"],
                alpha=0.1,
                color="#9575CD"
            )
        
        # Configure price axis
        ax_price.set_ylabel("Price ($)", color=text_color, fontsize=11)
        ax_price.set_title(
            f"{self.ticker} - {self.stock_name} ({self.current_period.upper()})",
            color=text_color,
            fontsize=14,
            fontweight="bold",
            pad=20
        )
        ax_price.grid(True, alpha=0.2, color=grid_color)
        ax_price.legend(loc="upper left", framealpha=0.9, facecolor=self.theme["button"])
        ax_price.tick_params(colors=text_color)
        ax_price.spines["bottom"].set_color(grid_color)
        ax_price.spines["top"].set_color(grid_color)
        ax_price.spines["left"].set_color(grid_color)
        ax_price.spines["right"].set_color(grid_color)
        
        # Format x-axis
        if ax_volume is None:
            ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            plt.setp(ax_price.xaxis.get_majorticklabels(), visible=False)
        
        # Draw volume chart
        if ax_volume is not None:
            self._draw_volume_chart(ax_volume)
            ax_volume.set_ylabel("Volume", color=text_color, fontsize=11)
            ax_volume.set_xlabel("Date", color=text_color, fontsize=11)
            ax_volume.grid(True, alpha=0.2, color=grid_color)
            ax_volume.tick_params(colors=text_color)
            ax_volume.spines["bottom"].set_color(grid_color)
            ax_volume.spines["top"].set_color(grid_color)
            ax_volume.spines["left"].set_color(grid_color)
            ax_volume.spines["right"].set_color(grid_color)
            ax_volume.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Apply tight layout and redraw
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _draw_line_chart(self, ax):
        """Draw a simple line chart."""
        ax.plot(
            self.hist_data.index,
            self.hist_data["Close"],
            label="Close Price",
            color="#64B5F6",
            linewidth=2
        )
    
    def _draw_area_chart(self, ax):
        """Draw an area chart."""
        ax.fill_between(
            self.hist_data.index,
            self.hist_data["Close"],
            alpha=0.3,
            color="#64B5F6",
            label="Close Price"
        )
        ax.plot(
            self.hist_data.index,
            self.hist_data["Close"],
            color="#64B5F6",
            linewidth=2
        )
    
    def _draw_candlestick_chart(self, ax):
        """Draw a candlestick chart."""
        # Prepare data
        up = self.hist_data[self.hist_data.Close >= self.hist_data.Open]
        down = self.hist_data[self.hist_data.Close < self.hist_data.Open]
        
        width = 0.6
        width2 = 0.05
        
        # Up candles (green)
        ax.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color="#66BB6A", alpha=0.8)
        ax.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color="#66BB6A")
        ax.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color="#66BB6A")
        
        # Down candles (red)
        ax.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color="#EF5350", alpha=0.8)
        ax.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color="#EF5350")
        ax.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color="#EF5350")
    
    def _draw_volume_chart(self, ax):
        """Draw volume bars."""
        # Color bars based on price change
        colors = ["#66BB6A" if close >= open_price else "#EF5350" 
                  for close, open_price in zip(self.hist_data["Close"], self.hist_data["Open"])]
        
        ax.bar(
            self.hist_data.index,
            self.hist_data["Volume"],
            color=colors,
            alpha=0.6,
            width=0.8
        )


def open_chart_window(parent: tk.Tk, theme: Dict[str, str], ticker: str, 
                      stock_name: str = "", period: str = "1y"):
    """Open a new chart window for a stock.
    
    Args:
        parent: Parent Tkinter window
        theme: Color theme dictionary
        ticker: Stock ticker symbol
        stock_name: Company name for display
        period: Initial time period (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
    """
    try:
        chart_window = HistoricalChartWindow(parent, theme, ticker, stock_name, period)
    except Exception as e:
        messagebox.showerror("Chart Error", f"Failed to open chart: {str(e)}")


# Example usage for testing
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Dark theme for testing
    theme = {
        "background": "#1e1e1e",
        "text": "#e0e0e0",
        "entry": "#2d2d2d",
        "button": "#3a3a3a",
    }
    
    open_chart_window(root, theme, "AAPL", "Apple Inc.", "1y")
    root.mainloop()
