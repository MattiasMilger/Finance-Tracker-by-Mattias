"""Historical Price Charts Module for Stock Tracker Application.

Provides beautiful, interactive charts with multiple timeframes, technical indicators,
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
            initial_period: Initial time period to display
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
        self.show_ema = tk.BooleanVar(value=False)
        self.chart_type = tk.StringVar(value="line")
        
        # Data storage
        self.hist_data: Optional[pd.DataFrame] = None
        
        # Create window with larger size
        self.window = tk.Toplevel(parent)
        self.window.title(f"📈 {ticker} - {self.stock_name}")
        self.window.geometry("1600x950")
        self.window.configure(bg=theme["background"])
        
        # Setup UI
        self._setup_ui()
        
        # Load initial data
        self.load_data()
    
    def _setup_ui(self):
        """Create and layout all UI components."""
        # Top control panel with better styling
        control_frame = tk.Frame(self.window, bg=self.theme["background"])
        control_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Left side - period selection with better buttons
        period_frame = tk.LabelFrame(
            control_frame, 
            text="📅 Time Period", 
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 10, "bold"),
            padx=10,
            pady=8
        )
        period_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        periods = [
            ("1W", "5d"),
            ("1M", "1mo"),
            ("3M", "3mo"),
            ("6M", "6mo"),
            ("YTD", "ytd"),
            ("1Y", "1y"),
            ("2Y", "2y"),
            ("5Y", "5y"),
            ("Max", "max")
        ]
        
        self.period_buttons = {}
        for i, (label, period) in enumerate(periods):
            btn = tk.Button(
                period_frame,
                text=label,
                command=lambda p=period: self.change_period(p),
                bg=self.theme["button"] if period != self.current_period else "#4a7ba7",
                fg=self.theme["text"],
                font=("Arial", 9, "bold"),
                width=6,
                height=1,
                relief=tk.RAISED,
                bd=2,
                cursor="hand2"
            )
            btn.grid(row=0, column=i, padx=2)
            self.period_buttons[period] = btn
            
        # Middle - chart type
        chart_frame = tk.LabelFrame(
            control_frame,
            text="📊 Chart Style",
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 10, "bold"),
            padx=10,
            pady=8
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
                activeforeground=self.theme["text"],
                font=("Arial", 9)
            )
            rb.grid(row=0, column=i, padx=8)
        
        # Right side - indicators
        indicator_frame = tk.LabelFrame(
            control_frame,
            text="📉 Technical Indicators",
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 10, "bold"),
            padx=10,
            pady=8
        )
        indicator_frame.pack(side=tk.LEFT)
        
        indicators = [
            ("50-Day MA", self.show_ma50),
            ("200-Day MA", self.show_ma200),
            ("EMA (20)", self.show_ema),
            ("Bollinger Bands", self.show_bollinger),
            ("Volume", self.show_volume)
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
                activeforeground=self.theme["text"],
                font=("Arial", 9)
            )
            cb.grid(row=0, column=i, padx=8)
        
        # Refresh button
        tk.Button(
            control_frame,
            text="🔄 Refresh",
            command=self.load_data,
            bg="#4a7ba7",
            fg=self.theme["text"],
            font=("Arial", 10, "bold"),
            width=10,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.RIGHT, padx=10)
        
        # Chart container with matplotlib
        chart_container = tk.Frame(self.window, bg=self.theme["background"])
        chart_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Create matplotlib figure with larger size and better styling
        self.fig = Figure(figsize=(16, 9), facecolor=self.theme["background"], dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = tk.Frame(chart_container, bg=self.theme["background"])
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Status bar with better styling
        self.status_label = tk.Label(
            self.window,
            text="Ready",
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 10),
            anchor=tk.W,
            relief=tk.FLAT,
            padx=15
        )
        self.status_label.pack(fill=tk.X, pady=(0, 10))
    
    def change_period(self, period: str):
        """Change the time period and reload data."""
        self.current_period = period
        self.load_data()
        
        # Update button states
        for p, btn in self.period_buttons.items():
            if p == period:
                btn.config(bg="#4a7ba7", relief=tk.SUNKEN)
            else:
                btn.config(bg=self.theme["button"], relief=tk.RAISED)
    
    def load_data(self):
        """Fetch historical data from yfinance and update chart."""
        self.status_label.config(text=f"⏳ Loading data for {self.ticker}...")
        self.window.update_idletasks()
        
        try:
            stock = yf.Ticker(self.ticker)
            
            # Determine interval based on period for better granularity
            interval_map = {
                "5d": "15m",
                "1mo": "1d",
                "3mo": "1d",
                "6mo": "1d",
                "ytd": "1d",
                "1y": "1d",
                "2y": "1wk",
                "5y": "1wk",
                "max": "1mo"
            }
            interval = interval_map.get(self.current_period, "1d")
            
            # Fetch data
            self.hist_data = stock.history(period=self.current_period, interval=interval)
            
            if self.hist_data.empty:
                messagebox.showerror("Error", f"No data available for {self.ticker}")
                self.status_label.config(text="❌ Error: No data available")
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
            
            period_display = self._get_period_display(date_diff)
            
            self.status_label.config(
                text=f"✅ Showing {len(self.hist_data)} data points from {start_date} to {end_date} ({period_display})"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_label.config(text=f"❌ Error: {str(e)}")
    
    def _get_period_display(self, date_diff):
        """Get human-readable period display."""
        days = date_diff.days
        
        if days < 7:
            return f"{days} days"
        elif days < 60:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''}"
        elif days < 365:
            months = days // 30
            return f"{months} month{'s' if months > 1 else ''}"
        else:
            years = days / 365.25
            return f"{years:.1f} year{'s' if years > 1 else ''}"
    
    def _calculate_indicators(self):
        """Calculate technical indicators on historical data."""
        if self.hist_data is None or self.hist_data.empty:
            return
        
        # Moving averages
        self.hist_data["MA50"] = self.hist_data["Close"].rolling(window=50).mean()
        self.hist_data["MA200"] = self.hist_data["Close"].rolling(window=200).mean()
        
        # Exponential Moving Average
        self.hist_data["EMA20"] = self.hist_data["Close"].ewm(span=20, adjust=False).mean()
        
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
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
            ax_price = self.fig.add_subplot(gs[0])
            ax_volume = self.fig.add_subplot(gs[1], sharex=ax_price)
        else:
            ax_price = self.fig.add_subplot(111)
            ax_volume = None
        
        # Configure colors for dark theme with better contrast
        text_color = "#e8e8e8"
        grid_color = "#404040"
        price_color = "#64B5F6"
        
        # Draw price chart
        if self.chart_type.get() == "line":
            self._draw_line_chart(ax_price, price_color)
        elif self.chart_type.get() == "candlestick":
            self._draw_candlestick_chart(ax_price)
        elif self.chart_type.get() == "area":
            self._draw_area_chart(ax_price, price_color)
        
        # Add EMA
        if self.show_ema.get() and "EMA20" in self.hist_data.columns:
            ax_price.plot(
                self.hist_data.index,
                self.hist_data["EMA20"],
                label="20-Day EMA",
                color="#9C27B0",
                linewidth=2,
                alpha=0.8,
                linestyle="--"
            )
        
        # Add moving averages with better styling
        if self.show_ma50.get() and "MA50" in self.hist_data.columns:
            ax_price.plot(
                self.hist_data.index,
                self.hist_data["MA50"],
                label="50-Day MA",
                color="#FFB74D",
                linewidth=2.5,
                alpha=0.9
            )
        
        if self.show_ma200.get() and "MA200" in self.hist_data.columns:
            ax_price.plot(
                self.hist_data.index,
                self.hist_data["MA200"],
                label="200-Day MA",
                color="#E57373",
                linewidth=2.5,
                alpha=0.9
            )
        
        # Add Bollinger Bands with better styling
        if self.show_bollinger.get():
            ax_price.plot(
                self.hist_data.index,
                self.hist_data["BB_Upper"],
                label="Upper BB",
                color="#AB47BC",
                linewidth=1.5,
                linestyle=":",
                alpha=0.7
            )
            ax_price.plot(
                self.hist_data.index,
                self.hist_data["BB_Lower"],
                label="Lower BB",
                color="#AB47BC",
                linewidth=1.5,
                linestyle=":",
                alpha=0.7
            )
            ax_price.fill_between(
                self.hist_data.index,
                self.hist_data["BB_Upper"],
                self.hist_data["BB_Lower"],
                alpha=0.15,
                color="#AB47BC"
            )
        
        # Configure price axis with better styling
        ax_price.set_ylabel("Price ($)", color=text_color, fontsize=12, fontweight="bold")
        
        # Better title with more info
        current_price = self.hist_data["Close"].iloc[-1]
        price_change = self.hist_data["Close"].iloc[-1] - self.hist_data["Close"].iloc[0]
        price_change_pct = (price_change / self.hist_data["Close"].iloc[0]) * 100
        change_color = "#66BB6A" if price_change >= 0 else "#EF5350"
        
        title_text = f"{self.ticker} - {self.stock_name}\n"
        title_text += f"${current_price:.2f}  "
        title_text += f"{'▲' if price_change >= 0 else '▼'} ${abs(price_change):.2f} ({price_change_pct:+.2f}%)"
        
        ax_price.set_title(
            title_text,
            color=text_color,
            fontsize=15,
            fontweight="bold",
            pad=20
        )
        
        # Better grid
        ax_price.grid(True, alpha=0.3, color=grid_color, linestyle="--", linewidth=0.8)
        ax_price.set_axisbelow(True)
        
        # Better legend
        legend = ax_price.legend(
            loc="upper left", 
            framealpha=0.95, 
            facecolor=self.theme["button"],
            edgecolor=grid_color,
            fontsize=10
        )
        for text in legend.get_texts():
            text.set_color(text_color)
        
        # Style axes
        ax_price.tick_params(colors=text_color, labelsize=10)
        ax_price.spines["bottom"].set_color(grid_color)
        ax_price.spines["top"].set_color(grid_color)
        ax_price.spines["left"].set_color(grid_color)
        ax_price.spines["right"].set_color(grid_color)
        
        # Format x-axis with better date formatting
        if ax_volume is None:
            self._format_xaxis(ax_price, text_color)
        else:
            plt.setp(ax_price.xaxis.get_majorticklabels(), visible=False)
        
        # Draw volume chart with better styling
        if ax_volume is not None:
            self._draw_volume_chart(ax_volume)
            ax_volume.set_ylabel("Volume", color=text_color, fontsize=11, fontweight="bold")
            ax_volume.set_xlabel("Date", color=text_color, fontsize=11, fontweight="bold")
            ax_volume.grid(True, alpha=0.3, color=grid_color, linestyle="--", linewidth=0.8)
            ax_volume.set_axisbelow(True)
            ax_volume.tick_params(colors=text_color, labelsize=10)
            ax_volume.spines["bottom"].set_color(grid_color)
            ax_volume.spines["top"].set_color(grid_color)
            ax_volume.spines["left"].set_color(grid_color)
            ax_volume.spines["right"].set_color(grid_color)
            self._format_xaxis(ax_volume, text_color)
        
        # Apply tight layout and redraw
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _format_xaxis(self, ax, text_color):
        """Format x-axis with appropriate date format based on period."""
        if self.current_period in ["5d", "1mo"]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(self.hist_data) // 10)))
        elif self.current_period in ["3mo", "6mo"]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=max(1, len(self.hist_data) // 15)))
        elif self.current_period in ["ytd", "1y"]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", color=text_color)
    
    def _draw_line_chart(self, ax, color):
        """Draw a simple line chart with better styling."""
        ax.plot(
            self.hist_data.index,
            self.hist_data["Close"],
            label="Close Price",
            color=color,
            linewidth=2.5,
            alpha=0.9
        )
    
    def _draw_area_chart(self, ax, color):
        """Draw an area chart with gradient effect."""
        ax.fill_between(
            self.hist_data.index,
            self.hist_data["Close"],
            alpha=0.4,
            color=color,
            label="Close Price"
        )
        ax.plot(
            self.hist_data.index,
            self.hist_data["Close"],
            color=color,
            linewidth=2.5,
            alpha=0.9
        )
    
    def _draw_candlestick_chart(self, ax):
        """Draw a candlestick chart with better colors."""
        up = self.hist_data[self.hist_data.Close >= self.hist_data.Open]
        down = self.hist_data[self.hist_data.Close < self.hist_data.Open]
        
        width = 0.8
        width2 = 0.1
        
        # Up candles (green) with better styling
        ax.bar(up.index, up.Close - up.Open, width, bottom=up.Open, 
               color="#26A69A", alpha=0.9, edgecolor="#1B7E72", linewidth=1.5)
        ax.bar(up.index, up.High - up.Close, width2, bottom=up.Close, 
               color="#26A69A", alpha=0.9)
        ax.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, 
               color="#26A69A", alpha=0.9)
        
        # Down candles (red) with better styling
        ax.bar(down.index, down.Close - down.Open, width, bottom=down.Open, 
               color="#EF5350", alpha=0.9, edgecolor="#C62828", linewidth=1.5)
        ax.bar(down.index, down.High - down.Open, width2, bottom=down.Open, 
               color="#EF5350", alpha=0.9)
        ax.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, 
               color="#EF5350", alpha=0.9)
    
    def _draw_volume_chart(self, ax):
        """Draw volume bars with better colors."""
        colors = ["#26A69A" if close >= open_price else "#EF5350" 
                  for close, open_price in zip(self.hist_data["Close"], self.hist_data["Open"])]
        
        ax.bar(
            self.hist_data.index,
            self.hist_data["Volume"],
            color=colors,
            alpha=0.7,
            width=0.8,
            edgecolor="none"
        )
        
        # Format volume axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))


def open_chart_window(parent: tk.Tk, theme: Dict[str, str], ticker: str, 
                      stock_name: str = "", period: str = "1y"):
    """Open a new chart window for a stock.
    
    Args:
        parent: Parent Tkinter window
        theme: Color theme dictionary
        ticker: Stock ticker symbol
        stock_name: Company name for display
        period: Initial time period (5d, 1mo, 3mo, 6mo, ytd, 1y, 2y, 5y, max)
    """
    try:
        chart_window = HistoricalChartWindow(parent, theme, ticker, stock_name, period)
    except Exception as e:
        messagebox.showerror("Chart Error", f"Failed to open chart: {str(e)}")


# Example usage for testing
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    theme = {
        "background": "#1e1e1e",
        "text": "#e0e0e0",
        "entry": "#2d2d2d",
        "button": "#3a3a3a",
    }
    
    open_chart_window(root, theme, "AAPL", "Apple Inc.", "1y")
    root.mainloop()
