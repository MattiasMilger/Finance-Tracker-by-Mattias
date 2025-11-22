"""Stock Tracker Application - Main Entry Point

A comprehensive stock tracking application with real-time data, technical analysis,
and customizable recommendations.

"""

import tkinter as tk
from main_window import StockTrackerApp


def main():
    """Main entry point for the Stock Tracker application."""
    root = tk.Tk()
    app = StockTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
