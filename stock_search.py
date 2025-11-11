"""
Stock Search Module for Stock Tracker
Provides a dialog to search and preview stocks before adding them to the tracker.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, Dict, Callable, List
import yfinance as yf
# Re-import yfinance as yf for use of yf.Search

class StockSearchDialog:
    """Dialog for searching and previewing stock information."""
    
    def __init__(self, parent: tk.Tk, theme: Dict[str, str], on_add_callback: Callable[[str], None]):
        """
        Initialize the stock search dialog.
        
        Args:
            parent: Parent Tkinter window
            theme: Theme dictionary with color settings
            on_add_callback: Callback function to add ticker to main tracker
        """
        self.parent = parent
        self.theme = theme
        self.on_add_callback = on_add_callback
        self.current_stock_data: Optional[Dict] = None
        self.search_results: List[Dict] = []
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Search for Stocks")
        self.dialog.geometry("800x600")
        self.dialog.configure(bg=theme["background"])
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Search Frame
        search_frame = tk.Frame(self.dialog, bg=self.theme["background"])
        search_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            search_frame, 
            text="Search by Ticker or Company Name:", 
            bg=self.theme["background"], 
            fg=self.theme["text"],
            font=("Arial", 11, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))
        
        entry_frame = tk.Frame(search_frame, bg=self.theme["background"])
        entry_frame.pack(fill=tk.X)
        
        self.search_entry = tk.Entry(
            entry_frame,
            width=30,
            bg=self.theme["entry"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            font=("Arial", 12)
        )
        self.search_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.search_entry.bind("<Return>", lambda e: self.search_stock())
        self.search_entry.focus()
        
        tk.Button(
            entry_frame,
            text="Search",
            command=self.search_stock,
            bg=self.theme["button"],
            fg=self.theme["text"],
            font=("Arial", 10, "bold"),
            width=12
        ).pack(side=tk.LEFT)
        
        tk.Label(
            search_frame,
            text="Examples: AAPL, Apple, Microsoft, TSLA, NVDA.ST",
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 9, "italic")
        ).pack(anchor=tk.W, pady=(5, 0))
        
        # Results List Frame (for multiple results)
        self.results_list_frame = tk.LabelFrame(
            self.dialog,
            text="Search Results",
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 10, "bold"),
            padx=10,
            pady=10
        )
        # Don't pack yet - will show when needed
        
        # Create Treeview for search results
        tree_frame = tk.Frame(self.results_list_frame, bg=self.theme["background"])
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        tree_scroll = tk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_tree = ttk.Treeview(
            tree_frame,
            columns=("Ticker", "Name", "Exchange"),
            show="headings",
            yscrollcommand=tree_scroll.set,
            height=8
        )
        tree_scroll.config(command=self.results_tree.yview)
        
        self.results_tree.heading("Ticker", text="Ticker")
        self.results_tree.heading("Name", text="Company Name")
        self.results_tree.heading("Exchange", text="Exchange")
        
        self.results_tree.column("Ticker", width=100)
        self.results_tree.column("Name", width=400)
        self.results_tree.column("Exchange", width=150)
        
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        self.results_tree.bind("<Double-1>", lambda e: self.on_result_selected())
        
        # Style the treeview
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background=self.theme["tree_bg"],
                        foreground=self.theme["tree_fg"],
                        fieldbackground=self.theme["tree_bg"],
                        borderwidth=0)
        style.map("Treeview", background=[("selected", self.theme["button"])])
        
        select_btn_frame = tk.Frame(self.results_list_frame, bg=self.theme["background"])
        select_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Button(
            select_btn_frame,
            text="Select & View Details",
            command=self.on_result_selected,
            bg=self.theme["button"],
            fg=self.theme["text"],
            font=("Arial", 10, "bold")
        ).pack()
        
        # Detail Results Frame (for single stock details)
        self.detail_frame = tk.LabelFrame(
            self.dialog,
            text="Stock Information",
            bg=self.theme["background"],
            fg=self.theme["text"],
            font=("Arial", 10, "bold"),
            padx=10,
            pady=10
        )
        self.detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Scrollable text widget for results
        scroll = tk.Scrollbar(self.detail_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(
            self.detail_frame,
            bg=self.theme["tree_bg"],
            fg=self.theme["text"],
            font=("Consolas", 10),
            wrap=tk.WORD,
            yscrollcommand=scroll.set,
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.results_text.yview)
        
        # Button Frame
        button_frame = tk.Frame(self.dialog, bg=self.theme["background"])
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.add_button = tk.Button(
            button_frame,
            text="Add to Tracker",
            command=self.add_to_tracker,
            bg=self.theme["button"],
            fg=self.theme["text"],
            font=("Arial", 10, "bold"),
            width=15,
            state=tk.DISABLED
        )
        self.add_button.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            button_frame,
            text="Close",
            command=self.dialog.destroy,
            bg=self.theme["button"],
            fg=self.theme["text"],
            font=("Arial", 10),
            width=15
        ).pack(side=tk.LEFT)
    
    def on_result_selected(self) -> None:
        """Handle selection from search results list."""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a stock from the list.")
            return
        
        item = self.results_tree.item(selection[0])
        ticker = item['values'][0]
        
        # Hide results list, show details
        self.results_list_frame.pack_forget()
        self.detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Fetch and display detailed info
        self._fetch_and_display_details(ticker)

    def _search_by_name(self, company_name: str) -> None:
        """
        Uses the official yf.Search() class to find stock quotes by name/ticker.
        Populates the Treeview with results.
        """
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, f"Searching using yf.Search() for: {company_name}...\n")
        self.results_text.config(state=tk.DISABLED)
        self.dialog.update_idletasks()
        
        # Clear old results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        self.search_results.clear()
        
        try:
            # Use the official yfinance Search class (reliable in recent versions like 0.2.65)
            search_obj = yf.Search(company_name, max_results=10)
            
            # The .quotes attribute is a list of dictionaries with search results
            quotes = search_obj.quotes
            
            if not quotes:
                self._display_error(f"No stocks found matching '{company_name}'.")
                return

            for result in quotes:
                # Filter results to only include relevant asset types
                if result.get('typeDisp') in ['Equity', 'ETF']: 
                    ticker = result.get('symbol', 'N/A')
                    name = result.get('longname', result.get('shortname', 'N/A'))
                    exchange = result.get('exchDisp', 'N/A')
                    
                    if ticker != 'N/A' and name != 'N/A':
                        self.search_results.append({
                            'ticker': ticker,
                            'name': name,
                            'exchange': exchange
                        })
                        self.results_tree.insert("", tk.END, values=(ticker, name, exchange))
            
            if not self.search_results:
                self._display_error(f"No stock quotes found for '{company_name}'.")
                return
            
            # Show the list of results
            self.detail_frame.pack_forget()
            self.results_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Multiple results found. Please double-click or select a result and click 'Select & View Details'.\n")
            self.results_text.config(state=tk.DISABLED)
            
            # Auto-select and display if only one perfect match is found (optional enhancement)
            if len(self.search_results) == 1:
                ticker = self.search_results[0]['ticker']
                self.results_list_frame.pack_forget()
                self.detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Best match found: {self.search_results[0]['name']} ({ticker}). Fetching details...\n")
                self.results_text.config(state=tk.DISABLED)
                self._fetch_and_display_details(ticker)

        except Exception as e:
            self._display_error(f"Search API failed for '{company_name}': {str(e)}")

    def search_stock(self) -> None:
        """Search for stock information by ticker or company name."""
        query = self.search_entry.get().strip()
        
        if not query:
            messagebox.showwarning("Empty Input", "Please enter a ticker symbol or company name.")
            return
        
        # --- UI State Reset ---
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Searching for '{query}'...\n")
        self.results_text.config(state=tk.DISABLED)
        self.add_button.config(state=tk.DISABLED)
        self.current_stock_data = None
        
        # Clear treeview
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        self.results_list_frame.pack_forget() # Hide list by default
        self.detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10)) # Show detail view by default
        self.dialog.update_idletasks()
        
        # --- Determine Search Type ---
        query_upper = query.upper()
        # Heuristic: if it's short (<= 10 chars) and alphanumeric/symbolic, assume it might be a direct ticker
        is_likely_ticker = len(query) <= 10 and (query.replace('.', '').replace('-', '').isalnum() or '.' in query or '-' in query)
        
        if is_likely_ticker:
            # 1. Try direct ticker fetch
            self.results_text.config(state=tk.NORMAL)
            self.results_text.insert(tk.END, f"Attempting direct fetch for ticker: {query_upper}...\n")
            self.results_text.config(state=tk.DISABLED)
            self._fetch_and_display_details(query_upper)
            
            if self.current_stock_data is None:
                # 2. If direct fetch fails, fall back to yf.Search() for broader lookup (e.g., user typed "TSL" for "TSLA")
                self.results_text.config(state=tk.NORMAL)
                self.results_text.insert(tk.END, f"Direct fetch failed. Falling back to company name search for: {query}...\n")
                self.results_text.config(state=tk.DISABLED)
                self._search_by_name(query)

        else:
            # Input is clearly a company name, use yf.Search() immediately
            self._search_by_name(query)
    
    def _fetch_and_display_details(self, ticker: str) -> None:
        """Fetch and display detailed information for a specific ticker."""
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Validate stock exists
            if not info.get('regularMarketPrice') and not info.get('currentPrice'):
                self._display_error(f"Stock '{ticker}' not found or has no price data.")
                return
            
            # Get price data
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            prev_close = info.get('previousClose')
            
            # Calculate daily change
            daily_change = ""
            daily_change_pct = ""
            if current_price and prev_close:
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                daily_change = f"{change:+.2f}"
                daily_change_pct = f"{change_pct:+.2f}%"
            
            # Store data
            self.current_stock_data = {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'price': current_price,
                'info': info
            }
            
            # Display results
            self._display_results(ticker, info, current_price, daily_change, daily_change_pct)
            self.add_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self._display_error(f"Error fetching data for '{ticker}': {str(e)}")
    
    def _display_results(self, ticker: str, info: dict, price: float, 
                         daily_change: str, daily_change_pct: str) -> None:
        """Display stock information in the results text widget."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Format the display
        result = f"{'='*60}\n"
        result += f"  {info.get('longName', info.get('shortName', 'N/A'))}\n"
        result += f"  {ticker}\n"
        result += f"{'='*60}\n\n"
        
        result += "PRICE INFORMATION\n"
        result += f"  Current Price:           {info.get('currency', 'USD')} {price:.2f}\n"
        if daily_change:
            result += f"  Daily Change:             {daily_change} ({daily_change_pct})\n"
        if info.get('previousClose'):
            result += f"  Previous Close:           {info.get('currency', 'USD')} {info.get('previousClose'):.2f}\n"
        if info.get('open'):
            result += f"  Open:                     {info.get('currency', 'USD')} {info.get('open'):.2f}\n"
        if info.get('dayHigh') and info.get('dayLow'):
            result += f"  Day Range:                {info.get('dayLow'):.2f} - {info.get('dayHigh'):.2f}\n"
        
        result += "\nKEY STATISTICS\n"
        if info.get('marketCap'):
            market_cap = info['marketCap']
            if market_cap >= 1e12:
                result += f"  Market Cap:               ${market_cap/1e12:.2f}T\n"
            elif market_cap >= 1e9:
                result += f"  Market Cap:               ${market_cap/1e9:.2f}B\n"
            elif market_cap >= 1e6:
                result += f"  Market Cap:               ${market_cap/1e6:.2f}M\n"
        
        if info.get('trailingPE'):
            result += f"  P/E Ratio:                {info.get('trailingPE'):.2f}\n"
        
        if info.get('dividendYield'):
            result += f"  Dividend Yield:           {info.get('dividendYield')*100:.2f}%\n"
        
        if info.get('fiftyTwoWeekHigh') and info.get('fiftyTwoWeekLow'):
            result += f"  52-Week Range:            {info.get('fiftyTwoWeekLow'):.2f} - {info.get('fiftyTwoWeekHigh'):.2f}\n"
        
        if info.get('fiftyDayAverage'):
            result += f"  50-Day Average:           {info.get('fiftyDayAverage'):.2f}\n"
        
        if info.get('twoHundredDayAverage'):
            result += f"  200-Day Average:          {info.get('twoHundredDayAverage'):.2f}\n"
        
        result += "\nCOMPANY INFORMATION\n"
        if info.get('sector'):
            result += f"  Sector:                   {info.get('sector')}\n"
        if info.get('industry'):
            result += f"  Industry:                 {info.get('industry')}\n"
        if info.get('country'):
            result += f"  Country:                  {info.get('country')}\n"
        if info.get('website'):
            result += f"  Website:                  {info.get('website')}\n"
        
        if info.get('longBusinessSummary'):
            result += f"\nBUSINESS SUMMARY\n"
            summary = info['longBusinessSummary']
            # Wrap text at 60 characters
            words = summary.split()
            line = "  "
            for word in words:
                if len(line) + len(word) + 1 <= 62:
                    line += word + " "
                else:
                    result += line + "\n"
                    line = "  " + word + " "
            result += line + "\n"
        
        result += f"\n{'='*60}\n"
        
        self.results_text.insert(tk.END, result)
        self.results_text.config(state=tk.DISABLED)
    
    def _display_error(self, message: str) -> None:
        """Display error message in results."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"❌ {message}\n\n")
        self.results_text.insert(tk.END, "Please check the ticker symbol and try again.\n")
        self.results_text.insert(tk.END, "Make sure to include the correct exchange suffix if needed.\n")
        self.results_text.insert(tk.END, "\nExamples:\n")
        self.results_text.insert(tk.END, "  - US stocks: AAPL, MSFT, GOOGL\n")
        self.results_text.insert(tk.END, "  - Swedish stocks: ERIC-B.ST, VOLV-B.ST\n")
        self.results_text.insert(tk.END, "  - UK stocks: BP.L, HSBA.L\n")
        self.results_text.config(state=tk.DISABLED)
        self.current_stock_data = None
    
    def add_to_tracker(self) -> None:
        """Add the searched stock to the main tracker."""
        if not self.current_stock_data:
            return
        
        ticker = self.current_stock_data['ticker']
        
        # Call the callback to add to main tracker
        self.on_add_callback(ticker)
        
        # Show confirmation
        messagebox.showinfo(
            "Added",
            f"'{ticker}' has been added to your tracker!\n\n"
            f"Click 'Fetch Data' to get the full analysis."
        )
        
        # Clear search for next search
        self.search_entry.delete(0, tk.END)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.add_button.config(state=tk.DISABLED)
        self.current_stock_data = None
        
        # Clear and hide results list
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.results_list_frame.pack_forget()
        self.detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.search_entry.focus()


def open_stock_search(parent: tk.Tk, theme: Dict[str, str], 
                      on_add_callback: Callable[[str], None]) -> None:
    """
    Open the stock search dialog.
    
    Args:
        parent: Parent Tkinter window
        theme: Theme dictionary with color settings
        on_add_callback: Callback function to add ticker to main tracker
    """
    StockSearchDialog(parent, theme, on_add_callback)


# For backwards compatibility
__all__ = ['StockSearchDialog', 'open_stock_search']