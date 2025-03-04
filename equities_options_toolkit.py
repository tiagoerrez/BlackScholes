# import modules
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_options_chain(ticker, expiration):
    """
    Get the current options chain based on a ticker and expiration date
    """
    ticker_obj = yf.Ticker(ticker)
    option_chain = ticker_obj.option_chain(expiration) # Returns tuple (calls, puts)

    # process calls and puts
    calls = option_chain.calls
    puts = option_chain.puts

    return {'calls': calls, 'puts':puts}

def plot_IVskew(ticker, expiration):
    """
    Plot the Implied Volatility skew chart, weighted by volume and open interest.
    The aim is visualize Implied Volatility accross different strikes for both calls and puts.
    """
    chain_data = get_options_chain(ticker, expiration)
    current_price = yf.Ticker(ticker).info['regularMarketPrice'] # Live price

    plt.figure(figsize=(14,7))

    sns.lineplot(x='strike', y='impliedVolatility', data=chain_data['puts'], marker='o', color='blue', label='Puts')
    sns.lineplot(x='strike', y='impliedVolatility', data=chain_data['calls'], marker='s', color='red', label='Calls')
    plt.axvline(x=current_price, linestyle='--', color='green', label='Current Price')  

    plt.title(f'Implied Volatility Skew by Strike Prices - Expiration: {expiration}')
    plt.xlabel('Strike Prices')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_weighted_cpivs(ticker, expiration):
    """
    Calculate the Call-Put Implied Volatility Spread (CPIV) in the magnitude of implied volatility.

    Parameters:
    df (DataFrame): DataFrame containing columns 'Strike', 'Implied_Volatility', 'Open_Interest', 'Volume'

    Returns:
    float: CPIV in the magnitude of implied volatility
    """

    chain_data = get_options_chain(ticker, expiration)
    
    # Weighted average IV for calls
    calls = chain_data['calls']
    weighted_avg_iv_call = (calls['impliedVolatility'] * (calls['openInterest'] + calls['volume'])).sum() / (calls['openInterest'] + calls['volume']).sum()
    
    # Weighted average IV for puts
    puts = chain_data['puts']
    weighted_avg_iv_put = (puts['impliedVolatility'] * (puts['openInterest'] + puts['volume'])).sum() / (puts['openInterest'] + puts['volume']).sum()
    
    # CPIV
    weighted_cpiv = weighted_avg_iv_call - weighted_avg_iv_put
    return weighted_cpiv

def get_and_sort_cpivs_for_tickers(tickers):
    """
    Get CPIV spread for all tickers provided for all expirations posible
    """
    cpiv_data = []

    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            expirations = ticker_obj.options # list of expirations dates
            for expiration in expirations:
                cpiv = calculate_weighted_cpivs(ticker, expiration)
                cpiv_data.append({'Ticker': ticker, 'Expiration': expiration, 'CPIV': cpiv})

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Sort the list of dictionaries by CPIV in descending order
    sorted_cpiv_data = sorted(cpiv_data, key=lambda x: x['CPIV'], reverse=True)

    # Print or return the sorted data
    for item in sorted_cpiv_data:
        print(f"{item['Ticker']} - Expiration: {item['Expiration']}, CPIV: {item['CPIV']}")

    return sorted_cpiv_data

def get_CPIVbyExpiration(tickers, expiration):
    """
    Get CPIV spread for all tickers provided and a given expiration
    """
    cpiv_data = []

    for ticker in tickers:
        try:
            # Calculate CPIV for the current options chain
            cpiv = calculate_weighted_cpivs(ticker, expiration)
            # Append ticker, expiration, and CPIV to the list
            cpiv_data.append({'Ticker': ticker, 'Expiration': expiration, 'CPIV': cpiv})
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Sort the list of dictionaries by CPIV in descending order
    sorted_cpiv_data = sorted(cpiv_data, key=lambda x: x['CPIV'], reverse=True)

    # Print or return the sorted data
    for item in sorted_cpiv_data:
        print(f"{item['Ticker']} - Expiration: {item['Expiration']}, CPIV: {item['CPIV']}")

    return sorted_cpiv_data

def vix_dynamic_allocation(balance=5000):
    """
    Dynamically determine the maximum portfolio allocation to short premium strategies based on the VIX.
    
    Parameters:
    - balance (float): Portfolio balance (default=5000).
    - fallback_vix (float): Fallback VIX value if fetching fails (default=20.0).
    
    Returns:
    - float: Maximum allocation amount based on VIX.
    """
    try:
        # Fetch VIX data using history for the latest close (more reliable than info)
        vix_ticker = yf.Ticker('^VIX')
        vix_data = vix_ticker.history(period='1d', interval='1d')
        if vix_data.empty:
            raise ValueError("No VIX data returned")
        current_vix = vix_data['Close'].iloc[-1].round(2)
    except Exception as e:
        print(f"Error fetching VIX with yfinance: {e}")
        current_vix = 15  # Use fallback value
    
    # Define allocation tiers
    if current_vix < 15:
        allocation_percentage = 0.25  # 25% MPA
    elif 15 <= current_vix < 20:
        allocation_percentage = 0.3   # 30% MPA
    elif 20 <= current_vix < 30:
        allocation_percentage = 0.35  # 35% MPA
    elif 30 <= current_vix < 40:
        allocation_percentage = 0.4   # 40% MPA
    else:
        allocation_percentage = 0.5   # 50% MPA
    
    allocation_value = balance * allocation_percentage
    print(f"VIX: {current_vix}, Allocation: {allocation_percentage*100}% of ${balance} = ${allocation_value:.2f}")
    return allocation_value

def get_prices(tickers, period='1y'):
    """
    Fetch a DataFrame of adjusted close prices for a list of tickers over a given period.
    
    Parameters:
    - tickers (list): List of ticker symbols (e.g., ['AAPL', 'MSFT']).
    - period (str): Period for data (e.g., '30d', '1mo', '1y', 'max'). Default is '1y'.
    
    Returns:
    - pd.DataFrame: Adjusted close prices.
    """
    # Use yf.download for efficiency with multiple tickers
    prices = yf.download(tickers, period=period, interval='1d')['Close']
    if isinstance(prices, pd.Series):  # Handle single ticker case
        prices = prices.to_frame(name=tickers[0])
    return prices

def corr_2assets(ticker1, ticker2, period='1y'):
    """
    Calculate the correlation between two assets over a given period.
    
    Parameters:
    - ticker1 (str): First ticker symbol (e.g., 'AAPL').
    - ticker2 (str): Second ticker symbol (e.g., 'MSFT').
    - period (str): Period for data (e.g., '30d', '1mo', '1y', 'max'). Default is '30d'.
    
    Returns:
    - float: Correlation coefficient.
    """
    prices = get_prices([ticker1, ticker2], period=period)
    returns = prices.pct_change().dropna()
    correlation = returns[ticker1].corr(returns[ticker2])
    return correlation

def show_corr_matrix(tickers, period='1y'):
    """
    Display a correlation matrix for a set of assets over a given period.
    
    Parameters:
    - tickers (list): List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL']).
    - period (str): Period for data (e.g., '30d', '1mo', '1y', 'max'). Default is '30d'.
    
    Returns:
    - Styled DataFrame: Correlation matrix with gradient.
    """
    prices = get_prices(tickers, period=period)
    returns = prices.pct_change().dropna()
    if len(tickers) < 2:
        print("Provide at least two tickers for correlation.")
        return None
    corr_matrix = returns.corr().style.background_gradient(cmap='coolwarm')
    return corr_matrix

def plot_corr_over_time(ticker1, ticker2, period='1y', window=30):
    """
    Plot the rolling correlation between two assets over time.
    
    Parameters:
    - ticker1 (str): First ticker symbol (e.g., 'AAPL').
    - ticker2 (str): Second ticker symbol (e.g., 'MSFT').
    - period (str): Period for data (e.g., '30d', '1mo', '1y', 'max'). Default is '30d'.
    - window (int): Rolling window size in days (default=30).
    """
    prices = yf.download([ticker1, ticker2], period=period, interval='1d')['Close']
    returns = prices.pct_change().dropna()
    rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2])
    
    plt.figure(figsize=(12, 6))
    rolling_corr.plot()
    plt.title(f'Rolling {window}-Day Correlation Between {ticker1} and {ticker2}')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.5)
    plt.show()


def kelly_criterion_allocation(r=0.05, dte=45, pop=0.7):
    """
    Calculates the proportion of capital to allocate to a position based on the POP 
    """
    f = r * (dte / 365) * (pop / (1-pop))
    print(f"Kelly Criterion Position Size: {round(f,4)}" )
    return f