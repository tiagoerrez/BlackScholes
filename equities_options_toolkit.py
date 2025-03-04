# import modules
from yahoo_fin import options as op
import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


def get_options_chain(ticker, expiration):
    """
    Get the current options chain based on a ticker and expiration date
    """
    chainData = op.get_options_chain(ticker, date=expiration)
    for key in chainData:
        chainData[key]['Implied Volatility'] = chainData[key]['Implied Volatility'].str.replace('%', '').astype(float)/100

    for key in chainData:
        chainData[key]['Volume'] = chainData[key]['Volume'].replace({'-': 0, ',':''}, regex=True).astype(int)
        chainData[key]['Open Interest'] = chainData[key]['Open Interest'].replace({'-': 0, ',':''}, regex=True).astype(float)

    for key in chainData:
        chainData[key].sort_values(by='Strike', inplace=True)

    return chainData

def plot_IVskew(ticker, expiration):
    """
    Plot the Implied Volatility skew chart, weighted by volume and open interest.
    The aim is visualize Implied Volatility accross different strikes for both calls and puts.
    """
    chainData = get_options_chain(ticker, expiration)

    plt.figure(figsize=(14,7))

    # Create a volatility smile or skew plot for puts
    sns.lineplot(x='Strike', y='Implied Volatility', data=chainData['puts'], marker='o', color='blue', label='Puts')

    # Create a volatility smile or skew plot for calls
    sns.lineplot(x='Strike', y='Implied Volatility', data=chainData['calls'], marker='s', color='red', label='Calls')

    # Add a vertical line of the current stock price
    plt.axvline(x=si.get_live_price(ticker), linestyle='--', color='green', label='Current Price')

    plt.title(f'Implied Volatility Skew by Strike Prices - Expiration: {expiration}')
    plt.xlabel('Strike Prices')
    plt.ylabel('Implied Volatility')
    plt.legend() # Add legend to differentiate vetween puts and calls
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

    chainData = get_options_chain(ticker, expiration)

    # Calculate the weighted average implied volatilities
    weighted_avg_iv_call = (chainData['calls']['Implied Volatility'] * (chainData['calls']['Open Interest'] + chainData['calls']['Volume'])).sum() / (chainData['calls']['Open Interest'] + chainData['calls']['Volume']).sum()
    weighted_avg_iv_put = (chainData['puts']['Implied Volatility'] * (chainData['puts']['Open Interest'] + chainData['puts']['Volume'])).sum() / (chainData['puts']['Open Interest'] + chainData['puts']['Volume']).sum()

    # Calculate CPIV as the difference in weighted average implied volatilities
    weighted_cpiv = weighted_avg_iv_call - weighted_avg_iv_put

    # print(f'Weighted Call-Put Implied Volatility Spread (CPIV): {weighted_cpiv}')
    return weighted_cpiv

def get_and_sort_cpivs_for_tickers(tickers):
    """
    Get CPIV spread for all tickers provided for all expirations posible
    """
    cpiv_data = []

    for ticker in tickers:
        try:
            expirations = op.get_expiration_dates(ticker)
            for expiration in expirations:
                # Get options chain for the current ticker and expiration date
                chain_data = get_options_chain(ticker, expiration)

                # Calculate CPIV for the current options chain
                cpiv = calculate_weighted_cpivs(chain_data)

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

def get_CPIVbyExpiration(tickers, expiration):
    """
    Get CPIV spread for all tickers provided and a given expiration
    """
    cpiv_data = []

    for ticker in tickers:
        try:
            # Get options chain for the current ticker and expiration date
            chain_data = get_options_chain(ticker, expiration)

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
        vix_data = vix_ticker.history(period='1d', interval='1d')['Close']
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

def get_prices(tickers):
    """
    Get a DataFrame of prices from a list of tickers (Adjusted Close)
    """
    prices = pd.DataFrame()
    for ticker in tickers:
        prices[ticker] = si.get_data(ticker)['adjclose']
    return prices

def corr_2assets(ticker1, ticker2):
    """
    Get the correlation between two assets
    """
    asset1 = si.get_data(ticker1)
    asset2 = si.get_data(ticker2)
    correlation = asset1.corr(asset2)
    return correlation

def show_corr_matrix(tickers):
    """
    Show a correlation between a set of returns
    """
    prices = get_prices(tickers)
    if isinstance(prices, pd.Series):
        print('Returns is an array and not a set of assets. Provide a set of returns')

    corr_matrix = prices.select_dtypes(exclude='object').corr().style.background_gradient(cmap='coolwarm')
    return corr_matrix

def kelly_criterion_allocation(r=0.05, dte=45, pop=0.7):
    """
    Calculates the proportion of capital to allocate to a position based on the POP 
    """
    f = r * (dte / 365) * (pop / (1-pop))
    print(f"Kelly Criterion Position Size: {round(f,4)}" )
    return f