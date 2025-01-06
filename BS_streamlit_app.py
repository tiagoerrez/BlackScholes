import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp  # Make sure to import these
import matplotlib.pyplot as plt
import seaborn as sns
import yahoo_fin.stock_info as si
from equities_options_toolkit import (
    vix_dynamic_allocation,
    kelly_criterion_allocation
)

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# (Include the BlackScholes class definition here)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity / 365.0 # Convert days to years
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
            ) / (
                volatility * sqrt(time_to_maturity)
            )
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)
        call_delta = self.call_delta 
        put_delta = self.put_delta

        # Gamma
        self.call_gamma = norm.pdf(d1) / (
            strike * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        return call_price, put_price, call_delta, put_delta

# Function to generate heatmaps
# ... your existing imports and BlackScholes class definition ...


# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    # st.write("`Created by:`")
    # linkedin_url = "https://www.linkedin.com/in/mprudhvi/"
    # st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Prudhvi Reddy, Muppala`</a>', unsafe_allow_html=True)

    # basic inputs
    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Days)", value=45, min_value=1)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    # Calculate initial option prices
    initial_bs = BlackScholes(
        time_to_maturity=time_to_maturity,
        strike=strike,
        current_price=current_price,
        volatility=volatility,
        interest_rate=interest_rate
    )
    initial_call, initial_put = initial_bs.calculate_prices()[:2]

    # position sizing inputs
    st.markdown("---")
    st.subheader("Position Sizing")

    # Account size input
    account_size = st.number_input(
        "Account Size ($)",
        value=10000.0,
        min_value=100.0,
        help="Enter your total account size for position sizing calculations"
    )

    max_premium_allocation = vix_dynamic_allocation(balance=account_size)

    try:
        current_vix = si.get_live_price('^VIX').round(2)
        st.info(f"Current VIX: {current_vix:.2f}")
    except Exception as e:
        st.warning("Unable to fetch current VIX level")
    
    st.info(f"Maximum premium allocation based on VIX: ${max_premium_allocation:.2f} "
            f"({(max_premium_allocation/account_size)*100:.1f}% of account)")

    # Kelly Criterion inputs
    col1 = st.columns(1)[0]
    with col1:
        prob_profit = st.slider(
            "Probability of Profit (%)",
            min_value=0.00,
            max_value=1.00,
            value=0.70,
            help="Estimated probability of profit for the trade"
        )

    kelly_fraction = kelly_criterion_allocation(interest_rate, time_to_maturity, prob_profit)
    kelly_allocation = min(kelly_fraction * account_size, max_premium_allocation)

    st.info(f"Kelly Criterion suggested allocation: ${kelly_allocation:.2f} ({(kelly_allocation/account_size)*100:.1f}% of account)")

    # Calculate recommended number of contracts based on current option price
    current_option_value = min(initial_call, initial_put)  # Use the cheaper of call or put
    if current_option_value > 0:
        max_contracts = int(kelly_allocation / (current_option_value * 100))  # Multiply by 100 since each contract represents 100 shares
    else:
        max_contracts = 0
        
    # Simple position sizing input
    position_units = st.number_input(
        "Number of Contracts/Units",
        value=1,
        min_value=1,
        max_value=1000,
        help="Enter the number of option contracts to analyze"
    )

    # visualization type selector
    st.markdown("---")
    st.subheader("Visualization Settings")
    viz_type = st.selectbox(
        "Visualization Type",
        ["Option Prices", "Position P&L"],
        help="Choose whether to display option prices or position P&L"
    )

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.0001)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.0001)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.0005)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.0005)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)



def plot_heatmap(bs_model, spot_range, vol_range, strike, position_units, viz_type):
    """
    Generate heatmaps for either option prices or P&L analysis
    """
    call_matrix = np.zeros((len(vol_range), len(spot_range)))
    put_matrix = np.zeros((len(vol_range), len(spot_range)))

    # Get initial option prices for P&L calculation
    initial_bs = BlackScholes(
        time_to_maturity=bs_model.time_to_maturity,
        strike=strike,
        current_price=bs_model.current_price,
        volatility=bs_model.volatility,
        interest_rate=bs_model.interest_rate
    )
    initial_call, initial_put = initial_bs.calculate_prices()[:2]

 # Calculate matrices for both options
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            call_price, put_price = bs_temp.calculate_prices()[:2]
            
            if viz_type == "Position P&L":
                # Calculate P&L for both options
                call_matrix[i, j] = (call_price - initial_call) * position_units
                put_matrix[i, j] = (put_price - initial_put) * position_units
            else:
                # Store raw prices
                call_matrix[i, j] = call_price
                put_matrix[i, j] = put_price
    
    # Create figures with appropriate formatting
    fmt = ".2f" if viz_type == "Position P&L" else ".4f"
    title_prefix = "P&L" if viz_type == "Position P&L" else "Price"
    cmap = "RdYlGn" if viz_type == "Position P&L" else "viridis"
    
    # Call option heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        call_matrix,
        xticklabels=np.round(spot_range, 4),
        yticklabels=np.round(vol_range, 4),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax_call
    )
    ax_call.set_title(f'CALL {title_prefix} Analysis')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Put option heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        put_matrix,
        xticklabels=np.round(spot_range, 4),
        yticklabels=np.round(vol_range, 4),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax_put
    )
    ax_put.set_title(f'PUT {title_prefix} Analysis')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put


# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Days)": [time_to_maturity], # Display in days
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
    "Position Size (Units)": [position_units],
}

input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price, call_delta, put_delta = bs_model.calculate_prices()

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.4f}</div>
                <div class="metric-label">CALL Delta</div>
                <div class="metric-value">{call_delta:.4f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.4f}</div>
                <div class="metric-label">PUT Delta</div>
                <div class="metric-value">{put_delta:.4f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title(f"Options {viz_type} Interactive Heatmap")
st.info(f"Explore how option {viz_type.lower()} fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

heatmap_fig_call, heatmap_fig_put = plot_heatmap(
    bs_model,
    spot_range,
    vol_range,
    strike,
    position_units,
    viz_type
)

with col1:
    st.subheader("Call Option Analysis")
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Option Analysis")
    st.pyplot(heatmap_fig_put)
