import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to get option prices
@st.cache_data
def get_option_prices(ticker, expiration_date):
    stock = yf.Ticker(ticker)
    options_chain = stock.option_chain(expiration_date)

    # Get call option data
    calls = options_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume']]
    calls.columns = ['Strike Price (K)', 'Option Price (C)', 'Bid', 'Ask', 'Volume']
    
    # Get put option data
    puts = options_chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume']]
    puts.columns = ['Strike Price (K)', 'Option Price (P)', 'Bid', 'Ask', 'Volume']
    
    return calls, puts

# Function to filter options within 50% of stock price and limit option price to 30% of stock price
def filter_options(df, stock_price):
    lower_bound = stock_price * 0.5
    upper_bound = stock_price * 1.5
    option_price_limit = stock_price * 0.3  # Limit option prices to 30% of the stock price
    if 'Option Price (C)' in df.columns:
        return df[(df['Strike Price (K)'] >= lower_bound) & 
                  (df['Strike Price (K)'] <= upper_bound) & 
                  (df['Option Price (C)'] <= option_price_limit)]
    else:
        return df[(df['Strike Price (K)'] >= lower_bound) & 
                  (df['Strike Price (K)'] <= upper_bound) & 
                  (df['Option Price (P)'] <= option_price_limit)]

# Function to calculate derivative with respect to strike price
def calculate_derivative_strike(df, option_type):
    derivatives = []
    
    for i in range(len(df) - 1):
        C_current = df['Option Price (C)'].iloc[i] if option_type == 'call' else df['Option Price (P)'].iloc[i]
        C_next = df['Option Price (C)'].iloc[i + 1] if option_type == 'call' else df['Option Price (P)'].iloc[i + 1]
        K_current = df['Strike Price (K)'].iloc[i]
        K_next = df['Strike Price (K)'].iloc[i + 1]
        
        derivative = (C_next - C_current) / (K_next - K_current)
        derivatives.append(derivative)

    df['dC/dK' if option_type == 'call' else 'dP/dK'] = [np.nan] + derivatives
    return df

# Function to highlight ITM calls
def highlight_itm_calls(df, stock_price):
    # In-the-money calls have strike price less than the current stock price
    return df.style.applymap(lambda x: 'background-color: lightgrey' if x < stock_price else '', subset=['Strike Price (K)'])

# Function to highlight ITM puts
def highlight_itm_puts(df, stock_price):
    # In-the-money puts have strike price greater than the current stock price
    return df.style.applymap(lambda x: 'background-color: lightgrey' if x > stock_price else '', subset=['Strike Price (K)'])

# Minimalist plot for derivatives with respect to strike price, limiting y-axis to -2 to +2
def plot_real_derivatives_minimalist(calls_df, puts_df, stock_price):
    plt.figure(figsize=(10, 6))

    # Drop NaN values for smooth plotting
    calls_df = calls_df.dropna(subset=['dC/dK'])
    puts_df = puts_df.dropna(subset=['dP/dK'])

    # Minimalist style
    plt.style.use('ggplot')

    # Plot call derivatives
    plt.scatter(calls_df['Strike Price (K)'], calls_df['dC/dK'], color='#FF9999', marker='o', label='Calls (dC/dK)', alpha=0.7)
    
    # Plot put derivatives
    plt.scatter(puts_df['Strike Price (K)'], puts_df['dP/dK'], color='#99CCFF', marker='o', label='Puts (dP/dK)', alpha=0.7)

    # Line for current stock price
    plt.axvline(x=stock_price, color='#99CC99', linestyle='--', label=f'Current Price: {stock_price:.2f}', lw=2)

    # Limit y-axis to -2 and +2
    plt.ylim(-2, 2)

    # Minimalist axis settings
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Clean background
    ax.set_facecolor('white')
    
    # Minimalist labels and title
    plt.title('Option Derivatives vs Strike Price', fontsize=14, fontweight='light')
    plt.xlabel('Strike Price', fontsize=12)
    plt.ylabel('Derivative (dC/dK and dP/dK)', fontsize=12)
    
    # Soft grid
    plt.grid(True, color='gray', alpha=0.3)
    
    # Legend
    plt.legend(loc='best', frameon=False)

    # Show the plot in Streamlit
    st.pyplot(plt)
    plt.close()

# Minimalist plot for derivatives with respect to strike price (second plot)
def plot_option_derivatives_minimalist(calls_df, puts_df, stock_price):
    plt.figure(figsize=(10, 6))

    # Drop NaN values for smooth plotting
    calls_df = calls_df.dropna(subset=['dC/dK'])
    puts_df = puts_df.dropna(subset=['dP/dK'])

    # Minimalist style
    plt.style.use('ggplot')

    # Plot call derivatives
    plt.scatter(calls_df['Strike Price (K)'], calls_df['dC/dK'], color='#FF9999', marker='o', label='Calls (dC/dK)', alpha=0.7)
    
    # Plot put derivatives
    plt.scatter(puts_df['Strike Price (K)'], puts_df['dP/dK'], color='#99CCFF', marker='o', label='Puts (dP/dK)', alpha=0.7)

    # Line for current stock price
    plt.axvline(x=stock_price, color='#99CC99', linestyle='--', label=f'Current Price: {stock_price:.2f}', lw=2)

    # Limit y-axis to -2 and +2
    plt.ylim(-2, 2)

    # Minimalist axis settings
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Clean background
    ax.set_facecolor('white')

    # Minimalist labels and title
    plt.title('Option Derivatives vs Strike Price (Second Plot)', fontsize=14, fontweight='light')
    plt.xlabel('Strike Price', fontsize=12)
    plt.ylabel('Derivative (dC/dK and dP/dK)', fontsize=12)

    # Soft grid
    plt.grid(True, color='gray', alpha=0.3)
    
    # Legend
    plt.legend(loc='best', frameon=False)

    # Show the plot in Streamlit
    st.pyplot(plt)
    plt.close()

# Streamlit app
st.title("Options Analysis")
ticker = st.text_input("Enter the stock ticker (e.g., AAPL):")

if ticker:
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options
    expiration_date = st.selectbox("Select the expiration date:", expiration_dates)
    
    if st.button("Get Options and Generate Plots"):
        try:
            # Get current stock price
            stock_price = stock.history(period="1d")['Close'].iloc[0]

            # Get option prices
            calls_df, puts_df = get_option_prices(ticker, expiration_date)

            # Filter options to be within 50% above and below current stock price, and option price <= 30% of stock price
            calls_df = filter_options(calls_df, stock_price)
            puts_df = filter_options(puts_df, stock_price)

            # Calculate derivatives
            calls_df = calculate_derivative_strike(calls_df, 'call')
            puts_df = calculate_derivative_strike(puts_df, 'put')

            # Create tabs for call and put options
            tab1, tab2 = st.tabs(["Call Options", "Put Options"])

            # Display call options in tab1
            with tab1:
                st.subheader("Call Option Prices")
                styled_calls_df = highlight_itm_calls(calls_df, stock_price)
                st.table(styled_calls_df)

            # Display put options in tab2
            with tab2:
                st.subheader("Put Option Prices")
                styled_puts_df = highlight_itm_puts(puts_df, stock_price)
                st.table(styled_puts_df)

            # Plot derivatives vs strike price
            plot_real_derivatives_minimalist(calls_df, puts_df, stock_price)

            # Plot derivatives vs strike price in second plot
            plot_option_derivatives_minimalist(calls_df, puts_df, stock_price)

        except Exception as e:
            st.error(f"Error: {e}")
