import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Function to filter options within 50% of stock price, limit option price to 30% of stock price, and remove low volume rows
def filter_options(df, stock_price):
    lower_bound = stock_price * 0.5
    upper_bound = stock_price * 1.5
    option_price_limit = stock_price * 0.3  # Limit option prices to 30% of the stock price
    
    # Filter options within 50% of stock price and option price <= 30% of stock price
    if 'Option Price (C)' in df.columns:
        df = df[(df['Strike Price (K)'] >= lower_bound) & 
                (df['Strike Price (K)'] <= upper_bound) & 
                (df['Option Price (C)'] <= option_price_limit)]
    else:
        df = df[(df['Strike Price (K)'] >= lower_bound) & 
                (df['Strike Price (K)'] <= upper_bound) & 
                (df['Option Price (P)'] <= option_price_limit)]

    # Remove rows where volume is NaN or less than 5
    df = df.dropna(subset=['Volume'])
    df = df[df['Volume'] >= 5]
    
    return df

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

# Convert dC/dK or dP/dK to in-the-money probability between 0% and 100%, replace NaN with "-"
def calculate_in_the_money_prob(df, option_type):
    if option_type == 'call':
        df['in-the-money prob'] = (-df['dC/dK'] * 100).clip(0, 100).round(1).astype(str) + '%'
    else:
        df['in-the-money prob'] = (df['dP/dK'] * 100).clip(0, 100).round(1).astype(str) + '%'

    # Replace NaN values with "-"
    df['in-the-money prob'] = df['in-the-money prob'].replace('nan%', '-')
    return df

# Function to format the dataframe to limit decimals
def format_dataframe(df, option_type):
    # Round decimals for option prices, bid, ask, and probabilities
    if option_type == 'call':
        df['Option Price (C)'] = df['Option Price (C)'].round(2)
        df['Bid'] = df['Bid'].round(2)
        df['Ask'] = df['Ask'].round(2)
    else:
        df['Option Price (P)'] = df['Option Price (P)'].round(2)
        df['Bid'] = df['Bid'].round(2)
        df['Ask'] = df['Ask'].round(2)
    
    # Convert the rest of the columns to integers
    df['Strike Price (K)'] = df['Strike Price (K)'].astype(int)
    df['Volume'] = df['Volume'].astype(int)
    
    return df

# Function to style and center the "in-the-money prob" column
def style_and_center(df, stock_price, option_type):
    # Style and center values in the "in-the-money prob" column
    styler = df.style.set_properties(subset=['in-the-money prob'], **{'text-align': 'center'})
    
    # Highlight ITM rows
    if option_type == 'call':
        styler = styler.applymap(lambda x: 'background-color: lightgrey' if x < stock_price else '', subset=['Strike Price (K)'])
    else:
        styler = styler.applymap(lambda x: 'background-color: lightgrey' if x > stock_price else '', subset=['Strike Price (K)'])
    
    # Format all other columns to 2 decimal places
    styler = styler.format(precision=2)
    
    return styler

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

# Minimalist plot for derivatives with respect to option price, limiting y-axis to -2 to +2
def plot_option_derivatives_minimalist(calls_df, puts_df, stock_price):
    plt.figure(figsize=(10, 6))

    # Drop NaN values for smooth plotting
    calls_df = calls_df.dropna(subset=['dC/dK'])
    puts_df = puts_df.dropna(subset=['dP/dK'])

    # Minimalist style
    plt.style.use('ggplot')

    # Plot call derivatives with respect to option price
    plt.scatter(calls_df['Option Price (C)'], calls_df['dC/dK'], color='#FF9999', marker='o', label='Calls (dC/dK)', alpha=0.7)
    
    # Plot put derivatives with respect to option price
    plt.scatter(puts_df['Option Price (P)'], puts_df['dP/dK'], color='#99CCFF', marker='o', label='Puts (dP/dK)', alpha=0.7)

    # Limit y-axis to -2 and +2
    plt.ylim(-2, 2)

    # Minimalist axis settings
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Clean background
    ax.set_facecolor('white')

    # Minimalist labels and title
    plt.title('Option Derivatives vs Option Price', fontsize=14, fontweight='light')
    plt.xlabel('Option Price', fontsize=12)
    plt.ylabel('Derivative (dC/dK and dP/dK)', fontsize=12)

    # Soft grid
    plt.grid(True, color='gray', alpha=0.3)
    
    # Legend
    plt.legend(loc='best', frameon=False)

    # Show the plot in Streamlit
    st.pyplot(plt)
    plt.close()
