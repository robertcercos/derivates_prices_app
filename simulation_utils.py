import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import timedelta

# Function to get historical stock data and calculate daily returns
@st.cache_data
def get_historical_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")  # Get max available historical data
    df['Daily Return'] = df['Close'].pct_change()  # Calculate daily percentage change
    df.dropna(inplace=True)  # Drop missing values
    return df

# Function to fit a T-distribution to the daily returns
def fit_t_distribution(df):
    returns = df['Daily Return'].values
    params = stats.t.fit(returns)  # Fit T-Student distribution
    return params

def format_value_as_percentage(value):
    return f"+{value * 100:.2f}%" if value >= 0 else f"{value * 100:.2f}%"

# Function to plot the histogram and T-distribution with percentiles
def plot_fitted_distribution(df, t_params, ticker):
    plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter the data to show only returns within [-20%, 20%]
    filtered_df = df[(df['Daily Return'] >= -0.2) & (df['Daily Return'] <= 0.2)]
    
    # Plot histogram of filtered data
    ax.hist(filtered_df['Daily Return'], bins=100, density=True, alpha=0.5, color='#1abc9c', edgecolor='none')
    
    # Generate points for T-distribution
    x = np.linspace(-0.2, 0.2, 1000)  # Limit to [-20%, 20%]
    pdf_t = stats.t.pdf(x, *t_params)
    
    # Plot T-distribution fit
    ax.plot(x, pdf_t, color='#e74c3c', lw=2, alpha=0.8)
    
    # Calculate percentiles for T-distribution
    t_percentiles = [stats.t.ppf(p, *t_params) for p in [0.05, 0.5, 0.95]]
    
    # Mark percentiles on the plot for T-Distribution and add the value to the label
    for p, value, v_offset, h_align in zip([5, 50, 95], t_percentiles, [0.1, 0.35, 0.1], ['right', 'center', 'left']):
        if -0.2 <= value <= 0.2:  # Only show percentiles within the range
            ax.axvline(value, color='#34495e', linestyle='--', lw=1, ymin=0, ymax=0.85, alpha=0.5)  # Increased transparency
            ax.text(value, max(pdf_t) * v_offset, f"p{p}\n({format_value_as_percentage(value)})", 
                    color='black', ha=h_align, va='top' if p == 50 else 'bottom')  # Black labels for percentiles
    
    # Add the stock ticker and period of data in the top right corner
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    ax.text(0.98, 1.05, f"{ticker.upper()} | {start_date} - {end_date}",
            transform=ax.transAxes, fontsize=10, ha='right', va='top', color='black', 
            fontweight='light', alpha=0.7)

    # Remove unnecessary chart elements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Format X-axis as percentage
    ax.set_xticks(np.linspace(-0.2, 0.2, 9))
    ax.set_xticklabels([f"{x * 100:.0f}%" for x in ax.get_xticks()])  # Show X-axis labels as percentages
    
    # Remove the Y-axis label and ticks
    ax.set_yticks([])
    ax.set_ylabel("")
    
    # Set X-axis label
    ax.set_xlabel("Daily Return", fontsize=12)
    ax.tick_params(axis='x', which='both', bottom=True, labelsize=10)
    
