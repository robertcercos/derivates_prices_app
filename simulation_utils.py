import yfinance as yf
import streamlit as st
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


def simulate_random_walks(t_params=None, empirical_returns=None, technique="t-student", days=252, sims=10000):
    if technique == "t-student":
        # Simulate using T-Student distribution
        random_returns = stats.t.rvs(*t_params, size=(sims, days))
    elif technique == "bootstrap":
        # Simulate using bootstrapping from empirical distribution
        random_returns = np.random.choice(empirical_returns, size=(sims, days), replace=True)
    else:
        raise ValueError("Invalid technique. Choose 't-student' or 'bootstrap'.")
    
    random_returns = np.clip(random_returns, -0.3, 0.3)
    # Convert returns to cumulative product for random walks
    random_walks = np.cumprod(1 + random_returns, axis=1)
    return random_walks


def plot_random_walks(random_walks, initial_price, ticker, sim_start_date, sim_end_date, technique):
    # Colors
    if technique == "t-student":
        color_main = '#00B3B3'
        color_95 = '#3498db'
        color_5 = '#e74c3c'
        label_prefix = "T-Student"
    elif technique == "bootstrap":
        color_main = '#FF6347'
        color_95 = '#FF8C00'
        color_5 = '#FF4500'
        label_prefix = "Bootstrap"
    
    # Calculate key statistics
    path_median = np.quantile(random_walks, 0.5, axis=0)
    path_95 = np.quantile(random_walks, 0.95, axis=0)
    path_5 = np.quantile(random_walks, 0.05, axis=0)
    
    # Calculate ending prices and percentiles for filtering
    ending_prices = random_walks[:, -1] * initial_price
    p99 = np.percentile(ending_prices, 99)
    p1 = np.percentile(ending_prices, 1)
    
   # Filter out paths with any value outside [p1, p99] during the whole path
    valid_paths = np.all((random_walks * initial_price >= p1) & (random_walks * initial_price <= p99), axis=1)
    filtered_random_walks = random_walks[valid_paths]
    
    # Final prices for each path
    final_price_median = path_median[-1] * initial_price
    final_price_95 = path_95[-1] * initial_price
    final_price_5 = path_5[-1] * initial_price
    
    # Create figure and gridspec
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=(3, 1))
    ax = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # Spaghetti plot: Select 1000 random paths to plot in light gray
    selected_paths = np.random.choice(filtered_random_walks.shape[0], 1000, replace=False)
    for i in selected_paths:
        ax.plot(filtered_random_walks[i] * initial_price, color='lightgray', linewidth=0.5, alpha=0.3)


    # Plot the simulated paths and percentiles
    ax.plot(path_median * initial_price, label=f'Median {label_prefix} (Final: {final_price_median:.2f})', color=color_main)
    ax.plot(path_95 * initial_price, label=f'$95^{{th}}$ Percentile {label_prefix} (Final: {final_price_95:.2f})', color=color_95)
    ax.plot(path_5 * initial_price, label=f'$5^{{th}}$ Percentile {label_prefix} (Final: {final_price_5:.2f})', color=color_5)
    ax.fill_between(np.arange(random_walks.shape[1]), y1=path_5 * initial_price, y2=path_95 * initial_price, color=color_main, alpha=0.2)
    
    
    # Label and style
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    
    # Minimalistic grid and lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(visible=True, linestyle='--', alpha=0.5)

    # Display the stock ticker and period in the bottom-left corner
    ax.text(0.02, -0.1, f"{ticker.upper()} | {sim_start_date} - {sim_end_date}", transform=ax.transAxes, 
            fontsize=10, color='#7f8c8d', ha='left', va='center', fontstyle='italic')

    # Display the initial price
    ax.text(0.98, 1.05, f"Initial Price: {initial_price:.2f}", transform=ax.transAxes, fontsize=12, ha='right', va='top')

    # Calculate and plot compounded growth rates at the end of the simulation
    ending_returns = (random_walks[:, -1] - 1) * 100
    p99 = np.percentile(ending_returns, 99)
    p1 = np.percentile(ending_returns, 1)
    p95 = np.percentile(ending_returns, 95)
    p5 = np.percentile(ending_returns, 5)
    
     # Plot histogram and add percentiles
    ax_hist.hist(ending_returns, orientation='horizontal', bins=40, color=color_main, alpha=0.3, range=(p1, p99))

    # Spaghetti plot: Plot a subset of individual ending returns in light gray
    #ax_hist.scatter(ending_returns[:200], np.arange(200), color='lightgray', alpha=0.3)

    ax_hist.axhline(np.median(ending_returns), label=f'Median {label_prefix} ({np.median(ending_returns):.2f}%)', color=color_main)
    ax_hist.axhline(p95, label=f'95th Percentile {label_prefix} ({p95:.2f}%)', color=color_95)
    ax_hist.axhline(p5, label=f'5th Percentile {label_prefix} ({p5:.2f}%)', color=color_5)

    ax_hist.set_ylabel('Compound Growth Rate (%)')
    ax_hist.set_xlabel('Frequency')
    ax_hist.legend()

    plt.tight_layout()
    plt.show()
    
