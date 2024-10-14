import streamlit as st
from datetime import timedelta
#import simulation_utils as sim_utils
import option_data_utils as opt_utils

#from simulation_utils import *
#from option_data_utils import *

# Function to get option prices
@st.cache_data

def run_app():
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
                calls_df, puts_df = opt_utils.get_option_prices(ticker, expiration_date)
    
                # Filter options to be within 50% above and below current stock price, and option price <= 30% of stock price
                calls_df = opt_utils.filter_options(calls_df, stock_price)
                puts_df = opt_utils.filter_options(puts_df, stock_price)
    
                # Calculate derivatives
                calls_df = opt_utils.calculate_derivative_strike(calls_df, 'call')
                puts_df = opt_utils.calculate_derivative_strike(puts_df, 'put')
    
                # Convert to in-the-money probabilities
                calls_df = opt_utils.calculate_in_the_money_prob(calls_df, 'call')
                puts_df = opt_utils.calculate_in_the_money_prob(puts_df, 'put')
    
                # Format the dataframes
                calls_df = opt_utils.format_dataframe(calls_df, 'call')
                puts_df = opt_utils.format_dataframe(puts_df, 'put')
    
                # Style and center the "in-the-money prob" column, apply 2 decimal format
                calls_df_styled = opt_utils.style_and_center(calls_df, stock_price, 'call')
                puts_df_styled = opt_utils.style_and_center(puts_df, stock_price, 'put')
    
                # Create tabs for call and put options
                tab1, tab2 = st.tabs(["Call Options", "Put Options"])
    
                # Display call options in tab1
                with tab1:
                    st.subheader("Call Option Prices")
                    st.table(calls_df_styled)
    
                # Display put options in tab2
                with tab2:
                    st.subheader("Put Option Prices")
                    st.table(puts_df_styled)
    
                # Plot derivatives vs strike price
                opt_utils.plot_real_derivatives_minimalist(calls_df, puts_df, stock_price)
    
                # Plot derivatives vs option price
                opt_utils.plot_option_derivatives_minimalist(calls_df, puts_df, stock_price)
    
            except Exception as e:
                st.error(f"Error: {e}")


# Execute the app
if __name__ == "__main__":
    run_app()
