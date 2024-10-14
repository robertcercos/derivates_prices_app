import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import timedelta
import simulation_utils as sim_utils
import option_data_utils as opt_utils 

def calculate_strike_probabilities(strike_prices, random_walks, initial_price):
    probabilities = []
    # Loop over each strike price
    for strike in strike_prices:
        # Count how many paths in the random walks reach or exceed the strike price at any point
        hits = np.any(random_walks * initial_price >= strike, axis=1)
        prob = np.mean(hits)  # Probability is the proportion of paths that reach the strike
        probabilities.append(prob)
    
    return probabilities

# Function to display options data and calculate probabilities for T-Student and Bootstrapping
def add_probability_columns(calls, puts, random_walks_t_student, random_walks_bootstrap, initial_price):
    # Calculate probabilities for call options
    calls['ITM T-Student'] = (calculate_strike_probabilities(calls['Strike Price (K)'], random_walks_t_student, initial_price)) * 100
    calls['ITM T-Student'] = calls['ITM T-Student'].round(1)
    
    calls['ITM Bootstrapping'] = (calculate_strike_probabilities(calls['Strike Price (K)'], random_walks_bootstrap, initial_price)) * 100
    calls['ITM Bootstrapping'] = calls['ITM Bootstrapping'].round(1)
    
    # Calculate probabilities for put options
    puts['ITM T-Student'] = (1 - calculate_strike_probabilities(puts['Strike Price (K)'], random_walks_t_student, initial_price)) * 100
    puts['ITM T-Student'] = puts['ITM T-Student'].round(1)
    
    puts['ITM Bootstrapping'] = (1 - calculate_strike_probabilities(puts['Strike Price (K)'], random_walks_bootstrap, initial_price)) * 100
    puts['ITM Bootstrapping'] = puts['ITM Bootstrapping'].round(1)

    return calls, puts



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
                df = sim_utils.get_historical_data(ticker)
                stock_price = stock.history(period="1d")['Close'].iloc[0]
                initial_price = df['Close'].iloc[-1]
                sim_start_date = pd.to_datetime("today").strftime('%Y-%m-%d')
                days_to_expiration = 30 #(pd.to_datetime(expiration_date).normalize() - pd.to_datetime("today").normalize() ).days
                sim_end_date = (pd.to_datetime("today") + timedelta(days=days_to_expiration)).strftime('%Y-%m-%d')
    
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


                # Generate random walks with T-Student
                t_params = sim_utils.fit_t_distribution(df)
                random_walks_t_student = sim_utils.simulate_random_walks(t_params=t_params, technique="t-student", days=days_to_expiration)

                # Generate random walks with Bootstrapping
                random_walks_bootstrap = sim_utils.simulate_random_walks(empirical_returns=df['Daily Return'].dropna().values, technique="bootstrap", days=days_to_expiration)
    
                # Calculate ITM probabilities and add to the DataFrame
                calls_df['ITM T-Student'] = calculate_strike_probabilities(calls_df['Strike Price (K)'], random_walks_t_student, stock_price)
                calls_df['ITM Bootstrapping'] = calculate_strike_probabilities(calls_df['Strike Price (K)'], random_walks_bootstrap, stock_price)
                puts_df['ITM T-Student'] = calculate_strike_probabilities(puts_df['Strike Price (K)'], random_walks_t_student, stock_price)
                puts_df['ITM Bootstrapping'] = calculate_strike_probabilities(puts_df['Strike Price (K)'], random_walks_bootstrap, stock_price)
    
               
    
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

                st.subheader(f"Fitted T-Student Distribution for Daily Returns of {ticker}")
                sim_utils.plot_fitted_distribution(df, t_params, ticker)  # Display the histogram and fitted curve


                # Plot the random walks for both T-Student and Bootstrapping
                st.subheader("Random Walks with T-Student Distribution")
                sim_utils.plot_random_walks(random_walks_t_student, initial_price, ticker, sim_start_date, sim_end_date, "t-student")
    
                st.subheader("Random Walks with Bootstrapping")
                sim_utils.plot_random_walks(random_walks_bootstrap, initial_price, ticker, sim_start_date, sim_end_date, "bootstrap")
    
    
            except Exception as e:
                st.error(f"Error: {e}")


# Execute the app
if __name__ == "__main__":
    run_app()
