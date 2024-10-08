import streamlit as st
import yfinance as yf
import pandas as pd

# Función para obtener los precios de las opciones
def get_option_prices(ticker, expiration_date):
    stock = yf.Ticker(ticker)
    options_chain = stock.option_chain(expiration_date)
    
    calls = options_chain.calls[['strike', 'lastPrice']]
    calls.columns = ['Strike Price (K)', 'Option Price (C)']
    
    puts = options_chain.puts[['strike', 'lastPrice']]
    puts.columns = ['Strike Price (K)', 'Option Price (P)']
    
    return calls, puts

# Calcular la derivada
def calculate_derivative(df, option_type):
    derivatives = []
    
    for i in range(len(df) - 1):
        C_current = df['Option Price (C)'].iloc[i] if option_type == 'call' else df['Option Price (P)'].iloc[i]
        C_next = df['Option Price (C)'].iloc[i + 1] if option_type == 'call' else df['Option Price (P)'].iloc[i + 1]
        K_current = df['Strike Price (K)'].iloc[i]
        K_next = df['Strike Price (K)'].iloc[i + 1]
        
        derivative = (C_next - C_current) / (K_next - K_current)
        derivatives.append(derivative)
    
    df['dC/dK' if option_type == 'call' else 'dP/dK'] = [None] + derivatives
    return df

# Streamlit app
st.title("Opciones y Derivadas")
ticker = st.text_input("Ingrese el ticker de la acción (ej. AAPL):")

if ticker:
    expiration_date = st.date_input("Seleccione la fecha de vencimiento de las opciones:")
    
    if st.button("Obtener opciones"):
        try:
            calls_df, puts_df = get_option_prices(ticker, expiration_date)
            calls_df = calculate_derivative(calls_df, 'call')
            puts_df = calculate_derivative(puts_df, 'put')

            st.subheader("Precios de Opciones de Compra (Calls)")
            st.write(calls_df)

            st.subheader("Precios de Opciones de Venta (Puts)")
            st.write(puts_df)
        except Exception as e:
            st.error(f"Error: {e}")
