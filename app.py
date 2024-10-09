import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime
import numpy as np

# Función para obtener los precios de las opciones
def get_option_prices(ticker, expiration_date):
    stock = yf.Ticker(ticker)
    options_chain = stock.option_chain(expiration_date)

    # Obtener datos de calls
    calls = options_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume']]
    calls.columns = ['Strike Price (K)', 'Option Price (C)', 'Bid', 'Ask', 'Volume']
    
    # Obtener datos de puts
    puts = options_chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume']]
    puts.columns = ['Strike Price (K)', 'Option Price (P)', 'Bid', 'Ask', 'Volume']
    

    
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

    df['dC/dK' if option_type == 'call' else 'dP/dK'] = [np.nan] + derivatives
    return df

def plot_interpolated_and_real_derivatives(calls_df, puts_df, stock_price):
    plt.figure(figsize=(10, 6))

    # Eliminar los valores nulos o None de las derivadas para la interpolación
    calls_df = calls_df.dropna(subset=['dC/dK'])
    puts_df = puts_df.dropna(subset=['dP/dK'])

    # Interpolación para calls (dC/dK)
    f_calls = interp1d(calls_df['Strike Price (K)'], calls_df['dC/dK'], kind='cubic', fill_value="extrapolate")
    strike_range_calls = np.linspace(calls_df['Strike Price (K)'].min(), calls_df['Strike Price (K)'].max(), 500)
    dC_dK_smooth = f_calls(strike_range_calls)

    # Interpolación para puts (dP/dK)
    f_puts = interp1d(puts_df['Strike Price (K)'], puts_df['dP/dK'], kind='cubic', fill_value="extrapolate")
    strike_range_puts = np.linspace(puts_df['Strike Price (K)'].min(), puts_df['Strike Price (K)'].max(), 500)
    dP_dK_smooth = f_puts(strike_range_puts)

    # Graficar los puntos reales de las derivadas de los calls
    plt.scatter(calls_df['Strike Price (K)'], calls_df['dC/dK'], color='red', marker='o', label='Puntos Reales dC/dK (Calls)')
    
    # Graficar la derivada interpolada de las calls
    plt.plot(strike_range_calls, dC_dK_smooth, color='red', linestyle='--', label='dC/dK (Calls - Interpolado)')
    
    # Graficar los puntos reales de las derivadas de los puts
    plt.scatter(puts_df['Strike Price (K)'], puts_df['dP/dK'], color='blue', marker='o', label='Puntos Reales dP/dK (Puts)')
    
    # Graficar la derivada interpolada de las puts
    plt.plot(strike_range_puts, dP_dK_smooth, color='blue', linestyle='--', label='dP/dK (Puts - Interpolado)')
    
    # Línea vertical que indica el precio actual de la acción
    plt.axvline(x=stock_price, color='green', linestyle='--', label=f'Precio Actual ({stock_price})')
    
    # Títulos y leyenda
    plt.title('Derivada Interpolada y Puntos Reales de los Precios de Opciones respecto al Strike Price')
    plt.xlabel('Precio de Ejercicio (Strike Price)')
    plt.ylabel('dC/dK y dP/dK (Interpolado y Real)')
    plt.axhline(0, color='black', lw=0.5)  # Línea horizontal en y=0
    plt.grid()
    plt.legend()

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)


# Streamlit app
st.title("Opciones y Derivadas")
ticker = st.text_input("Ingrese el ticker de la acción (ej. AAPL):")

if ticker:
    expiration_date = st.date_input("Seleccione la fecha de vencimiento de las opciones:")
    
    # Formatear la fecha en el formato YYYY-MM-DD
    formatted_date = expiration_date.strftime("%Y-%m-%d")
    st.write(f"Fecha seleccionada: {formatted_date}")
    
    if st.button("Obtener opciones y gráfico interpolado con puntos reales"):
        try:
            # Obtener el precio actual de la acción
            stock = yf.Ticker(ticker)
            stock_price = stock.history(period="1d")['Close'].iloc[0]  # Último precio de cierre

            # Obtener los precios de opciones
            calls_df, puts_df = get_option_prices(ticker, formatted_date)
            calls_df = calculate_derivative(calls_df, 'call')
            puts_df = calculate_derivative(puts_df, 'put')

            st.subheader("Precios de Opciones de Compra (Calls)")
            st.table(calls_df)  # Mostrar la tabla completa sin scroll

            st.subheader("Precios de Opciones de Venta (Puts)")
            st.table(puts_df)  # Mostrar la tabla completa sin scroll

            # Graficar la derivada interpolada con los puntos reales respecto al Strike Price
            plot_interpolated_and_real_derivatives(calls_df, puts_df, stock_price)

    except yf.YFException as yf_error:
        st.error(f"Error al obtener datos de Yahoo Finance: {yf_error}")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
