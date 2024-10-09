import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Función para obtener los precios de las opciones
@st.cache_data
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

# Calcular la derivada respecto al precio de ejercicio (strike price)
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

# Calcular la derivada respecto al precio de la opción
def calculate_derivative_option(df, option_type):
    derivatives = []
    
    for i in range(len(df) - 1):
        K_current = df['Strike Price (K)'].iloc[i]
        K_next = df['Strike Price (K)'].iloc[i + 1]
        C_current = df['Option Price (C)'].iloc[i] if option_type == 'call' else df['Option Price (P)'].iloc[i]
        C_next = df['Option Price (C)'].iloc[i + 1] if option_type == 'call' else df['Option Price (P)'].iloc[i + 1]
        
        derivative = (K_next - K_current) / (C_next - C_current)
        derivatives.append(derivative)

    df['dK/dC' if option_type == 'call' else 'dK/dP'] = [np.nan] + derivatives
    return df

# Función para aplicar el color gris a las filas ITM
def highlight_itm_calls(df, stock_price):
    return ['background-color: lightgrey' if row['Strike Price (K)'] < stock_price else '' for idx, row in df.iterrows()]

def highlight_itm_puts(df, stock_price):
    return ['background-color: lightgrey' if row['Strike Price (K)'] > stock_price else '' for idx, row in df.iterrows()]

# Gráfico minimalista
def plot_real_derivatives_minimalist(calls_df, puts_df, stock_price):
    plt.figure(figsize=(10, 6))

    # Eliminar los valores nulos o None de las derivadas para mostrar solo los valores reales
    calls_df = calls_df.dropna(subset=['dC/dK'])
    puts_df = puts_df.dropna(subset=['dP/dK'])

    # Estilo minimalista
    plt.style.use('seaborn-whitegrid')

    # Graficar los puntos reales de las derivadas de los calls
    plt.scatter(calls_df['Strike Price (K)'], calls_df['dC/dK'], color='#FF9999', marker='o', label='Calls (dC/dK)', alpha=0.7)
    
    # Graficar los puntos reales de las derivadas de los puts
    plt.scatter(puts_df['Strike Price (K)'], puts_df['dP/dK'], color='#99CCFF', marker='o', label='Puts (dP/dK)', alpha=0.7)

    # Línea vertical que indica el precio actual de la acción
    plt.axvline(x=stock_price, color='#99CC99', linestyle='--', label=f'Precio Actual: {stock_price:.2f}', lw=2)

    # Estilo minimalista: eliminar bordes superiores y derechos
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Limpiar el fondo
    ax.set_facecolor('white')
    
    # Títulos y etiquetas con estilo minimalista
    plt.title('Derivadas Reales de Opciones respecto al Strike Price', fontsize=14, fontweight='light')
    plt.xlabel('Precio de Ejercicio (Strike Price)', fontsize=12)
    plt.ylabel('Derivada (dC/dK y dP/dK)', fontsize=12)
    
    # Añadir cuadrícula suave
    plt.grid(True, color='gray', alpha=0.3)
    
    # Leyenda
    plt.legend(loc='best', frameon=False)

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)
    plt.close()

# Gráfico minimalista para derivadas respecto al precio de la opción
def plot_option_derivatives_minimalist(calls_df, puts_df, stock_price):
    plt.figure(figsize=(10, 6))

    # Eliminar los valores nulos o None de las derivadas para mostrar solo los valores reales
    calls_df = calls_df.dropna(subset=['dK/dC'])
    puts_df = puts_df.dropna(subset=['dK/dP'])

    # Estilo minimalista
    plt.style.use('seaborn-whitegrid')

    # Graficar los puntos reales de las derivadas de los calls respecto al precio de la opción
    plt.scatter(calls_df['Option Price (C)'], calls_df['dK/dC'], color='#FF9999', marker='o', label='Calls (dK/dC)', alpha=0.7)
    
    # Graficar los puntos reales de las derivadas de los puts respecto al precio de la opción
    plt.scatter(puts_df['Option Price (P)'], puts_df['dK/dP'], color='#99CCFF', marker='o', label='Puts (dK/dP)', alpha=0.7)

    # Estilo minimalista: eliminar bordes superiores y derechos
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Limpiar el fondo
    ax.set_facecolor('white')

    # Títulos y etiquetas con estilo minimalista
    plt.title('Derivadas Reales respecto al Precio de la Opción', fontsize=14, fontweight='light')
    plt.xlabel('Precio de la Opción', fontsize=12)
    plt.ylabel('Derivada (dK/dC y dK/dP)', fontsize=12)

    # Añadir cuadrícula suave
    plt.grid(True, color='gray', alpha=0.3)
    
    # Leyenda
    plt.legend(loc='best', frameon=False)

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)
    plt.close()

# Streamlit app
st.title("Opciones y Derivadas - Minimalista")
ticker = st.text_input("Ingrese el ticker de la acción (ej. AAPL):")

if ticker:
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options
    expiration_date = st.selectbox("Seleccione la fecha de vencimiento de las opciones:", expiration_dates)
    
    if st.button("Obtener opciones y gráficos minimalistas"):
        try:
            # Obtener el precio actual de la acción
            stock_price = stock.history(period="1d")['Close'].iloc[0]  # Último precio de cierre

            # Obtener los precios de opciones
            calls_df, puts_df = get_option_prices(ticker, expiration_date)
            calls_df = calculate_derivative_strike(calls_df, 'call')
            puts_df = calculate_derivative_strike(puts_df, 'put')

            # Calcular derivadas respecto al precio de la opción
            calls_df = calculate_derivative_option(calls_df, 'call')
            puts_df = calculate_derivative_option(puts_df, 'put')

            st.subheader("Precios de Opciones de Compra (Calls)")
            styled_calls_df = calls_df.style.apply(highlight_itm_calls, stock_price=stock_price, axis=1)
            st.dataframe(styled_calls_df)  # Mostrar la tabla con formato

            st.subheader("Precios de Opciones de Venta (Puts)")
            styled_puts_df = puts_df.style.apply(highlight_itm_puts, stock_price=stock_price, axis=1)
            st.dataframe(styled_puts_df)  # Mostrar la tabla con formato

            # Graficar los puntos reales respecto al Strike Price en estilo minimalista
            plot_real_derivatives_minimalist(calls_df, puts_df, stock_price)

            # Graficar las derivadas respecto al precio de la opción en estilo minimalista
            plot_option_derivatives_minimalist(calls_df, puts_df, stock_price)

        except Exception as e:
            st.error(f"Error: {e}")
