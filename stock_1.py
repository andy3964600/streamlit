import yfinance as yf
import streamlit as st
import datetime
import talib
import ta
import pandas as pd
import requests
import matplotlib.pyplot as plt
import mpl_finance as mpf
yf.pdr_override()

st.write("""
# DAS(Data Analysis Stock) Overview
Shown below are the **MA**, **BB**, **MACD**, **CCI**,**KBar**, **Close**, **Volume** and **RSI** of yours input!
""")

st.sidebar.header('**Options**')
today = datetime.date.today()

# take 3d inofrmation


def User_input():
    ticker = st.sidebar.text_input("Ticker", 'AAPL')
    start_date = st.sidebar.text_input("Start Date", '2020-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    return ticker, start_date, end_date


symbol, start, end = User_input()


tickerData = yf.Ticker(symbol)
tickerDF = tickerData.history(period='1d', start="%s" % start)
company_name = symbol
start = pd.to_datetime(start)
end = pd.to_datetime(end)

# Read data
data = yf.download(symbol, start, end)

# Adjusted Close Price
st.header(f"Adjusted Close Price\n {company_name}")
st.line_chart(data['Adj Close'])

# Volume
st.bar_chart(tickerDF.Volume)

# ## SMA and EMA
# Simple Moving Average
data['SMA'] = talib.SMA(data['Adj Close'], timeperiod=20)

# Exponential Moving Average
data['EMA'] = talib.EMA(data['Adj Close'], timeperiod=20)

# Plot
st.header(f"Simple Moving Average vs. Exponential Moving Average\n {company_name}")
st.line_chart(data[['Adj Close', 'SMA', 'EMA']])

# KBar
st.header(f"KBar\n {company_name}")
tickerDF.index = tickerDF.index.format(formatter=lambda x: x.strftime('%Y-%m-%d'))

fig = plt.figure(figsize=(15, 10), dpi=160)

ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(range(0, len(tickerDF.index), 10))
ax.set_xticklabels(tickerDF.index[::10], rotation=90)
ax.grid()
mpf.candlestick2_ochl(ax, tickerDF['Open'], tickerDF['Close'], tickerDF['High'],
                      tickerDF['Low'], width=0.6, colorup='r', colordown='g', alpha=0.6)
st.pyplot(fig)
# Bollinger Bands
data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(
    data['Adj Close'], timeperiod=20)

# Plot
st.header(f"Bollinger Bands\n {company_name}")
st.line_chart(data[['Adj Close', 'upper_band', 'middle_band', 'lower_band']])

# ## MACD (Moving Average Convergence Divergence)
# MACD
data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(
    data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Plot
st.header(f"Moving Average Convergence Divergence\n {company_name}")
st.line_chart(data[['macd', 'macdsignal']])

# CCI (Commodity Channel Index)
# CCI
cci = ta.trend.cci(data['High'], data['Low'], data['Close'], n=31, c=0.015)

# Plot
st.header(f"Commodity Channel Index\n {company_name}")
st.line_chart(cci)

# ## RSI (Relative Strength Index)
# RSI
data['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)

# Plot
st.header(f"Relative Strength Index\n {company_name}")
st.line_chart(data['RSI'])

# ## OBV (On Balance Volume)
# OBV
data['OBV'] = talib.OBV(data['Adj Close'], data['Volume'])/10**6

# Plot
st.header(f"On Balance Volume\n {company_name}")
st.line_chart(data['OBV'])
