import yfinance as yf
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
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
    start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.date(2021, 11, 10))
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
st.header(f"Adjusted Close Price\n {company_name}")
st.bar_chart(tickerDF.Volume)

# dataframe
st.header(f"DataFrame!\n {company_name}")
st.write(tickerDF)
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
st.header('**Bollinger Bands**')
qf = cf.QuantFig(tickerDF, title='BB For 20MA', legend='top', name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

# Plot

# ## MACD (Moving Average Convergence Divergence)
# MACD
# Plot
# CCI
# Plot
# ## RSI (Relative Strength Index)
# RSI
# Plot
# ## OBV (On Balance Volume)
# OBV
# Plot
