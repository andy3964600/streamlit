import talib
import datetime as datetime
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mpl_finance as mpf
st.title('My Stock Data Search')
st.sidebar.title(' Option')

# df = pd.DataFrame(np.random.randn(50, 20), columns=('stock %d' % i for i in range(20)))

# st.dataframe(df)

option = st.sidebar.selectbox(
    "Which Dashboard?", ('stock_id', 'Discussion and News'))

st.header(option)
if option == 'stock_id':
    tickerSymbol = st.sidebar.text_input(
        "Please in put your stock_ID(ex: xxxx.TW in Taiwan, XXXX in America)", value='MSFT', max_chars=None, key=None, type='default')
    st.subheader('This is %s' % tickerSymbol + ' line_chart :')
    tickerData = yf.Ticker(tickerSymbol)
    tickerDF = tickerData.history(period='1d', start='2021-01-01')
    st.subheader('這是 %s' % tickerSymbol + ' 的收盤價圖 :')
    st.line_chart(tickerDF.Close)
    st.subheader('這是 %s' % tickerSymbol + ' 的成交量圖 :')
    st.bar_chart(tickerDF.Volume)
    tickerDF.index = tickerDF.index.format(formatter=lambda x: x.strftime('%Y-%m-%d'))

    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(range(0, len(tickerDF.index), 10))
    ax.set_xticklabels(tickerDF.index[::10])
    mpf.candlestick2_ochl(ax, tickerDF['Open'], tickerDF['Close'], tickerDF['High'],
                          tickerDF['Low'], width=0.6, colorup='r', colordown='g', alpha=0.6)
    st.subheader('這是 %s' % tickerSymbol + ' 的Kbar :')
    st.pyplot(fig)
    sma_10 = talib.SMA(np.array(tickerDF['Close']), 10)
    sma_30 = talib.SMA(np.array(tickerDF['Close']), 30)
    sma_5 = talib.SMA(np.array(tickerDF['Close']), 5)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(range(0, len(tickerDF.index), 10))
    ax.set_xticklabels(tickerDF.index[::10], rotation=90)

    mpf.candlestick2_ochl(ax, tickerDF['Open'], tickerDF['Close'], tickerDF['High'],
                          tickerDF['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    ax.plot(sma_10, label='10MA')
    ax.plot(sma_30, label='30MA')
    ax.plot(sma_5, label='5MA')
    ax.legend()
    st.subheader('這是 %s' % tickerSymbol + ' 的5MA+10MA+30MA+的Kbar :')
    st.pyplot(fig)
    st.dataframe(tickerDF)
