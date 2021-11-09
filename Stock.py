import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
st.title('My Stock Data Search')
st.sidebar.title(' Option')

#df = pd.DataFrame(np.random.randn(50, 20), columns=('stock %d' % i for i in range(20)))

# st.dataframe(df)

option = st.sidebar.selectbox(
    "Which Dashboard?", ('twitter', 'Kbar', 'stock_id', 'chart', 'pattern'), 3)

st.header(option)
if option == 'twitter':
    st.subheader('This is twitters Dashboard')

if option == 'Kbar':
    st.subheader('This is Kbar')
if option == 'stock_id':
    tickerSymbol = st.sidebar.text_input(
        "Please in put your stock_ID(ex: xxxx.TW in Taiwan, XXXX in America)", value='MSFT', max_chars=None, key=None, type='default')
    st.subheader('This is %s' % tickerSymbol + ' line_chart :')
    tickerData = yf.Ticker(tickerSymbol)
    tickerDF = tickerData.history(period='1d', start='2021-01-01')
    st.line_chart(tickerDF.Close)
    st.line_chart(tickerDF.Volume)
