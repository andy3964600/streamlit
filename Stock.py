import streamlit as st
import pandas as pd
import numpy as np
import requests
import psycopg2.extras
import plotly.graph_objects as go
from FinMind.data import DataLoader
import matplotlib.pyplot as plt

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
    stock_no = st.sidebar.text_input('Stocknumber', max_chars=5)
    dl = DataLoader()
    stock_data = dl.taiwan_stock_daily(stock_id=stock_no, start_date='2021-01-01')
    st.write(stock_data)
    x_months = [i for i in range(1, 207)]
    y_sale = [x for x in stock_data['close']]
    plt.plot(x_months, y_sale, color="g")
    plt.xlabel('date')
    plt.ylabel('close $ in TSD')
    plt.grid(True)
    st.pyplot(fig=plt)
