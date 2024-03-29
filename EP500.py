import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go


st.set_page_config(layout="wide")
st.title('S&P 500 App')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")


def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    return df


df = load_data()
sector = df.groupby('GICS Sector')

# Sidebar - Sector selection
sorted_selected = sorted(df['GICS Sector'].unique())

selectbox_sorted = st.sidebar.multiselect('Which sector you want select', sorted_selected,default=["Communication Services"])

# filter the data

select_sector = df[df['GICS Sector'].isin(selectbox_sorted)]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' +
         str(select_sector.shape[0]) + ' rows and ' + str(select_sector.shape[1]) + ' columns.')
st.dataframe(select_sector)

# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806


# https://pypi.org/project/yfinance/

data = yf.download(
    tickers=list(select_sector.Symbol),
    period="ytd",
    interval="1d",
    group_by='ticker',
    auto_adjust=True,
    prepost=True,
    threads=True,
    proxy=None
)
selectbox_sorted_Symbol = st.sidebar.multiselect('ticker?', select_sector.Symbol)
# Plot Closing Price of Query Symbol


def plotly_plot(symbol):
    df = pd.DataFrame(data[symbol].Close)
    df['Date'] = df.index
    Fig = go.Scatter(x=df.index, y=df.Close)
    layout = go.Layout(title=symbol)
    Fig = go.Figure(data=Fig, layout=layout)
    Fig.update_layout(xaxis_rangeslider_visible=True)
    Fig.update_layout(
        autosize=False,
        width=800,
        height=650)
    return st.plotly_chart(Fig)


# if st.sidebar.button('Show Plots'):
st.header('Stock Closing Price')
for i in list(selectbox_sorted_Symbol)[:len(selectbox_sorted_Symbol)]:
    plotly_plot(i)
