import streamlit as st
import pandas as pd
import base64
import seaborn as sns
import numpy as np
import yfinance as yf
import plotly.graph_objs as go

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

selectbox_sorted = st.sidebar.multiselect('Which sector you want select', sorted_selected)

# filter the data

select_sector = df[df['GICS Sector'].isin(selectbox_sorted)]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' +
         str(select_sector.shape[0]) + ' rows and ' + str(select_sector.shape[1]) + ' columns.')
st.dataframe(select_sector)

# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href


st.markdown(filedownload(select_sector), unsafe_allow_html=True)

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

# Plot Closing Price of Query Symbol


def plotly_plot(symbol):
    df = pd.DataFrame(data[symbol].Close)
    df['Date'] = df.index
    fig = go.Scatter(x=df.index, y=df.Close)
    layout = go.Layout(title=symbol)
    Fig = go.Figure(data=fig, layout=layout)
    Fig.update_layout(xaxis_rangeslider_visible=True)
    Fig.update_layout(
        autosize=False,
        width=800,
        height=650)
    return st.plotly_chart(Fig)


num_company = st.sidebar.slider('Number of Companies', 1, 5)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in list(select_sector.Symbol)[:num_company]:
        plotly_plot(i)
