import yfinance as yf
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
import mpl_finance as mpf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
yf.pdr_override()

st.write("""
# DAS(Data Analysis Stock) Overview
Shown below are the **BB**, **KBar**, **Close**, **Volume** of yours input!
""")

st.sidebar.header('**Options**')
today = datetime.date.today()
option = st.sidebar.selectbox(
    "Deeplearning", ('Tech_Analysis', 'Start_Analysis_DL'))
# take 3d inofrmation


def User_input():
    ticker = st.sidebar.text_input("Ticker", '2376.TW')
    start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.date(2021, 11, 10))
    return ticker, start_date, end_date


symbol, start, end = User_input()
tickerData = yf.Ticker(symbol)
tickerDF = tickerData.history(start="%s" % start)
tickerDF = tickerData.history(end="%s" % end)
company_name = symbol
start = pd.to_datetime(start)
end = pd.to_datetime(end)
# Read data
data = yf.download(symbol, start, end)
st.write(data)
if option == 'Tech_Analysis':

    # Adjusted Close Price
    st.header(f"Adjusted Close Price\n {company_name}")
    st.line_chart(data['Adj Close'])

    # Volume
    st.header(f"Adjusted Close Price\n {company_name}")
    st.bar_chart(data.Volume)

# dataframe
    st.header(f"DataFrame!\n {company_name}")
    st.write(data.tail())
# KBar
    st.header(f"KBar\n {company_name}")
    data.index = data.index.format(formatter=lambda x: x.strftime('%Y-%m-%d'))

    KF = plt.figure(figsize=(32, 15), dpi=160)

    ax = KF.add_subplot(1, 1, 1)
    ax.set_xticks(range(0, len(data.index), 10))
    ax.set_xticklabels(data.index[::10], rotation=90)
    mpf.candlestick2_ochl(ax, data['Open'], data['Close'], data['High'],
                          data['Low'], width=0.6, colorup='r', colordown='g', alpha=0.6)
    st.pyplot(KF)
    # Bollinger Bands
    st.header(f"Bollinger Bands\n {company_name}")
    qf = cf.QuantFig(data, title='BB For 20MA', legend='top', name='GS')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

if option == 'Start_Analysis_DL':
    st.text('*Calculating the Deeplearning for prediction the stock close....Plz wait*')
# Predict forecast with Prophet.
    df1 = data.reset_index()['Close']


# LSTM are sensitive to the scale of the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    st.write('Your numbers of train_set of data')
    st.write(df1.shape)
# splitting dataset into train and test split
# train_set
    training_size = int(len(df1)*0.65)
# test_ser
    test_size = len(df1)-training_size

# reset the df1 (upper = train, else = test, time series)
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
# reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    st.write('Training the model....')
    model.fit(X_train, y_train, validation_data=(
        X_test, ytest), epochs=100, batch_size=64, verbose=1)
    st.write('Model fitting is done...')
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # Transformback to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    look_back = 100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    # plot baseline and predictions
    fig = plt
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    st.header(f"After learning, the learning result..(green and orange line)\n {company_name}")
    st.pyplot(fig)
    plt.clf()
    # demonstrate prediction for next 10 days
    st.write(len(test_data))
    st.write('Now, Calculate to prediction for next 10 days... plz wait')
    x_input = test_data[len(test_data)-100:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = 100
    i = 0
    while(i < 30):
        if(len(temp_input) > 100):
         # print(temp_input)
            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
         # print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
         # print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i+1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i+1

    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)

    dff = plt
    dff.plot(day_new, scaler.inverse_transform(df1[len(df1)-100:]))
    dff.plot(day_pred, scaler.inverse_transform(lst_output))
    st.write('Your next 10 day stock close price in predicion 30 day')
    st.write(scaler.inverse_transform(lst_output))
    st.pyplot(dff)

    df3 = df1.tolist()
    df3.extend(lst_output)
    st.header(f"The orange line is your prediction for 30 days\n {company_name}")
    st.line_chart(df3[1200:])

    df3 = scaler.inverse_transform(df3).tolist()
    st.header(f"Combine the real data and prediction data\n {company_name}")
    st.line_chart(df3)
