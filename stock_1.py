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
import math
yf.pdr_override()
plt.style.use('seaborn')
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

st.sidebar.header('**Options**')
today = datetime.date.today()
option = st.sidebar.selectbox(
    "Deeplearning", ('Tech_Analysis', 'DL(LSTM) Prediction For Next Day',  'DL(LSTM) Prediction For 1 week', 'DL(LSTM) Prediction For 1 month'))

if option == 'Tech_Analysis':
    st.title("""
    # DAS(Data Analysis Stock) Overview
    Shown below are the **BB**, **KBar**, **Close**, **Volume** of yours input!
    """)

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

    KF = plt.figure(figsize=(16, 8))

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
if option == 'DL(LSTM) Prediction For Next Day':
    st.title("""
    DL model(LSTM) Prediction For Next Day
    """)
    DF = data.filter(['Close'])
    DFset = DF.values
    # Create the train/test dataset
    train_set_len = math.ceil(len(DFset) * 0.75)
    Normalize_scalar = MinMaxScaler(feature_range=(0.0, 1.0))
    df1 = Normalize_scalar.fit_transform(np.array(DFset).reshape(-1, 1))
    train_data, test_data = df1[0:train_set_len, :], df1[train_set_len:len(df1), :1]
    x_train = []
    y_train = []

    for i in range(20, len(train_data)):
        x_train.append(train_data[i-20:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = []
    y_test = []

    for i in range(20, len(test_data)):
        x_test.append(test_data[i-20:i, 0])
        y_test.append(test_data[i, 0])
    x_test, y_test = np.array(x_test),  np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # Create the stocl LSTM NN
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(x_train, y_train, batch_size=64, epochs=80, verbose=1)
    # Get the models prdicted price values
    pred = model.predict(x_test)
    pred = Normalize_scalar.inverse_transform(pred)
    train = DF[: train_set_len+20]
    Val = DF[train_set_len+20:]
    Val['Predictions'] = pred
    # Visualize the data
    st.header(f"Your trained model for validation : \n {company_name}")
    fig = plt
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price TWD ($)')
    plt.plot(train['Close'])
    plt.plot(Val[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
    plt.show()
    st.pyplot(fig)
    plt.clf()
    # Create the new DF
    new_df = data.filter(['Close'])
    # Get last 60 day's closing price
    last_month = new_df[-20:].values
    last_month_scaled = Normalize_scalar.transform(last_month)
    x_test = []
    x_test.append(last_month_scaled)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # prediciton
    pred_price = model.predict(x_test)
    pred_price = Normalize_scalar.inverse_transform(pred_price)
    st.subheader('The prediction of Close price on next day= %s' %
                 pred_price)
# prediciton
pred_price = model.predict(x_test)
pred_price = Normalize_scalar.inverse_transform(pred_price)
print('The prediction of Close price on next day= %s' % pred_price)
if option == 'DL(LSTM) Prediction For 1 week':
    st.title("""
    DL model(LSTM) Prediction for 1 week
    """)
# Predict forecast with Prophet.
    df1 = data.reset_index()['Close']


# LSTM are sensitive to the scale of the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    st.write('Your numbers of train_set of data')
    st.write(df1.shape)
# splitting dataset into train and test split
# train_set
    training_size = int(len(df1) * 0.75)
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
    time_step = 30
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
# reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(30, 1)))
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
    look_back = 30
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    # plot baseline and predictions
    st.header(f"Your trained model for validation : \n {company_name}")
    fig = plt
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price TWD ($)')
    plt.legend(['Train', 'Validaition'], loc='upper left')
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    st.pyplot(fig)
    plt.clf()
    # demonstrate prediction for next 10 days
    x_input = test_data[len(test_data)-30:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = 30
    i = 0
    while(i < 5):
        if(len(temp_input) > 30):
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

    day_new = np.arange(1, 31)
    day_pred = np.arange(31, 36)
    st.header(f"The prediction in 5 days : \n {company_name}")
    dff = plt
    dff.title('Model')
    dff.xlabel('Date')
    dff.ylabel('Close Price TWD ($)')
    dff.plot(day_new, scaler.inverse_transform(df1[len(df1)-30:]))
    dff.plot(day_pred, scaler.inverse_transform(lst_output))
    dff.legend(['Train', 'Predictions'], loc='upper left')
    st.write(scaler.inverse_transform(lst_output))
    st.pyplot(dff)
    df3 = df1.tolist()
    df3.extend(lst_output)
    st.line_chart(df3[1200:])

    df3 = scaler.inverse_transform(df3).tolist()
    st.header(f"Combine the real data and prediction data\n {company_name}")
    st.line_chart(df3)
if option == 'DL(LSTM) Prediction For 1 month':
    st.title("""
    DL model(LSTM) Prediction for 1 month
    """)
# Predict forecast with Prophet.
    df1 = data.reset_index()['Close']


# LSTM are sensitive to the scale of the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    st.write('Your numbers of train_set of data')
    st.write(df1.shape)
# splitting dataset into train and test split
# train_set
    training_size = int(len(df1)*0.8)
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
    model.add(LSTM(80, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(80, return_sequences=True))
    model.add(LSTM(80))
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
    st.header(f"Your trained model for validation :\n {company_name}")
    fig = plt
    plt.xlabel('Date')
    plt.ylabel('Close Price TWD ($)')
    plt.legend(['Train', 'Validaition'], loc='upper left')
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    st.pyplot(fig)
    plt.clf()
    # demonstrate prediction for next 10 days
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
    st.header(f"The prediction in 30 days : \n {company_name}")
    dff = plt
    dff.title('Model')
    dff.xlabel('Date')
    dff.ylabel('Close Price TWD ($)')
    dff.plot(day_new, scaler.inverse_transform(df1[len(df1)-100:]))
    dff.plot(day_pred, scaler.inverse_transform(lst_output))
    dff.legend(['Train', 'Predictions'], loc='upper left')
    st.write(scaler.inverse_transform(lst_output))
    st.pyplot(dff)
    df3 = df1.tolist()
    df3.extend(lst_output)
    st.line_chart(df3[1200:])

    df3 = scaler.inverse_transform(df3).tolist()
    st.header(f"Combine the real data and prediction data :\n {company_name}")
    st.line_chart(df3)
