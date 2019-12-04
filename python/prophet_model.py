from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

from datetime import datetime  
from datetime import timedelta  


def train_prophet_model(X):
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:size+100]
    
    m = Prophet()
    train_df = addDates(train)
    m.fit(train_df)

    window = 24

    future = m.make_future_dataframe(periods=window)
    forecast = m.predict(future)
    
    fig1 = m.plot(forecast)
    
    fig1.show()
    plt.show()
    # yhat_df = forecast['yhat']
    # yhat = yhat_df.values.tolist()

    # print('len train is',len(train_df))
    # print('yhat is ', len(yhat), yhat[-10:])
    
    # # plot
    # plt.clf()
    # plt.plot(test[:window])
    # plt.plot(yhat[-window:], color='red')
    # plt.show()
    pass

def addDates(X):
    data = {'ds': [], 'y': []}

    dates = []
    ys = []

    date = datetime.today()
    for y in X:
        dates.append(str(date))
        ys.append(y)
        date = date + timedelta(days=1)
        

    data['ds'] = dates
    data['y'] = ys
    df = pd.DataFrame(data)

    print(df.head)
    return df

def train_arima_model(X):
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:size+100]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    plt.clf()
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()
    pass


if __name__ == '__main__':



    pass