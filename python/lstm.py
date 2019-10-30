from utils import *
import pandas
import matplotlib.pyplot as plt

# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
from dataclasses import dataclass
import logging
from tensorflow.python.client import device_lib


@dataclass
class Lstm_model_disc:
    predict_hour: int 
    window: int
    dataset_len: int
    epcoh: int
    model_name: str
    wind_tprh_file: str = DATAPATH + WIND_TPRH_CSV
    nwp_file: str = DATAPATH + NWP_CSV
    wind_tprh_nwp_file: str = DATAPATH + WIND_TPRH_NWP_CSV
    dropout: float =  0
    train_test_split: float = 0.67


    def __init__(self, predict_hour, window, dataset_len, epcoh, dropout):
        self.predict_hour = predict_hour
        self.window = window
        self.dataset_len = dataset_len
        self.epcoh = epcoh
        self.dropout = dropout
        self.model_name = 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh) + 'drop'+ str(self.dropout*100)

# convert an array of values into a dataset matrix
def create_dataset(dataset, window ,predict_hour=1):
    dataX, dataY = [], []
    i = window
    while i < len(dataset)-predict_hour + 1:
    #for i in range(len(dataset)-predict_hour-1):
        a = dataset[(i-window):(i), 0]
        dataX.append(a)
        dataY.append(dataset[i+predict_hour-1, 0])
        i += 1
    return numpy.array(dataX), numpy.array(dataY)

def lstm(obs_csv,npw_csv,window, predict_hour, train_test_split = 0.67):
    numpy.random.seed(7)
    # load the dataset
    dataframe = read_csv(obs_csv, usecols=[1], engine='python')
    npwDataframe = read_csv(npw_csv, usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    npwDataset = npwDataframe.values
    npwDataset = npwDataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * train_test_split)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    npw_test = npwDataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1   


    trainX, trainY = create_dataset(train, window,predict_hour)
    testX, testY = create_dataset(test,window, predict_hour)
    nwpX, nwpY = create_dataset(npw_test,window,predict_hour)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, window)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    nwpScore = math.sqrt(mean_squared_error(testY[0], nwpY))
    print('nwp Score: %.2f RMSE' % (nwpScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[window + predict_hour - 1 :len(trainPredict)+window + predict_hour - 1, :] = trainPredict

    print('trainX trainY testX testY npwX nwpY shape', trainX.shape, trainY.shape, testX.shape, testY.shape, nwpX.shape, nwpY.shape)
    print('trainPredict, testPredict, nwpPrdict', trainPredict.shape, testPredict.shape, nwpY.shape)
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    print('train_size', train_size)
    testPredictPlot[train_size + window + predict_hour -1 : train_size + window + predict_hour -1 + len(testPredict), :] = testPredict
    


    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(npwDataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    #plt.plot(nwpPredictPlot)
    plt.savefig(RESULTPATH+'train'+str(trainX.shape[0])+'window_'+str(window)+'predict_' + str(predict_hour)+'.pdf')
    #plt.show()



def run_saved_model(model_disc: Lstm_model_disc, testX):
    numpy.random.seed(7)
    # load the dataset
    dataframeY = read_csv(model_disc.wind_tprh_file, usecols=['wind'], engine='python')
    
    dataset_len = model_disc.dataset_len
    datasetY = dataframeY.values[:dataset_len]
    datasetY = datasetY.astype('float32')

    # normalize the dataset
    scalerY = MinMaxScaler(feature_range=(0, 1))
    datasetY = scalerY.fit_transform(datasetY)
    # create and fit the LSTM network
    model = keras.models.load_model(MODEL_PATH + model_disc.model_name + '.h5')
    testPredict = model.predict(testX)
    testPredict = scalerY.inverse_transform(testPredict)
    return testPredict

def create_multifeature_dataset(dataset, window,predict_hour = 1):
    dataX, dataY = [], []
    i = window
    while i < len(dataset)-predict_hour + 1:
        a = dataset[(i-window):(i), :]
        dataX.append(a)
        dataY.append(dataset[i+predict_hour-1, 0])
        i += 1
    return numpy.array(dataX), numpy.array(dataY)

def create_multifeature_nwp_dataset(dataset, nwp_start, window, predict_hour = 1 ):
    dataX, dataY = [], []
    i = window
    while i < len(dataset)-predict_hour + 1:
        a = dataset[(i-window):(i), : nwp_start]
        nwp_predict = dataset[i + predict_hour - 1, nwp_start:]
        print('a',a.shape)
        obs_nwp_featue_list = []
        for row in a:
            tmp = numpy.append(row, nwp_predict)
            obs_nwp_featue_list.append(tmp)
        obs_nwp_featue = numpy.array(obs_nwp_featue_list)
        print('obs_nwp_feature', obs_nwp_featue.shape)
        dataX.append(obs_nwp_featue)
        dataY.append(dataset[i+predict_hour-1, 0])
        i += 1
    return numpy.array(dataX), numpy.array(dataY)
    
def lstm_multifeature(model_disc: Lstm_model_disc, model_note:str = ''):
    numpy.random.seed(7)
    # load the dataset
    #dataframe = read_csv(model_disc.wind_tprh_file, usecols=['wind','ap','tmp','humi'], engine='python')
    dataframe = read_csv(model_disc.wind_tprh_nwp_file, usecols = ['wind','ap','tmp','humi', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp'],engine='python')
    dataframeY = read_csv(model_disc.wind_tprh_file, usecols=['wind'], engine='python')
    npwDataframe = read_csv(model_disc.nwp_file, usecols=['wind'], engine='python')
    
    dataset_len = model_disc.dataset_len
    dataset = dataframe.values[:dataset_len]
    dataset = dataset.astype('float32')
    datasetY = dataframeY.values[:dataset_len]
    datasetY = datasetY.astype('float32')

    npwDataset = npwDataframe.values[:dataset_len]
    npwDataset = npwDataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))
    print('dataset shape',dataset.shape)
    print('dataset Y shape',dataset.shape)
    dataset = scaler.fit_transform(dataset)
    datasetY = scalerY.fit_transform(datasetY)
    # split into train and test sets
    train_test_split = model_disc.train_test_split
    train_size = int(len(dataset) * train_test_split)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    npw_test = npwDataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1   


    window = model_disc.window
    predict_hour = model_disc.predict_hour
    epoch = model_disc.epcoh
    #trainX, trainY = create_multifeature_dataset(train, window, predict_hour)
    trainX, trainY = create_multifeature_nwp_dataset(train, 4, window, predict_hour)
    print('trainY shape', trainY.shape)
    #testX, testY = create_multifeature_dataset(test, window, predict_hour)
    testX, testY = create_multifeature_nwp_dataset(test, 4, window, predict_hour)
    nwpX, nwpY = create_dataset(npw_test,window,predict_hour)
    print('trainX ', trainX.shape)
    print('trainX 0',trainX[0].shape)

    # create and fit the LSTM network
    model = Sequential()
    #model.add(LSTM(window, input_shape=(trainX.shape[1], trainX.shape[2])))
    #model.add(LSTM(window, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(window, input_shape=(trainX.shape[1], trainX.shape[2]), recurrent_dropout= model_disc.dropout))
    model.add(Dense(int(window/2)))
    model.add(Dense(int(window/4)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=2)
    model.save(MODEL_PATH + model_disc.model_name + '_' + model_note +'.h5')
    #sys.exit()
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    print('train',trainPredict.shape)
    # invert predictions
    trainPredict = scalerY.inverse_transform(trainPredict)

    trainY = scalerY.inverse_transform([trainY])
    testPredict = scalerY.inverse_transform(testPredict)
    testY = scalerY.inverse_transform([testY])
    # calculate root mean squared error
    print('trainY shape, trainPredict shape', trainY.shape, trainPredict.shape)
    
    logging.info(model_disc.model_name)
    trainScore = math.sqrt(mean_squared_error(numpy.transpose(trainY), trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    logging.info('  Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(numpy.transpose(testY), testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    logging.info('  Test Score: %.2f RMSE' % (testScore))

    nwpScore = math.sqrt(mean_squared_error(numpy.transpose(testY), nwpY))
    print('nwp Score: %.2f RMSE' % (nwpScore))
    logging.info('  nwp Score: %.2f RMSE' % (nwpScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[window + predict_hour - 1 :len(trainPredict)+window + predict_hour - 1, :] = trainPredict

    print('trainX trainY testX testY npwX nwpY shape', trainX.shape, trainY.shape, testX.shape, testY.shape, nwpX.shape, nwpY.shape)
    print('trainPredict, testPredict, nwpPrdict', trainPredict.shape, testPredict.shape, nwpY.shape)
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    print('train_size', train_size)
    testPredictPlot[train_size + window + predict_hour -1 : train_size + window + predict_hour -1 + len(testPredict), :] = testPredict
    


    plt.plot(scaler.inverse_transform(dataset)[:,0])
    
    plt.plot(npwDataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    #plt.plot(nwpPredictPlot)
    plt.savefig(RESULTPATH+  model_disc.model_name + '_' + model_note+'.pdf')
    #plt.show()




def load_models_and_run():
    
    window = 24
    dataset_len = 5000
    epcoh = 5
    predict_hour = 1
    dropout = 0.4
    model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh, dropout)
    dataframe = read_csv(model_disc.wind_tprh_file, usecols=['wind','ap','tmp','humi'], engine='python')     
    dataframeY = read_csv(model_disc.wind_tprh_file, usecols=['wind'], engine='python')
    
    datasetY = dataframeY.values[:dataset_len]
    datasetY = datasetY.astype('float32')

    npwDataframe = read_csv(model_disc.nwp_file, usecols=['wind'], engine='python')
    npwDataset = npwDataframe.values[:dataset_len]
    npwDataset = npwDataset.astype('float32')

    # normalize the dataset

    dataset = dataframe.values[:dataset_len]
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_test_split = model_disc.train_test_split
    train_size = int(len(dataset) * train_test_split)
    
    
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    testY = datasetY[train_size:len(dataset),:]
    
    predict_start, predict_end = 1, 25
    
    predictY = []

    npw_test = npwDataset[train_size:len(dataset),:]
   
    model_predicts = []
    for predict_hour in range(predict_start, predict_end):        
        print('processing model ', predict_hour)
        testX, dummy = create_multifeature_dataset(test, window, predict_hour)
        model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh,dropout)
        # split into train and test sets
        # print('testX_0',testX_0)
        model_predict = [[0]] * (window + predict_hour - 1)
        model_predict.extend(run_saved_model(model_disc, testX))
        model_predicts.append(model_predict)
    
    hour = 1
    for i in range(len(testY)):


        predict = model_predicts[hour-1][i]
        predictY.append(predict)
        hour += 1
        if hour >= 25:
            hour = 1


    testScore = math.sqrt(mean_squared_error(testY, predictY))
    print('Test Score: %.3f RMSE' % (testScore))
    logging.info('  Test Score: %.3f RMSE' % (testScore))

    npwScore = math.sqrt(mean_squared_error(testY, npw_test))
    print('npw Score : %.3f RMSE' % (npwScore))
    logging.info('  npw Score : %.3f RMSE' % (npwScore))

    plt.plot(testY)
    plt.plot(predictY)
    plt.plot(npw_test)
    #plt.show()




if __name__ == '__main__':
    #lstm(DATAPATH + WIND_CSV, DATAPATH + NWP_CSV, window, predict_hour)
    LOG_FILENAME = "logfile.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)    

    #load_models_and_run()
    #sys.exit()

    #predict_hour, window, datalen, epcoh
    for predict_hour in range(1,23):        
        train_test_split = 0.67
        window = 24
        dataset_len = 5000
        epcoh = 5
        dropout = 0.4
        model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh, dropout)
        lstm_multifeature(model_disc, 'nwpFeature')