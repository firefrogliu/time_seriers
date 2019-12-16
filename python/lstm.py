import pandas
from constants import *
import matplotlib.pyplot as plt
import csv
# LSTM for international airline passengers problem with regression framing
import numpy
from pandas import read_csv
import keras
from keras.models import Sequential
from keras.layers import Dense, RepeatVector, TimeDistributed
from keras.layers import LSTM,CuDNNLSTM
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
from dataclasses import dataclass
import logging
from tensorflow.python.client import device_lib
from pyplotz.pyplotz import PyplotZ
from clean_val_data import get_val_testX

from detrend import moving_average, emd_detrend, ceemd_detrend, vmd_detrend
import pickle
import random
import copy

from garget import *
from keras import backend as K

from prophet_model import *

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

@dataclass
class Lstm_model_disc:
    predict_hour: int 
    window: int
    dataset_len: int
    epcoh: int
    model_name: str
    site: str
    wind_tprh_nwp_file: str
    min_wind: int = 0 
    max_wind: int = 100
    dropout: float =  0
    train_test_split: float = 0.67


    def __init__(self, predict_hour, window, dataset_len, epcoh, dropout, site, min_wind, max_wind):
        self.predict_hour = predict_hour
        self.window = window
        self.dataset_len = dataset_len
        self.epcoh = epcoh
        self.dropout = dropout
        self.site = site
        self.min_wind = min_wind
        self.max_wind = max_wind   
        
        #self.wind_tprh_nwp_file = DATAPATH + site + WIND_TPRH_NWP_ALLYEAR_CSV
        self.wind_tprh_nwp_file = DATAPATH + site + UPDATA_OBS_NWP_ALLYEAR_CSV
        self.model_name = 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh) + 'drop'+ str(self.dropout*100) + 'minWind' + str(self.min_wind) + 'maxWind' + str(self.max_wind)
        print('model name', self.model_name)
    
class Classification_model_disc(Lstm_model_disc):
    wind_bar = 4
    
    def __init__(self, predict_hour, window, dataset_len, epcoh, dropout, site, wind_bar):
        self.predict_hour = predict_hour
        self.window = window
        self.dataset_len = dataset_len
        self.epcoh = epcoh
        self.dropout = dropout
        self.site = site
        self.wind_bar = wind_bar
        
        self.wind_tprh_nwp_file = DATAPATH + site + UPDATA_OBS_NWP_ALLYEAR_CSV
        self.model_name = 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh) + 'drop'+ str(self.dropout*100) + 'windBar' + str(self.wind_bar)
        print('model name', self.model_name)    

    

class Moving_ave_model_disc(Lstm_model_disc):
    ma_window = 4

    def __init__(self, predict_hour, window, dataset_len, epcoh, dropout, site, ma_window):
        self.predict_hour = predict_hour
        self.window = window
        self.dataset_len = dataset_len
        self.epcoh = epcoh
        self.dropout = dropout
        self.site = site
        self.ma_window = ma_window
        
        self.wind_tprh_nwp_file = DATAPATH + site + UPDATA_OBS_NWP_ALLYEAR_CSV
        self.model_name = 'ma_' + 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh) + 'drop'+ str(self.dropout*100) + 'maWin' + str(self.ma_window)
        print('model name', self.model_name)    

    def set_predict_hour(self,predict_hour):
        self.predict_hour = predict_hour
        #print('updateing name')
    
        self.__update_name()
        #print('name is', self.model_name)
    def __update_name(self):
        self.model_name = 'ma_' + 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh) + 'drop'+ str(self.dropout*100) + 'maWin' + str(self.ma_window)

class Ma_res_model_disc(Moving_ave_model_disc):
    def __init__(self, predict_hour, window, dataset_len, epcoh, dropout, site, ma_window):
        self.predict_hour = predict_hour
        self.window = window
        self.dataset_len = dataset_len
        self.epcoh = epcoh
        self.dropout = dropout
        self.site = site
        self.ma_window = ma_window
        
        self.wind_tprh_nwp_file = DATAPATH + site + UPDATA_OBS_NWP_ALLYEAR_CSV
        self.model_name = 'maRes_' + 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh) + 'drop'+ str(self.dropout*100) + 'maWin' + str(self.ma_window)
        print('model name', self.model_name)   

    def set_predict_hour(self,predict_hour):
        self.predict_hour = predict_hour
        print('updateing name')    
        self.__update_name()
        print('name is', self.model_name)


    def __update_name(self):
        self.model_name = 'maRes_' + 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh) + 'drop'+ str(self.dropout*100) + 'maWin' + str(self.ma_window)

class Seq2seq_model_disc(Lstm_model_disc):
    look_forward = 24

    def __init__(self, look_forward, window, dataset_len, epcoh, dropout, site):
        self.look_forward = look_forward
        self.window = window
        self.dataset_len = dataset_len
        self.epcoh = epcoh
        self.dropout = dropout
        self.site = site
        self.wind_tprh_nwp_file = DATAPATH + site + UPDATA_OBS_NWP_ALLYEAR_CSV
        self.model_name = 's2s_'+'lfwindow' + str(self.look_forward) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh)
        print('model name', self.model_name)

class Emd_model_disc(Lstm_model_disc):

    imfs_idx = 4

    def __init__(self, predict_hour, window, dataset_len, epcoh, dropout, imfs_idx, site, train_test_split = 0.67):
        self.predict_hour = predict_hour
        self.window = window
        self.dataset_len = dataset_len
        self.epcoh = epcoh
        self.dropout = dropout
        self.site = site
        self.imfs_idx = imfs_idx
        self.train_test_split = train_test_split
        self.wind_tprh_nwp_file = DATAPATH + site + UPDATA_OBS_NWP_ALLYEAR_CSV
        self.__update_name()
        print('model name', self.model_name)

    def set_immfs_idx(self,imfs_idx):
        self.imfs_idx = imfs_idx
        print('updateing name')    
        self.__update_name()
        print('name is', self.model_name)

    def set_predict_hour(self, predict_hour):
        self.predict_hour = predict_hour
        print('updateing name')    
        self.__update_name()
        print('name is', self.model_name)


    def __update_name(self):
        self.model_name = 'emd_'+ str(self.imfs_idx) + 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh)

class Merge_model_disc(Lstm_model_disc):
    

    def set_predict_hour(self, predict_hour):
        self.predict_hour = predict_hour
        print('updateing name')    
        self.__update_name()
        print('name is', self.model_name)

    def __init__(self, predict_hour, window, dataset_len, epcoh,  site, train_test_split = 0.67):
        self.predict_hour = predict_hour
        self.window = window
        self.dataset_len = dataset_len
        self.epcoh = epcoh
        self.site = site
        self.train_test_split = train_test_split
        self.wind_tprh_nwp_file = DATAPATH + site + UPDATA_OBS_NWP_ALLYEAR_CSV
        self.__update_name()
        print('model name', self.model_name)

    def __update_name(self):
        self.model_name = 'merge_emd_' + 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'train_len' + str(int(self.dataset_len * self.train_test_split)) + 'eph' + str(self.epcoh)
        

# convert an array of values into a dataset matrix
def create_dataset(dataset, window ,predict_hour=1, min_wind = 0, max_wind = 100):
    dataX, dataY = [], []
    i = window
    while i < len(dataset)-predict_hour + 1:
    #for i in range(len(dataset)-predict_hour-1):
        wind = dataset[i+predict_hour-1, 0]

        if wind < min_wind or wind >= max_wind:
            i = i + 1
            continue
        
        dataY.append(wind)        
        a = dataset[(i-window):(i), 0]
        dataX.append(a)
        i += 1
    return numpy.array(dataX), numpy.array(dataY)

def run_saved_classification_model(model_disc:Classification_model_disc, testX):
    numpy.random.seed(7)
    # load the dataset
    # create and fit the LSTM network
    model = keras.models.load_model(MODEL_PATH + model_disc.site + model_disc.model_name + '.h5', custom_objects={'f1_m': f1_m, 'precision_m':precision_m, 'recall_m':recall_m})
    testPredict = model.predict(testX)
    return testPredict

def run_saved_model(model_disc: Lstm_model_disc, testX, Ycolumn = 'wind'):
    numpy.random.seed(7)
    # load the dataset
    dataframeY = read_csv(model_disc.wind_tprh_nwp_file, usecols=[Ycolumn], engine='python')
    
    
    datasetY = dataframeY.values[:]
    datasetY = datasetY.astype('float32')

    # normalize the dataset
    scalerY = MinMaxScaler(feature_range=(0, 1))
    datasetY = scalerY.fit_transform(datasetY)
    # create and fit the LSTM network
    model = keras.models.load_model(MODEL_PATH + model_disc.site + model_disc.model_name + '.h5')
    testPredict = model.predict(testX)
    testPredict = scalerY.inverse_transform(testPredict)
    return testPredict



def creat_classification_dataset(dataset, nwp_start, window, predict_hour = 1, wind_bar = 4):
    dataX, dataY= [], []
    i = window
    while i < len(dataset)-predict_hour + 1:
        wind = dataset[i+predict_hour-1, 0]

        if wind < wind_bar:        
            dataY.append(0)
        else:
            dataY.append(1)
        
        a = dataset[(i-window):(i), : nwp_start]
        nwp_predict = dataset[i + predict_hour - 1, nwp_start:]
        obs_nwp_featue_list = []
        for row in a:
            tmp = numpy.append(row, nwp_predict)
            obs_nwp_featue_list.append(tmp)
        obs_nwp_featue = numpy.array(obs_nwp_featue_list)
        dataX.append(obs_nwp_featue)
        i += 1
    return numpy.array(dataX), numpy.array(dataY)

def create_multifeature_nwp_dataset(dataset, nwp_start, window, predict_hour = 1, Ycolumn_idx = 0, min_wind = 0, max_wind = 100, obs_end = 6, nwp_end = 7):
    dataX, dataY, nwpY = [], [], []
    i = window

    print('yclumn idx is', Ycolumn_idx)
    #print('creating dataset min max', min_wind, max_wind)
    while i < len(dataset)-predict_hour + 1:

        wind = dataset[i+predict_hour-1, Ycolumn_idx]
 
        dataY.append(wind)
        nwp_predict = dataset[i + predict_hour - 1, [nwp_start, -1]]
        nwp_predict_wind = dataset[i + predict_hour - 1, nwp_start]
        nwpY.append(nwp_predict_wind)
        a = dataset[(i-window):(i), [Ycolumn_idx,0,2,3]]
        
        obs_nwp_featue_list = []
        for row in a:
            tmp = numpy.append(row, nwp_predict)
            obs_nwp_featue_list.append(tmp)
        obs_nwp_featue = numpy.array(obs_nwp_featue_list)
        dataX.append(obs_nwp_featue)

        i += 1


    return numpy.array(dataX), numpy.array(dataY), numpy.array(nwpY)


    
def prepare_data(model_disc: Lstm_model_disc, Ycolumn : str = 'wind'):
    dataframe = read_csv(model_disc.wind_tprh_nwp_file, usecols = ['wind','dir','slp','t2', 'rh2', 'td2', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp', 'residual'],engine='python')
    dataframeNwp = read_csv(model_disc.wind_tprh_nwp_file, usecols=['nwp_wind'], engine='python')
    
    dataset_len = model_disc.dataset_len
    dataset = dataframe.values[:]
    dataset = dataset.astype('float32')
    
    
    dataframeY = read_csv(model_disc.wind_tprh_nwp_file, usecols=[Ycolumn], engine='python')
    datasetY = dataframeY.values[:]
    datasetY = datasetY.astype('float32')




    scalerY = MinMaxScaler(feature_range=(0, 1))
    datasetY = scalerY.fit_transform(datasetY)
    
    datasetNwp = dataframeNwp.values[:]
    datasetNwp = datasetNwp.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    scalerNwp = MinMaxScaler(feature_range=(0, 1))
    scalerNwp.fit_transform(datasetNwp)
    dataset = dataset[:dataset_len]
    datasetY = datasetY[:dataset_len]
    # split into train and test sets
    train_test_split = model_disc.train_test_split
    train_size = int(len(dataset) * train_test_split)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    
    min_wind = model_disc.min_wind
    max_wind = model_disc.max_wind

    min_max = [[min_wind], [max_wind]]
    min_max = numpy.array(min_max)
    numpy.reshape(min_max, (2,1))
    print('min_max',min_max.shape)
    print('datasetY',datasetY.shape)

    min_max = scalerY.transform(min_max)

    min_wind = min_max[0][0]
    max_wind = min_max[1][0]
    window = model_disc.window
    predict_hour = model_disc.predict_hour

    Ycolumn_idx = 0
    if Ycolumn == 'residual':
        Ycolumn_idx = 14
        print('Ycolumn data', Ycolumn_idx)
    

    trainX, trainY, dummy = create_multifeature_nwp_dataset(train, NWP_START_INDEX, window, predict_hour, Ycolumn_idx, min_wind, max_wind)
    testX, testY, test_nwpY = create_multifeature_nwp_dataset(test, NWP_START_INDEX, window, predict_hour, Ycolumn_idx, min_wind, max_wind)
    

    #print('trainX, trainY, testX, testY, nwpY',trainX.shape, trainY.shape, testX.shape, testY.shape, test_nwpY.shape)
    #print('trainX[0:5], trainY[0:5]', trainX[0:5], trainY[0:4])
    return window, predict_hour, dataset, trainX, trainY, testX, testY, test_nwpY,  train_size, scalerNwp, scalerY



def train_results(model_disc: Lstm_model_disc, model: Sequential, window, predict_hour ,dataset, trainX, trainY, testX, testY, test_nwpY, train_size, scalerNwp, scalerY, residual_Y  = False):
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    print('train',trainPredict.shape)
    # invert predictions
    trainPredict = scalerY.inverse_transform(trainPredict)

    trainY = scalerY.inverse_transform([trainY])
    trainY = numpy.transpose(trainY)
    testPredict = scalerY.inverse_transform(testPredict)
    testY = scalerY.inverse_transform([testY])
    testY = numpy.transpose(testY)

    test_nwpY = scalerNwp.inverse_transform([test_nwpY])
    test_nwpY = numpy.transpose(test_nwpY)

    if residual_Y:
        testPredict = testPredict + test_nwpY
        testY = testY + test_nwpY
    # calculate root mean squared error
    print('testPredict, testY, test_nwpY', testPredict.shape, testY.shape, test_nwpY.shape)    
    logging.info(model_disc.model_name)
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    logging.info('  Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    logging.info('  Test Score: %.2f RMSE' % (testScore))

    nwpScore = math.sqrt(mean_squared_error(testY, test_nwpY))
    print('nwp Score: %.2f RMSE' % (nwpScore))
    logging.info('  nwp Score: %.2f RMSE' % (nwpScore))


    bw_testY = []
    bw_predictY = []
    bw_nwp_test = []
    for i in range(len(testY)):
        if testY[i] > 4:
            bw_testY.append(testY[i])    
            bw_predictY.append(testPredict[i])
            bw_nwp_test.append(test_nwpY[i])
    bw_testScore = 0
    bw_npwScore = 0
    if len(bw_testY) > 0:
        bw_testScore = math.sqrt(mean_squared_error(bw_testY, bw_predictY))
        print('Big Wind Test Score: %.3f RMSE' % (bw_testScore))
        logging.info('  Big Wind  Test Score: %.3f RMSE' % (bw_testScore))

        bw_npwScore = math.sqrt(mean_squared_error(bw_testY, bw_nwp_test))
        print('Big Wind npw Score : %.3f RMSE' % (bw_npwScore))
        logging.info('  Big Wind npw Score : %.3f RMSE' % (bw_npwScore))


    sw_testY = []
    sw_predictY = []
    sw_nwp_test = []
    for i in range(len(testY)):
        if testY[i] < 4:
            sw_testY.append(testY[i])    
            sw_predictY.append(testPredict[i])
            sw_nwp_test.append(test_nwpY[i])
    sw_testScore = 0
    sw_npwScore = 0
    if len(sw_testY) > 0:
        sw_testScore = math.sqrt(mean_squared_error(sw_testY, sw_predictY))
        print('Small Wind Test Score: %.3f RMSE' % (sw_testScore))
        logging.info('  Small Wind  Test Score: %.3f RMSE' % (sw_testScore))

        sw_npwScore = math.sqrt(mean_squared_error(sw_testY, sw_nwp_test))
        print('Small Wind npw Score : %.3f RMSE' % (sw_npwScore))
        logging.info('  Small Wind npw Score : %.3f RMSE' % (sw_npwScore))
    
    # plt.clf()
    # plt.plot(testY)    
    # plt.plot(test_nwpY)
    # plt.plot(testPredict)
    # plt.savefig(RESULTPATH + model_disc.site + model_disc.model_name + '.png')
    # plt.show()
    pass

def lstm_multifeature(model_disc: Lstm_model_disc):
 
    print('failed loading model')
    
    window, predict_hour, dataset, trainX, trainY, testX, testY, test_nwpY,  train_size, scalerNwp, scalerY = prepare_data(model_disc, 'residual')
    epoch = model_disc.epcoh
    print('training on', model_disc.model_name)
    numpy.random.seed(7)
    # create and fit the LSTM network
    model = Sequential()
    model.add(CuDNNLSTM(window, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(window/2)))
    model.add(Dense(int(window/4)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #print('trainX, trainY', trainX[0:10], trainY[0:10])
    #sys.exit()
    model.fit(trainX, trainY, epochs= epoch, batch_size= 128, verbose=2)
    model.save(MODEL_PATH + model_disc.site + model_disc.model_name  +'.h5')

    #show train results
    residual_Y = True
    train_results(model_disc, model, window, predict_hour, dataset, trainX, trainY, testX, testY, test_nwpY,  train_size, scalerNwp, scalerY, residual_Y)



def prepare_classification_data(model_disc: Classification_model_disc):
    dataframe = read_csv(model_disc.wind_tprh_nwp_file, usecols = ['wind','dir','slp','t2', 'rh2', 'td2', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp'],engine='python')
    
    dataset_len = model_disc.dataset_len
    dataset = dataframe.values[:]
    dataset = dataset.astype('float32')
    

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)


    dataframeY = read_csv(model_disc.wind_tprh_nwp_file, usecols=['wind'], engine='python')
    datasetY = dataframeY.values[:]
    datasetY = datasetY.astype('float32')
    scalerY = MinMaxScaler(feature_range=(0, 1))
    datasetY = scalerY.fit_transform(datasetY)
    
    dataset = dataset[:dataset_len]
    # split into train and test sets
    train_test_split = model_disc.train_test_split
    train_size = int(len(dataset) * train_test_split)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    wind_bar =  model_disc.wind_bar
    wind_bar = numpy.array([[wind_bar]])
    
    

    wind_bar = scalerY.transform(wind_bar)
    wind_bar = wind_bar[0][0]

    window = model_disc.window
    predict_hour = model_disc.predict_hour
    trainX, trainY= creat_classification_dataset(train, NWP_START_INDEX, window, predict_hour, wind_bar)
    testX, testY = creat_classification_dataset(test, NWP_START_INDEX, window, predict_hour, wind_bar)
    
    print('trainX, trainY, testX, testY',trainX.shape, trainY.shape, testX.shape, testY.shape)
    #sys.exit()
    return window, predict_hour, test, trainX, trainY, testX, testY, train_size


def lstm_classfier(model_disc: Lstm_model_disc):
    window, predict_hour, test, trainX, trainY, testX, testY, train_size = prepare_classification_data(model_disc)
    epoch = model_disc.epcoh
    print('training on', model_disc.model_name)
    numpy.random.seed(7)
    # create and fit the LSTM network
    model = Sequential()
    model.add(CuDNNLSTM(window, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(window/2)))
    model.add(Dense(int(window/4)))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision_m])
    model.fit(trainX, trainY, epochs= epoch, batch_size= 256, verbose=2)
    model.save(MODEL_PATH + model_disc.site + model_disc.model_name  +'.h5')
    scores = model.evaluate(testX, testY, verbose=0)
    
    testPredict = model.predict(testX)
    # plt.clf()
    # plt.plot(test[:,0])    
    # plt.plot(testPredict)
    
    #plt.show()



def load_classification_models_run(first_model_disc: Classification_model_disc):
    dataset_len = first_model_disc.dataset_len
    dataframe = read_csv(first_model_disc.wind_tprh_nwp_file, usecols = ['wind','dir','slp','t2', 'rh2', 'td2', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp'],engine='python')
    dataset = dataframe.values[:]
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    validation = dataset[dataset_len:, :]

    predict_start, predict_end = 1, 25    
    predict_classicfication = []
   
    model_predicts = []
    for predict_hour in range(predict_start, predict_end):        
        print('processing classification model ', predict_hour)
        window = first_model_disc.window
        dataset_len = first_model_disc.dataset_len
        epcoh = first_model_disc.epcoh        
        site = first_model_disc.site
        dropout = first_model_disc.dropout

        validationX, dummy = creat_classification_dataset(validation, 4, window, predict_hour)
        model_disc = Classification_model_disc(predict_hour, window, dataset_len, epcoh,dropout, site, first_model_disc.wind_bar)
        # split into train and test sets
        # print('testX_0',testX_0)
        model_predict = [[0]] * (window + predict_hour - 1)
        model_predict.extend(run_saved_classification_model(model_disc, validationX))
        model_predicts.append(model_predict)

    hour = 1
    for i in range(len(model_predicts[0])):

        predict = model_predicts[hour-1][i]
        predict_classicfication.append(predict)
        hour += 1
        if hour >= 25:
            hour = 1    

    return numpy.array(predict_classicfication)



def validation_model(first_model_disc: Lstm_model_disc, predictY, Ycolumn = 'wind',sw = [], bw = [], classification = []):

    dataset_len = first_model_disc.dataset_len
    site = first_model_disc.site
    window = first_model_disc.window
    

    dataframeY = read_csv(first_model_disc.wind_tprh_nwp_file, usecols = [Ycolumn],engine='python')
    datasetY = dataframeY.values[:]
    datasetY = datasetY.astype('float32')
    validationY = datasetY[dataset_len + window:, :]
    
    scalerClassifi = MinMaxScaler(feature_range=(0, 1))
    scalerClassifi.fit(datasetY)
    if len(classification) > 0:
        classification = scalerClassifi.inverse_transform(classification)
        classification = classification[window:]
    
    npwDataframe = read_csv(first_model_disc.wind_tprh_nwp_file, usecols=['nwp_wind'], engine='python')
    npwDataset = npwDataframe.values[:]
    npwDataset = npwDataset.astype('float32')
    npw_validation_Y = npwDataset[dataset_len + window:,:]

    predictY = predictY[window:]

    if Ycolumn == 'residual':
        validationY = validationY + npw_validation_Y
        #predictY = predictY / 2
        predictY = predictY + npw_validation_Y
        print('ycolumn si ', Ycolumn)

    if len(sw) > 0:
        sw = sw[window:]
    
    if len(bw) > 0:
        bw = bw[window:]
    
    print('predictY, validationY, nwpY', predictY.shape, validationY.shape, npw_validation_Y.shape)
    
    logging.info('dealing' + site)
    testScore = math.sqrt(mean_squared_error(validationY, predictY))
    print('Test Score: %.3f RMSE' % (testScore))
    logging.info('  Test Score: %.3f RMSE' % (testScore))

    npwScore = math.sqrt(mean_squared_error(validationY, npw_validation_Y))
    print('npw Score : %.3f RMSE' % (npwScore))
    logging.info('  npw Score : %.3f RMSE' % (npwScore))
    
    bw_testY = []
    bw_predictY = []
    bw_nwp_test = []

    sw_testY = []
    sw_predictY = []
    sw_nwp_test = []
    

    for i in range(len(validationY)):
        if validationY[i][0] > 4:
            bw_testY.append(validationY[i][0])    
            bw_predictY.append(predictY[i][0])
            bw_nwp_test.append(npw_validation_Y[i][0])
        else:
            sw_testY.append(validationY[i][0])    
            sw_predictY.append(predictY[i][0])
            sw_nwp_test.append(npw_validation_Y[i][0])
    bw_testScore = 0
    bw_npwScore = 0
    sw_testScore = 0
    sw_npwScore = 0
    if len(bw_testY) > 0:
        bw_testScore = math.sqrt(mean_squared_error(bw_testY, bw_predictY))
        print('Big Wind Test Score: %.3f RMSE' % (bw_testScore))
        logging.info('  Big Wind  Test Score: %.3f RMSE' % (bw_testScore))

        bw_npwScore = math.sqrt(mean_squared_error(bw_testY, bw_nwp_test))
        print('Big Wind npw Score : %.3f RMSE' % (bw_npwScore))
        logging.info('  Big Wind npw Score : %.3f RMSE' % (bw_npwScore))

        sw_testScore = math.sqrt(mean_squared_error(sw_testY, sw_predictY))
        print('Small Wind Test Score: %.3f RMSE' % (sw_testScore))
        logging.info('  Small Wind  Test Score: %.3f RMSE' % (sw_testScore))

        sw_npwScore = math.sqrt(mean_squared_error(sw_testY, sw_nwp_test))
        print('Small Wind npw Score : %.3f RMSE' % (sw_npwScore))
        logging.info('  Small Wind npw Score : %.3f RMSE' % (sw_npwScore))



    
    plt.clf()
    legends = []
    plt.figure(figsize=(16,9))
    line_testY, = plt.plot(validationY, label = 'Observation')
    line_npwTest, = plt.plot(npw_validation_Y, label = 'NWP Model Prediction')
    legends.extend([line_testY, line_npwTest])
    if len(sw) > 0:
        line_sw_predictY, = plt.plot(sw, label = 'Deep Learning Model Prediction SW')   
        legends.append(line_sw_predictY)
    if len(bw) > 0:
        line_bw_predictY, = plt.plot(bw, label = 'Deep Learning Model Prediction BW') 
        legends.append(line_bw_predictY)
    
    if len(classification) > 0:
        line_classification_predictY, = plt.plot(classification, label = 'Deep Learning Model Prediction Classification') 
        legends.append(line_classification_predictY)
    line_predictY, = plt.plot(predictY, label = 'Deep Learning Model Prediction')
    legends.append(line_predictY)
    
    plt.xlabel('Hours')
    plt.ylabel('Wind-speed')
    plt.legend(handles = legends)
    plt.savefig(RESULTPATH + first_model_disc.site + first_model_disc.model_name+'24modelResult.png')
    #plt.show()
    return testScore, npwScore, bw_testScore, bw_npwScore, sw_testScore, sw_npwScore
    

def load_models_and_run(first_model_disc: Lstm_model_disc, Ycolumn = 'wind'):
    dataset_len = first_model_disc.dataset_len
    #dataframe = read_csv(model_disc.wind_tprh_file, usecols=['wind','dir','slp','t2', 'rh2', 'td2'], engine='python')     
    dataframe = read_csv(first_model_disc.wind_tprh_nwp_file, usecols = ['wind','dir','slp','t2', 'rh2', 'td2', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp', 'residual'],engine='python')
    dataset = dataframe.values[:]
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    validation = dataset[dataset_len:, :]

    datesFrame =  read_csv(first_model_disc.wind_tprh_nwp_file, usecols = ['data-time'],engine='python')
    datesDataset = datesFrame.values[:]

    predict_start, predict_end = 1, 25    
    predictY = []

    model_predicts = []
    for predict_hour in range(predict_start, predict_end):        
        print('processing model ', predict_hour)
        #testX, dummy = create_multifeature_dataset(test, window, predict_hour)
        window = first_model_disc.window
        dataset_len = first_model_disc.dataset_len
        epcoh = first_model_disc.epcoh        
        site = first_model_disc.site
        dropout = first_model_disc.dropout
        Ycolumn_idx = 0
        if Ycolumn == 'residual':
            Ycolumn_idx = 14
        validationX, dummy, dummy2 = create_multifeature_nwp_dataset(validation, NWP_START_INDEX, window, predict_hour, Ycolumn_idx)
        model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh,dropout, site, first_model_disc.min_wind, first_model_disc.max_wind)
        # split into train and test sets
        # print('testX_0',testX_0)
        model_predict = [[0]] * (window + predict_hour - 1)
        model_predict.extend(run_saved_model(model_disc, validationX, Ycolumn))
        model_predicts.append(model_predict)
    
    hour = 1
    for i in range(len(model_predicts[0])):

        predict = model_predicts[hour-1][i]
        predictY.append(predict)
        hour += 1
        if hour >= 25:
            hour = 1


        
    return  numpy.array(predictY)


def load_and_validate_model(first_wind_model: Lstm_model_disc, newlyTrained, Ycolumn = 'wind'):
    if not newlyTrained:
        try:
            predictY = numpy.load(RESULTPATH + first_wind_model.site + 'tmp/' +first_wind_model.model_name + 'predictY' + '.npy')
        except:
            predictY = load_models_and_run(first_wind_model, Ycolumn)
            numpy.save(RESULTPATH + first_wind_model.site + 'tmp/' +first_wind_model.model_name + 'predictY', predictY) 
    else:
        predictY = load_models_and_run(first_wind_model, Ycolumn)
        numpy.save(RESULTPATH + first_wind_model.site + 'tmp/' +first_wind_model.model_name + 'predictY', predictY) 
        
    testScore, npwScore, bw_testScore, bw_npwScore, sw_testScore, sw_npwScore = validation_model(first_wind_model, predictY, Ycolumn)
    return first_wind_model.site, testScore, npwScore, bw_testScore, bw_npwScore, sw_testScore, sw_npwScore


def combin_two_models(small_wind_model: Lstm_model_disc, big_wind_model:Lstm_model_disc, classifcation_model: Classification_model_disc):

    try:
        sw_predictY = numpy.load(RESULTPATH + small_wind_model.site + 'tmp/' +small_wind_model.model_name + 'predictY' + '.npy')
    except:
        sw_predictY = load_models_and_run(small_wind_model)
        numpy.save(RESULTPATH + small_wind_model.site + 'tmp/'+ small_wind_model.model_name + 'predictY', sw_predictY) 
    

    try:
        bw_predictY = numpy.load(RESULTPATH + big_wind_model.site + 'tmp/' +big_wind_model.model_name + 'predictY' + '.npy')
    except:
        bw_predictY = load_models_and_run(big_wind_model)
        numpy.save(RESULTPATH + big_wind_model.site + 'tmp/' +big_wind_model.model_name + 'predictY', bw_predictY) 
    
    try:
        predict_classicfication = numpy.load(RESULTPATH + classifcation_model.site + 'tmp/' +classifcation_model.model_name + 'predictY' + '.npy')
    except :
        predict_classicfication = load_classification_models_run(classifcation_model)
        numpy.save(RESULTPATH + classifcation_model.site + 'tmp/' +classifcation_model.model_name + 'predictY', predict_classicfication) # protocol 0 is printable ASCII

    
    
    predictY = []

    for i in range(len(predict_classicfication)):
        sw = sw_predictY[i]
        bw = bw_predictY[i]
        classicfication = predict_classicfication[i]
        wind = 0
        if classicfication < 0.8:
            wind = sw
        else:
            wind = sw * (1-classicfication) + bw * classicfication
        predictY.append(wind)

    predictY = numpy.array(predictY)
    validation_model(small_wind_model, predictY, sw_predictY, bw_predictY, predict_classicfication)

    pass



def read_dataframe(csvfile, datastart, dataend, ycolumn = 'wind'):
    print('reawding', csvfile)
    dataframe = read_csv(csvfile, usecols = CSV_COLUMNS,engine='python')
    
    dataset = dataframe.values[:]
    dataset = dataset.astype('float32')
    
    dataframeY = read_csv(csvfile, usecols=[ycolumn], engine='python')
    datasetY = dataframeY.values[:]
    datasetY = datasetY.astype('float32')

    if dataend == 'end':
        dataset = dataset[datastart:, :]
        datasetY = datasetY[datastart:, :]
    else:   
        dataset = dataset[datastart:dataend, :]
        datasetY = datasetY[datastart:dataend, :]

    return dataset, datasetY

def prepare_ma_data(ma_model_disc:Moving_ave_model_disc, datastart, dataend, train_test_split):

    dataset, wind_obs = read_dataframe(ma_model_disc.wind_tprh_nwp_file, datastart, dataend, 'wind')
    scaler, scalerY = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    wind_obs= scalerY.fit_transform(wind_obs)

  
    train_size = int(len(dataset) * train_test_split)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    window = ma_model_disc.window
    predict_hour = ma_model_disc.predict_hour
    trainX, trainY, dummy = create_multifeature_nwp_dataset(train, NWP_START_INDEX, window, predict_hour)
    testX, testY, dummy = create_multifeature_nwp_dataset(test, NWP_START_INDEX, window, predict_hour)
    
    #cal moving average and ma_residual
    ma_window = ma_model_disc.ma_window

    trainY, trainY_residual = moving_average(trainY, ma_window)
    testY, testY_residual = moving_average(testY, ma_window)
    
    trainX, testX = trainX[ma_window-1:], testX[ma_window-1:]

    plt.clf()
    plt.plot(testY)
    plt.plot(testY_residual)
    #plt.show()
    print(testY.shape)
    
    return  trainX, trainY, trainY_residual,testX, testY,testY_residual, scalerY
    

def train_ma_model(ma_model_disc:Moving_ave_model_disc, trainX, trainY):
    epoch = ma_model_disc.epcoh
    numpy.random.seed(7)
    # create and fit the LSTM network

    
    print('training on', ma_model_disc.model_name)
    #sys.exit()
    window = ma_model_disc.window
    model = Sequential()
    model.add(CuDNNLSTM(200, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(100)))
    #model.add(Dense(int(50)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #print('trainX, trainY', trainX[0:10], trainY[0:10])
    #sys.exit()
    model.fit(trainX, trainY, epochs= epoch, batch_size= 128, verbose=2)
    model.save(MODEL_PATH + ma_model_disc.site + ma_model_disc.model_name  +'.h5')
    return model

def train_ma_res_model(ma_res_model_disc:Ma_res_model_disc, trainX, trainY):
    epoch = ma_res_model_disc.epcoh
    numpy.random.seed(7)
    # create and fit the LSTM network

    print('trainX, trainY', trainX[0], trainY[0])
    print('training on', ma_res_model_disc.model_name)
    #sys.exit()
    window = ma_res_model_disc.window
    model = Sequential()
    model.add(CuDNNLSTM(200, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(200/2)))
    #model.add(Dense(int(50/4)))
    model.add(Dense(1))

    # model.add(LSTM(200, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    # model.add(RepeatVector(1))
    # model.add(LSTM(200, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(100, activation='relu')))
    # model.add(TimeDistributed(Dense(1)))

    model.compile(loss='mean_squared_error', optimizer='adam')
    #print('trainX, trainY', trainX[0:10], trainY[0:10])
    #sys.exit()

    #trainY = trainY.reshape(trainX.shape[0], 1, 1)
    model.fit(trainX, trainY, epochs= epoch, batch_size= 128, verbose=2)
    model.save(MODEL_PATH + ma_res_model_disc.site + ma_res_model_disc.model_name  +'.h5')
    return model

def testPerform(mmodel_disc:Lstm_model_disc, model: Sequential, testX, testY, scalerY):
    predictY = model.predict(testX)

    predictY = predictY.reshape(predictY.shape[0],1)
    print('predictY before inverse', predictY.shape)
    if scalerY is not None:
        predictY = scalerY.inverse_transform(predictY)

    testY = testY.reshape(predictY.shape[0],1)
    print('testY before inverse', testY.shape)
    if scalerY is not None:
        testY = scalerY.inverse_transform(testY)
    #testY = numpy.transpose(testY)

    print('predicty, testY', predictY.shape, testY.shape)
    testScore = math.sqrt(mean_absolute_error(testY, predictY))
    print(mmodel_disc.model_name,'Test Score: %.3f MAE' % (testScore))

    testScore = math.sqrt(mean_squared_error(testY, predictY))
    print(mmodel_disc.model_name,'Test Score: %.3f RMSE' % (testScore))

    return testY, predictY

def load_24_ma_maRes_models_and_run(first_ma_model_disc: Moving_ave_model_disc, first_ma_res_model:Ma_res_model_disc):

    predict_start, predict_end = 1, 25
    predictY = []

    model_24_predicts = []

    ground_truth = None
    for predict_hour in range(predict_start, predict_end):
        print('processing model', predict_hour)        
        # validation_start = first_ma_model_disc.dataset_len
        # validation_end = -1
        #validation_split = 0
        validation_start = TRAIN_TEST_SIZE
        validation_end = -1
        validation_split = 0

        ma_model_disc = first_ma_model_disc
        ma_model_disc.set_predict_hour(predict_hour)

        ma_res_model_disc = first_ma_res_model
        ma_res_model_disc.set_predict_hour(predict_hour)

        trainX, trainY, trainY_residual, validationX, validationY, validationY_residual, scalerY = prepare_ma_data(ma_model_disc,validation_start,validation_end, validation_split)

        if predict_hour == 1:
            ground_truth = validationY + validationY_residual
            ground_truth = scalerY.inverse_transform([ground_truth])
            ground_truth = numpy.transpose(ground_truth)

        window = first_ma_model_disc.window
        model_predict = [[0]] * (predict_hour - 1)
        
        # ma_pred = run_saved_model(ma_model_disc, validationX)
        # ma_res_pred = run_saved_model(ma_res_model_disc, validationX)

        m1 = keras.models.load_model(MODEL_PATH + ma_model_disc.site + ma_model_disc.model_name + '.h5')
        m2 = keras.models.load_model(MODEL_PATH + ma_res_model_disc.site + ma_res_model_disc.model_name + '.h5')
        ma_truth, ma_pred = testPerform(ma_model_disc, m1,validationX,validationY,scalerY)
        ma_res_truth, ma_res_pred = testPerform(ma_res_model_disc, m2,validationX,validationY_residual,scalerY)

        test_ma_maRes_combined(ma_truth, ma_pred, ma_res_truth, ma_res_pred)


        ma_maRes_pred = ma_pred + ma_res_pred
        model_predict.extend(ma_maRes_pred)
        model_24_predicts.append(model_predict)
    
    hour = 1
    for i in range(len(model_24_predicts[0])):

        #predict = model_predicts[hour-1][i]
        predict = model_24_predicts[hour-1][i]
        predictY.append(predict)
        hour += 1
        if hour >= 25:
            hour = 1


    finalscore = math.sqrt(mean_squared_error(predictY, ground_truth))
    print('Test Score: %.3f RMSE' % (finalscore))

    plt.clf()
    plt.plot(ground_truth)
    plt.plot(predictY)
    plt.show()
    
    
    return  numpy.array(predictY), ground_truth

def train_24_ma_maRes_models(site, window, epcoh, dropout, ma_window, dataset_len, forceTrain):
    if forceTrain:

        first_model = None

        for predict_hour in range(1,25):        
            dataset_len = TRAIN_TEST_SIZE
            ma_model_disc = Moving_ave_model_disc(predict_hour, window, dataset_len, epcoh, dropout, site, ma_window)
            ma_res_model_disc = Ma_res_model_disc(predict_hour, window, dataset_len, epcoh, dropout, site, ma_window)
            if predict_hour == 24:
                first_ma_model = ma_model_disc
                first_ma_res_model =ma_res_model_disc
            
            ma_model(ma_model_disc)
            ma_res_model(ma_res_model_disc)
    
        return first_ma_model, first_ma_res_model , True
    
    else:
        try:
            ma_model_disc = Moving_ave_model_disc(1, window, dataset_len, epcoh, dropout, site, ma_window)
            ma_res_model_disc = Ma_res_model_disc(1, window, dataset_len, epcoh, dropout, site, ma_window)
            #return first_model
            
            keras.models.load_model(MODEL_PATH + ma_model_disc.site + ma_model_disc.model_name + '.h5')
            keras.models.load_model(MODEL_PATH + ma_res_model_disc.site + ma_res_model_disc.model_name + '.h5')

            return ma_model_disc, ma_res_model_disc, False
            
        except:

            first_model = None

            for predict_hour in range(1,25):        
                dataset_len = TRAIN_TEST_SIZE
                ma_model_disc = Moving_ave_model_disc(predict_hour, window, dataset_len, epcoh, dropout, site, ma_window)
                ma_res_model_disc = Ma_res_model_disc(predict_hour, window, dataset_len, epcoh, dropout, site, ma_window)
                if predict_hour == 1:
                    first_ma_model = ma_model_disc
                    first_ma_res_model =ma_res_model_disc
                
                ma_model(ma_model_disc)
                ma_res_model(ma_res_model_disc)
        
            return first_ma_model, first_ma_res_model , True


def ma_res_model(ma_res_model_disc:Ma_res_model_disc):
    train_test_size = ma_res_model_disc.dataset_len
    train_test_split = ma_res_model_disc.train_test_split
    trainX, trainY, trainY_residual,testX, testY,testY_residual, scalerY = prepare_ma_data(ma_res_model_disc, 0, train_test_size, train_test_split)

    model = train_ma_res_model(ma_res_model_disc, trainX, trainY_residual)

    testY, predictY =testPerform(ma_res_model_disc, model, testX, testY_residual, scalerY)
    return testY, predictY

def ma_model(ma_model_disc:Moving_ave_model_disc):
    
    train_test_size = ma_model_disc.dataset_len
    train_test_split = ma_model_disc.train_test_split
    trainX, trainY, trainY_residual,testX, testY,testY_residual, scalerY = prepare_ma_data(ma_model_disc, 0, train_test_size, train_test_split)

    model = train_ma_model(ma_model_disc, trainX, trainY)

    testY, predictY = testPerform(ma_model_disc, model, testX, testY, scalerY)
    return testY, predictY


def test_ma_maRes_combined(ma_testY, ma_predY, ma_res_testY,  ma_res_predY):
    wind_obs = ma_testY + ma_res_testY
    wind_pred = ma_predY + ma_res_predY

    testScore = math.sqrt(mean_absolute_error(wind_obs, wind_pred))
    print('Test Score: %.3f MAE' % (testScore))

    testScore = math.sqrt(mean_squared_error(wind_obs, wind_pred))
    print('Test Score: %.3f RMSE' % (testScore))

    plt.clf()
    plt.plot(wind_obs)
    plt.plot(wind_pred)
    plt.show()



def create_seq2seq_dataset(dataset, nwp_start, window, look_forward_window = 24, Ycolumn_idx = 0, obs_end = 6, nwp_end = 7):
    dataX, dataY = [], []
    i = window
    while i < len(dataset)- look_forward_window + 1:
        predict_start_hour = i
        predict_end_hour = predict_start_hour + look_forward_window
        wind_24_hour = []
        
        for hour in range(predict_start_hour, predict_end_hour):
            wind = dataset[hour, Ycolumn_idx]
            wind_24_hour.append(wind)

        dataY.append(wind_24_hour)
        
        nwp_predict = dataset[predict_end_hour - window: predict_end_hour, nwp_start: nwp_end]
        X = dataset[(i-window):(i), [0,2,3]]
        
        obs_nwp_featue_list = []
        
        row = 0 
        while row < window:
            X_part = X[row]
            nwp_part = nwp_predict[row]
            total_row = numpy.append(X_part, nwp_part)
            obs_nwp_featue_list.append(total_row)
            row += 1

        obs_nwp_featue = numpy.array(obs_nwp_featue_list)
        dataX.append(obs_nwp_featue)

        i += 1

    dataX = numpy.array(dataX)
    dataY = numpy.array(dataY)
    print('dataX, dataY', dataX.shape, dataY.shape)
    return numpy.array(dataX), numpy.array(dataY)

def prepare_seq2seq_data(model_disc:Lstm_model_disc, datastart, dataend, train_test_split):
    dataset, wind_obs = read_dataframe(model_disc.wind_tprh_nwp_file, datastart, dataend, 'wind')
    scaler, scalerY = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    wind_obs= scalerY.fit_transform(wind_obs)

  
    train_size = int(len(dataset) * train_test_split)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    window = model_disc.window
    trainX, trainY = create_seq2seq_dataset(train, NWP_START_INDEX, window, model_disc.look_forward)
    testX, testY = create_seq2seq_dataset(test, NWP_START_INDEX, window, model_disc.look_forward)
    
    return  trainX, trainY, testX, testY, scalerY
    

def prepare_seq2seq2_val_data(model_disc:Lstm_model_disc, datastart, dataend):
    dataset, wind_obs = read_dataframe(model_disc.wind_tprh_nwp_file, datastart, dataend, 'wind')
    scaler= MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    dataset = dataset[datastart:dataend]
    pass

def train_seq2seq_model(model_disc:Lstm_model_disc, trainX, trainY):

    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
	# reshape output into [samples, timesteps, features]
    trainY = trainY.reshape((trainY.shape[0], trainY.shape[1], 1))
	# define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs= model_disc.epcoh , batch_size = 128, verbose= 1)
    return model

def prepare_val_seq2seq_data(window, look_forward, testX):
    continue_testX = []
    size = testX.shape[0]

    i = 0 
    while i + window + look_forward < size:
        X = testX[i]
        continue_testX.append(X)

        i += look_forward


    start_idx = window
    end_idx = start_idx + look_forward * len(continue_testX)
    return numpy.array(continue_testX), start_idx, end_idx

def val_seq2seq(model_disc, model, validationX, validationY, scalerY):

    validationX, start, end  = prepare_val_seq2seq_data(model_disc.window, model_disc.look_forward, validationX)
    validationY, start, end = prepare_val_seq2seq_data(model_disc.window, model_disc.look_forward, validationY)

    predictY = model.predict(validationX)

    predictY = predictY.flatten()
    validationY = validationY.flatten()

    predictY = predictY.reshape((len(predictY),1))
    validationY = validationY.reshape((len(validationY),1))

    predictY = scalerY.inverse_transform(predictY)
    validationY = scalerY.inverse_transform(validationY)

    finalscore = math.sqrt(mean_squared_error(predictY, validationY))
    print('Test Score: %.3f RMSE' % (finalscore))

    plt.clf()
    plt.plot(validationY)
    plt.plot(predictY)
    plt.show()

    pass

def seq2seqModel(model_disc:Lstm_model_disc):
    trainX, trainY, testX, testY, scalerY = prepare_seq2seq_data(model_disc, 0, TRAIN_TEST_SIZE, 0.67)
    model = train_seq2seq_model(model_disc, trainX, trainY)

    dummy, dummy, validationX, validationY, dummy = prepare_seq2seq_data(model_disc, TRAIN_TEST_SIZE, -1, 0)
    val_seq2seq(model_disc, model, validationX, validationY, scalerY)
    pass



def train_test_validation_split(dataset, test_start, validation_start):
    data_train, data_test, data_val = dataset[0: test_start, :], dataset[test_start: validation_start, :], dataset[validation_start:, :]
    return data_train, data_test, data_val

def prepare_emd_data(first_emd_model_disc:Emd_model_disc):
    dataset, datasetY = read_dataframe(first_emd_model_disc.wind_tprh_nwp_file, 0, 'end')
    xticts = read_csv(first_emd_model_disc.wind_tprh_nwp_file, usecols=['data-time'], engine='python')
    xticts = xticts.values[:,0]
    # datasetY_tmp = datasetY[:1201]
    # xticts_tmp = xticts[0:1201,0]
    # plotlines([datasetY_tmp], ['Observation'], xlabel='Date-Time', ylabel='Wind speed (m/s)', xticks= xticts_tmp, xtick_space = 168, savepath= './wind_obs'+'.png')
    # sys.exit()

    scaler = MinMaxScaler(feature_range=(0, 1))
    
    dataset = scaler.fit_transform(dataset)
   
    scalerY =  MinMaxScaler(feature_range=(0, 1))
    datasetY = scalerY.fit_transform(datasetY)
    

    dummy, nwp = read_dataframe(first_emd_model_disc.wind_tprh_nwp_file, 0, 'end', ycolumn='nwp_wind')
    scalerNwp = MinMaxScaler(feature_range=(0, 1))
    nwp = scalerNwp.fit_transform(nwp)
    #cal_correlation(datasetY[:,0], nwp[ : ,0])
    # imfs_res = emd_detrend(datasetY)
    # nwp_imfs = emd_detrend(nwp)

    # sys.exit()
    imfs_res = vmd_detrend(datasetY)

    # datas = [[datasetY[:,0].tolist()]]
    # legends = [['Observation']]

    # for i in range(imfs_res.shape[1]):
    #     data = [imfs_res[:,i].tolist()]
    #     lengend = ['imfs_'+ str(i)]
    #     datas.append(data)
    #     legends.append(lengend)

    # plotlines_multifigue(datas, legends, xlabel= 'Date-Time', ylabel= 'Wind speed (m/s)', xticks= xticts, xtick_space= 336, display_lenth= 1201, show = True, figsize=(12,9), savepath='./obs_imfs.png')
    # # sys.exit()
    nwp_imfs = vmd_detrend(nwp)


    test_start = int(first_emd_model_disc.dataset_len * first_emd_model_disc.train_test_split)
    validation_start = first_emd_model_disc.dataset_len

    imfs_res_te = vmd_detrend(datasetY[0:test_start])
    
    plotlines([imfs_res[:100,1], imfs_res_te[:100,1]], ['a', 'b'], show = True)
    sys.exit()
    dataset_tr, dataset_te, dataset_val = train_test_validation_split(dataset, test_start, validation_start)

    imfs_tr, imfs_te, imfs_val = train_test_validation_split(imfs_res, test_start, validation_start)
    nwp_imfs_tr, nwp_imfs_te, nwp_imfs_val = train_test_validation_split(nwp_imfs, test_start, validation_start)


    return dataset_tr, dataset_te, dataset_val, nwp_imfs_tr, nwp_imfs_te, nwp_imfs_val, imfs_tr, imfs_te, imfs_val, scalerY, scalerNwp


def load_model(model_disc):

    try: 
        model = keras.models.load_model(MODEL_PATH + model_disc.site + model_disc.model_name + '.h5')
    except:
        print('loading ' + model_disc.model_name + 'failed')
        model = None    

    return model

def train_emd_model(model_disc, trainX, trainY, forceTraind = False):
    newlyTrained = False
    # create and fit the LSTM network
    if not forceTraind:
        model = load_model(model_disc)
        if model is not None:
            return model, newlyTrained
        

    epoch = model_disc.epcoh
    numpy.random.seed(7)
    print('training on', model_disc.model_name)
    #sys.exit()
    model = Sequential()
    model.add(CuDNNLSTM(200, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(int(100)))
    #model.add(Dense(int(50)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #print('trainX, trainY', trainX[0:10], trainY[0:10])
    #sys.exit()
    model.fit(trainX, trainY, epochs= epoch, batch_size= 128, verbose=1)
    print('saving', MODEL_PATH + model_disc.site + model_disc.model_name  +'.h5')
    model.save(MODEL_PATH + model_disc.site + model_disc.model_name  +'.h5')
    newlyTrained = True
    return model, newlyTrained

def random_float(low, high):
    return random.random()*(high-low) + low


def append_component(dataset, components, component_id):
    print('dataset', dataset.shape)
    component = components[:, component_id]
    component = component.reshape((component.shape[0],1))
    re = numpy.append(dataset, component, axis=1)

    print('re',re.shape)
    return re
def train_merge_models(merge_model_disc, trainX, trainY, forceTrain = False):
    if not forceTrain:
        print('loading merge model')
        model = load_model(merge_model_disc)

        if model is not None:
            print('loading merge model succeed')
            return model
    
    print('loading merge model failed')
    model = Sequential()
    model.add(Dense(200, input_dim=trainX.shape[1], activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs= 200, batch_size= 128, verbose=1)
    print('saving', MODEL_PATH + merge_model_disc.site + merge_model_disc.model_name  +'.h5')
    model.save(MODEL_PATH + merge_model_disc.site + merge_model_disc.model_name  +'.h5')
    return model
    
def save_np_array(model_disc, tosave, name):
    print('saving ', RESULTPATH + model_disc.site + 'tmp/' +model_disc.model_name  +'_' + name + '.npy')
    numpy.save(RESULTPATH + model_disc.site + 'tmp/' +model_disc.model_name  +'_' + name + '.npy', tosave) 

def load_np_array(model_disc, name):
    try:
        re = numpy.load(RESULTPATH + model_disc.site + 'tmp/' +model_disc.model_name  +'_' + name + '.npy')
        return re
    except:
        print('loading ' + RESULTPATH + model_disc.site + 'tmp/' +model_disc.model_name  +'_' + name + '.npy' + ' failed')
        return None

def comb_24_emd_models(first_emd_model_disc:Emd_model_disc):
    obs_val = None
    nwp_val = None
    merge_24_pred_val = []
    for predict_hour in range(1,25):

        emd_model = copy.copy(first_emd_model_disc)
        emd_model.set_predict_hour(predict_hour)

        print(emd_model.model_name)

        merge_pred_val = load_np_array(emd_model, 'merge_pred_val')
        merge_pred_val = numpy.insert(merge_pred_val, 0, [0] * (predict_hour - 1))
        
        merge_24_pred_val.append(merge_pred_val)
        if predict_hour == 1:
            obs_val = load_np_array(emd_model, 'obs_val')   
            nwp_val = load_np_array(emd_model, 'nwp_val')
    
    merge_24_pred_val = np.array(merge_24_pred_val)
    merge_24_pred_val = merge_24_pred_val.T

    final_24_val = []
    predict_hour = 1
    
    for idx in range(len(obs_val)):
        best_predict = merge_24_pred_val[idx, predict_hour -1]
        final_24_val.append(best_predict)
        predict_hour += 1
        if predict_hour == 25:
            predict_hour = 1

    final_24_val = np.array(final_24_val).T

    # print('obs_val', obs_val[:10])
    # print('pre_val', final_24_val[:10])
    # sys.exit()

    score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up = compare_nwp_dpl(obs_val, nwp_val, final_24_val)
    #plotlines([obs_val, final_24_val, nwp_val ], ['obs', 'dpl', 'nwp'], savepath=RESULTPATH + first_emd_model_disc.site + 'vmd_combined.png')
    return score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up, final_24_val


def val_emd_models(first_emd_model_disc:Emd_model_disc, newlyTrained_emd = False):
    merge_pred_val = load_np_array(first_emd_model_disc, 'merge_pred_val')
    nwp_val = load_np_array(first_emd_model_disc, 'nwp_val')
    obs_val = load_np_array(first_emd_model_disc, 'obs_val')

    if merge_pred_val is None or nwp_val is None or obs_val is None or newlyTrained_emd:
        dummy, dummy, dataset_val, dummy, dummy, nwp_imfs_val, dummy, dummy, imfs_val, scalerY, scalerNwp = prepare_emd_data(first_emd_model_disc)

        pred_val_results = None
        for imfs_id in range(imfs_val.shape[1]):
            model_disc = copy.copy(first_emd_model_disc)
            model_disc.set_immfs_idx(imfs_id)

            emd_model = load_model(model_disc)

            dateset_val_app_comp = append_component(dataset_val, imfs_val, imfs_id)
            dateset_val_app_comp = append_component(dateset_val_app_comp, nwp_imfs_val, imfs_id)
            valX, dummy, dummy = create_multifeature_nwp_dataset(dateset_val_app_comp, NWP_START_INDEX, model_disc.window, model_disc.predict_hour, nwp_end=8, Ycolumn_idx=dateset_val_app_comp.shape[1] -2 )
            #dummy, pred_val = testPerform(model_disc, emd_model, valX, valY, scalerY = None)
            pred_val = emd_model.predict(valX)
            pred_val = pred_val.reshape(pred_val.shape[0],1)

            if imfs_id == 0:
                pred_val_results =pred_val
            else:
                pred_val_results = numpy.append(pred_val_results, pred_val, axis=1)
                print('pred_val_results', pred_val_results.shape)


        merge_model_disc = Merge_model_disc(first_emd_model_disc.predict_hour, first_emd_model_disc.window, first_emd_model_disc.dataset_len, first_emd_model_disc.epcoh,  site = first_emd_model_disc.site,  train_test_split=first_emd_model_disc.train_test_split)
        merge_model =  load_model(merge_model_disc)

        dummy, obs_val, nwp_val = create_multifeature_nwp_dataset(dataset_val, NWP_START_INDEX, first_emd_model_disc.window, first_emd_model_disc.predict_hour)

        obs_val, merge_pred_val = testPerform(merge_model_disc, merge_model, pred_val_results, obs_val, scalerY)
        
        nwp_val = scalerNwp.inverse_transform([nwp_val])
        nwp_val = numpy.transpose(nwp_val)
        save_np_array(first_emd_model_disc, merge_pred_val, 'merge_pred_val')
        save_np_array(first_emd_model_disc, nwp_val, 'nwp_val')
        save_np_array(first_emd_model_disc, obs_val, 'obs_val')


    score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up = compare_nwp_dpl(obs_val, nwp_val, merge_pred_val)
    return score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up


def update_imfs():
    pass

def run_emd_model_on_one_newday(first_emd_model_disc, today_obs, tomorrow_nwp):
    dummy, dataset_te, dataset_val, dummy, nwp_imfs_te, nwp_imfs_val, dummy, imfs_te, imfs_val, scalerY, scalerNwp = prepare_emd_data(first_emd_model_disc)

    pred_val_results = None
    for imfs_id in range(imfs_val.shape[1]):
        model_disc = copy.copy(first_emd_model_disc)
        model_disc.set_immfs_idx(imfs_id)

        emd_model = load_model(model_disc)

        dateset_val_app_comp = append_component(dataset_val, imfs_val, imfs_id)
        dateset_val_app_comp = append_component(dateset_val_app_comp, nwp_imfs_val, imfs_id)
        valX, dummy, dummy = create_multifeature_nwp_dataset(dateset_val_app_comp, NWP_START_INDEX, model_disc.window, model_disc.predict_hour, nwp_end=8, Ycolumn_idx=dateset_val_app_comp.shape[1] -2 )
        #dummy, pred_val = testPerform(model_disc, emd_model, valX, valY, scalerY = None)
        pred_val = emd_model.predict(valX)
        pred_val = pred_val.reshape(pred_val.shape[0],1)

        if imfs_id == 0:
            pred_val_results =pred_val
        else:
            pred_val_results = numpy.append(pred_val_results, pred_val, axis=1)
            print('pred_val_results', pred_val_results.shape)


    merge_model_disc = Merge_model_disc(first_emd_model_disc.predict_hour, first_emd_model_disc.window, first_emd_model_disc.dataset_len, first_emd_model_disc.epcoh,  site = first_emd_model_disc.site,  train_test_split=first_emd_model_disc.train_test_split)
    merge_model =  load_model(merge_model_disc)

    dummy, obs_val, nwp_val = create_multifeature_nwp_dataset(dataset_val, NWP_START_INDEX, first_emd_model_disc.window, first_emd_model_disc.predict_hour)

    obs_val, merge_pred_val = testPerform(merge_model_disc, merge_model, pred_val_results, obs_val, scalerY)

    score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up = compare_nwp_dpl(obs_val, nwp_val, merge_pred_val)


#train emdModels and essemble emd models with a dense layer
def emdModels(first_emd_model_disc:Emd_model_disc, forceTraind = False):
    dataset_tr, dummy, dummy, nwp_imfs_tr, dummy, dummy, imfs_tr, dummy, dummy, dummy, dummy = prepare_emd_data(first_emd_model_disc)

    pred_tr_results = None
    one_newlyTrained = False
    
    xticts = read_csv(first_emd_model_disc.wind_tprh_nwp_file, usecols=['data-time'], engine='python')
    xticts = xticts.values[:,0]

    obs_tr = dataset_tr[:, 0]

    dummy, trainY, dummy = create_multifeature_nwp_dataset(dataset_tr, NWP_START_INDEX, first_emd_model_disc.window, first_emd_model_disc.predict_hour)
    datas = [[trainY]]
    legends = [['Wind observation']]
    for imfs_id in range(imfs_tr.shape[1]):
        model_disc = copy.copy(first_emd_model_disc)
        model_disc.set_immfs_idx(imfs_id)
        dateset_tr_app_comp = append_component(dataset_tr, imfs_tr, imfs_id)
        dateset_tr_app_comp = append_component(dateset_tr_app_comp, nwp_imfs_tr, imfs_id)

        trainX, trainY, dummy = create_multifeature_nwp_dataset(dateset_tr_app_comp, NWP_START_INDEX, model_disc.window, model_disc.predict_hour, nwp_end=8, Ycolumn_idx=dateset_tr_app_comp.shape[1] -2 )

        print('trainX shape, trainY shape', trainX.shape, trainY.shape)
        model, newlyTrained = train_emd_model(model_disc, trainX, trainY, forceTraind= forceTraind)

        one_newlyTrained = one_newlyTrained or newlyTrained    

        #dummy, pred_tr = testPerform(model_disc, model, trainX, trainY, scalerY = None)
        pred_tr = model.predict(trainX)
        pred_tr = pred_tr.reshape(pred_tr.shape[0],1)
     
        datas.append([trainY, pred_tr])
        legends.append(['imfs_'+str(imfs_id), 'imfs_'+ str(imfs_id) + '_prediction'])

        #plotlines([trainY, pred_tr, dummy], ['obs', 'pred', 'nwp'], show = True)
        print('pred_tr', pred_tr.shape)

        if imfs_id == 0:
            pred_tr_results =pred_tr
        else:
            pred_tr_results = numpy.append(pred_tr_results, pred_tr, axis=1)
            print('pred_tr_results', pred_tr_results.shape)

    # plotlines_multifigue(datas, legends, xlabel= 'Date-Time', ylabel= 'Wind speed (m/s)', xticks= xticts, xtick_space= 336, display_lenth= 1201, show = True, savepath='./obs_imfs_pred.png', figsize=(12,9) )

    # sys.exit()
    # datasetY_tmp = datasetY[:1201]
    # xticts_tmp = xticts[0:1201,0]

    dummy, obs_tr, dummy = create_multifeature_nwp_dataset(dataset_tr, NWP_START_INDEX, first_emd_model_disc.window, first_emd_model_disc.predict_hour)
    print('pred_tr_resutls, obs_tr', pred_tr_results.shape, obs_tr.shape)   
    
    merge_model_disc = Merge_model_disc(first_emd_model_disc.predict_hour, first_emd_model_disc.window, first_emd_model_disc.dataset_len, first_emd_model_disc.epcoh, site = first_emd_model_disc.site, train_test_split=first_emd_model_disc.train_test_split)
    
    model = train_merge_models(merge_model_disc, pred_tr_results, obs_tr, newlyTrained)
    merge_pred = model.predict(pred_tr_results)
    merge_pred = merge_pred.reshape(merge_pred.shape[0],1)
    
    datas.pop(0)
    legends.pop(0)

    # plotlines_multifigue(datas, legends, xlabel= 'Date-Time', ylabel= 'Wind speed (m/s)', xticks= xticts, xtick_space= 336, display_lenth= 1201, show = True, savepath='./imfs_preds.png', figsize=(8,9))

    plotlines([obs_tr, merge_pred], ['Observation', 'Final prediction'], xlabel= 'Date-Time', ylabel= 'Wind speed (m/s)', xticks= xticts, xtick_space= 168, display_lenth= 1201, show = True, savepath='./obs_finalPrediction.png', figsize=(16,9))

    return one_newlyTrained
    




if __name__ == '__main__':

    
    #lstm(DATAPATH + WIND_CSV, DATAPATH + NWP_CSV, window, predict_hour)

    pass