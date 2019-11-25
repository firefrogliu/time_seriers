from constants import *
import pandas
import matplotlib.pyplot as plt
import csv
# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,CuDNNLSTM
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
from dataclasses import dataclass
import logging
from tensorflow.python.client import device_lib
from pyplotz.pyplotz import PyplotZ
from clean_val_data import get_val_testX
import pickle


from keras import backend as K

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
    train_test_split: float = 0.8


    def __init__(self, predict_hour, window, dataset_len, epcoh, dropout, site, min_wind, max_wind):
        self.predict_hour = predict_hour
        self.window = window
        self.dataset_len = dataset_len
        self.epcoh = epcoh
        self.dropout = dropout
        self.site = site
        self.min_wind = min_wind
        self.max_wind = max_wind   
        
        self.wind_tprh_nwp_file = DATAPATH + site + WIND_TPRH_NWP_ALLYEAR_CSV
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
        
        self.wind_tprh_nwp_file = DATAPATH + site + WIND_TPRH_NWP_ALLYEAR_CSV
        self.model_name = 'pre' + str(self.predict_hour) + 'window' + str(self.window) + 'datalen' + str(self.dataset_len) + 'eph' + str(self.epcoh) + 'drop'+ str(self.dropout*100) + 'windBar' + str(self.wind_bar)
        print('model name', self.model_name)    

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

def run_saved_model(model_disc: Lstm_model_disc, testX):
    numpy.random.seed(7)
    # load the dataset
    dataframeY = read_csv(model_disc.wind_tprh_nwp_file, usecols=['wind'], engine='python')
    
    
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

# def create_multifeature_dataset(dataset, window,predict_hour = 1):
#     dataX, dataY = [], []
#     i = window
#     while i < len(dataset)-predict_hour + 1:
#         a = dataset[(i-window):(i), :]
#         dataX.append(a)
#         dataY.append(dataset[i+predict_hour-1, 0])
#         i += 1
#     return numpy.array(dataX), numpy.array(dataY)

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

def create_multifeature_nwp_dataset(dataset, nwp_start, window, predict_hour = 1, min_wind = 0, max_wind = 100):
    dataX, dataY, nwpY = [], [], []
    i = window
    #print('creating dataset min max', min_wind, max_wind)
    while i < len(dataset)-predict_hour + 1:
        wind = dataset[i+predict_hour-1, 0]

        if wind < min_wind or wind >= max_wind:
            i = i + 1
            #print('found wind', wind, 'bigger than 4')
            continue
        
        dataY.append(wind)
        
        nwp_predict = dataset[i + predict_hour - 1, nwp_start:]
        nwp_predict_wind = dataset[i + predict_hour - 1, nwp_start]
        nwpY.append(nwp_predict_wind)
        a = dataset[(i-window):(i), : nwp_start]
        
        obs_nwp_featue_list = []
        for row in a:
            tmp = numpy.append(row, nwp_predict)
            obs_nwp_featue_list.append(tmp)
        obs_nwp_featue = numpy.array(obs_nwp_featue_list)
        dataX.append(obs_nwp_featue)

        # nwp_predict_wind = dataset[i + predict_hour - 1, nwp_start]
        # nwpY.append(nwp_predict_wind)
        # a = dataset[(i-window):(i), : nwp_start]
        
        # obs_nwp_featue_list = []

        # nwp_pred_hour = 0
        # for row in a:
        #     nwp_predict = dataset[i - window +  nwp_pred_hour + predict_hour - 1, nwp_start:]
        #     tmp = numpy.append(row, nwp_predict)
        #     obs_nwp_featue_list.append(tmp)
        #     nwp_pred_hour += 1
        # obs_nwp_featue = numpy.array(obs_nwp_featue_list)
        # dataX.append(obs_nwp_featue)

        i += 1
    return numpy.array(dataX), numpy.array(dataY), numpy.array(nwpY)
    
def prepare_data(model_disc: Lstm_model_disc):
    dataframe = read_csv(model_disc.wind_tprh_nwp_file, usecols = ['wind','ap','tmp','humi', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp'],engine='python')
    dataframeNwp = read_csv(model_disc.wind_tprh_nwp_file, usecols=['nwp_wind'], engine='python')
    
    dataset_len = model_disc.dataset_len
    dataset = dataframe.values[:]
    dataset = dataset.astype('float32')
    
    
    dataframeY = read_csv(model_disc.wind_tprh_nwp_file, usecols=['wind'], engine='python')
    datasetY = dataframeY.values[:]
    datasetY = datasetY.astype('float32')
    scalerY = MinMaxScaler(feature_range=(0, 1))
    datasetY = scalerY.fit_transform(datasetY)
    
    datasetNwp = dataframeNwp.values[:]
    datasetNwp = datasetNwp.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scalerNwp = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
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
    trainX, trainY, dummy = create_multifeature_nwp_dataset(train, 4, window, predict_hour, min_wind, max_wind)
    testX, testY, test_nwpY = create_multifeature_nwp_dataset(test, 4, window, predict_hour, min_wind, max_wind)
    
    print('trainX, trainY, testX, testY, nwpY',trainX.shape, trainY.shape, testX.shape, testY.shape, test_nwpY.shape)
    #sys.exit()
    return window, predict_hour, dataset, trainX, trainY, testX, testY, test_nwpY,  train_size, scalerNwp, scalerY



def train_results(model_disc: Lstm_model_disc, model: Sequential, window, predict_hour ,dataset, trainX, trainY, testX, testY, test_nwpY, train_size, scalerNwp, scalerY):
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
    
    #plt.clf()
    #plt.plot(testY)    
    #plt.plot(test_nwpY)
    #plt.plot(testPredict)
    #plt.savefig(RESULTPATH + model_disc.site + model_disc.model_name + '_' + '.pdf')
    #plt.show()
    pass

def lstm_multifeature(model_disc: Lstm_model_disc):
 
    print('failed loading model')
    
    window, predict_hour, dataset, trainX, trainY, testX, testY, test_nwpY,  train_size, scalerNwp, scalerY = prepare_data(model_disc)
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
    model.fit(trainX, trainY, epochs= epoch, batch_size= 128, verbose=2)
    model.save(MODEL_PATH + model_disc.site + model_disc.model_name  +'.h5')

    #show train results
    train_results(model_disc, model, window, predict_hour, dataset, trainX, trainY, testX, testY, test_nwpY,  train_size, scalerNwp, scalerY)



def prepare_classification_data(model_disc: Classification_model_disc):
    dataframe = read_csv(model_disc.wind_tprh_nwp_file, usecols = ['wind','ap','tmp','humi', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp'],engine='python')
    
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
    trainX, trainY= creat_classification_dataset(train, 4, window, predict_hour, wind_bar)
    testX, testY = creat_classification_dataset(test, 4, window, predict_hour, wind_bar)
    
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
    dataframe = read_csv(first_model_disc.wind_tprh_nwp_file, usecols = ['wind','ap','tmp','humi', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp'],engine='python')
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

def validation_model(first_model_disc: Lstm_model_disc, predictY, sw = [], bw = [], classification = []):

    dataset_len = first_model_disc.dataset_len
    site = first_model_disc.site
    window = first_model_disc.window
    

    dataframeY = read_csv(first_model_disc.wind_tprh_nwp_file, usecols = ['wind'],engine='python')
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
    #print(test_dates[:100])
    plt.legend(handles = legends)
    plt.savefig(RESULTPATH + first_model_disc.site + '24modelResult.png')
    # plt.clf()
    # plt.plot(testY)
    plt.show()
    return testScore, npwScore, bw_testScore, bw_npwScore, sw_testScore, sw_npwScore,
    

def load_models_and_run(first_model_disc: Lstm_model_disc):
    dataset_len = first_model_disc.dataset_len
    #dataframe = read_csv(model_disc.wind_tprh_file, usecols=['wind','ap','tmp','humi'], engine='python')     
    dataframe = read_csv(first_model_disc.wind_tprh_nwp_file, usecols = ['wind','ap','tmp','humi', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp'],engine='python')
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

        validationX, dummy, dummy2 = create_multifeature_nwp_dataset(validation, 4, window, predict_hour)
        model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh,dropout, site, first_model_disc.min_wind, first_model_disc.max_wind)
        # split into train and test sets
        # print('testX_0',testX_0)
        model_predict = [[0]] * (window + predict_hour - 1)
        model_predict.extend(run_saved_model(model_disc, validationX))
        model_predicts.append(model_predict)
    
    hour = 1
    for i in range(len(model_predicts[0])):

        predict = model_predicts[hour-1][i]
        predictY.append(predict)
        hour += 1
        if hour >= 25:
            hour = 1


        
    return  numpy.array(predictY)


def load_and_validate_model(first_wind_model: Lstm_model_disc):
    try:
        predictY = numpy.load(first_wind_model.model_name + 'predictY' + '.npy')
    except:
        predictY = load_models_and_run(first_wind_model)
        numpy.save(first_wind_model.model_name + 'predictY', predictY) 
    
    validation_model(first_wind_model, predictY)


def combin_two_models(small_wind_model: Lstm_model_disc, big_wind_model:Lstm_model_disc, classifcation_model: Classification_model_disc):

    try:
        sw_predictY = numpy.load(small_wind_model.model_name + 'predictY' + '.npy')
    except:
        sw_predictY = load_models_and_run(small_wind_model)
        numpy.save(small_wind_model.model_name + 'predictY', sw_predictY) 
    

    try:
        bw_predictY = numpy.load(big_wind_model.model_name + 'predictY' + '.npy')
    except:
        bw_predictY = load_models_and_run(big_wind_model)
        numpy.save(big_wind_model.model_name + 'predictY', bw_predictY) 
    
    try:
        predict_classicfication = numpy.load(classifcation_model.model_name + 'predictY' + '.npy')
    except :
        predict_classicfication = load_classification_models_run(classifcation_model)
        numpy.save(classifcation_model.model_name + 'predictY', predict_classicfication) # protocol 0 is printable ASCII

    
    
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


def cal_val_data(site, window,wind_tprh_val_csv, nwp_val_csv):
    dataset_len = 5000
    epcoh = 5
    predict_hour = 1
    dropout = 0.4
    model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh, dropout, site)
    dataX_24_hours = get_val_testX(wind_tprh_val_csv, nwp_val_csv)

    predict_start, predict_end = 1, 25

    
    preditcts = {}
  
    for predict_hour in range(predict_start, predict_end):
        predict = []    
        test_features = dataX_24_hours[predict_hour - 1]    
        print('processing model ', predict_hour)        
        model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh,dropout, site)
        # split into train and test sets
        # print('testX_0',testX_0)
        predict = run_saved_model(model_disc,test_features)
        preditcts[predict_hour] = predict


    datetimeData = read_csv(nwp_val_csv, usecols = ['data-time'], engine='python')
    datetimes = datetimeData.values[:]
    
    
    predicts_in_series = [['time','wind']]
    for day in range(len(preditcts[1])):
        for hour in range(1,25):
            day_time = datetimes[ (2*day + 1) * window + hour - 1][0]
            predicts_in_series.append([day_time,preditcts[hour][day][0]])
    
    with open(RESULTPATH + site + 'predict_val.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(predicts_in_series)

if __name__ == '__main__':
    #lstm(DATAPATH + WIND_CSV, DATAPATH + NWP_CSV, window, predict_hour)

    pass