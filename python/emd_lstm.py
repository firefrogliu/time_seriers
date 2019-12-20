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
from pyplotz.pyplotz import PyplotZ

from detrend import moving_average, emd_detrend, ceemd_detrend, vmd_detrend
import pickle
import random
import copy

from garget import *
from keras import backend as K

from lstm import Emd_model_disc, load_np_array, save_np_array, create_multifeature_nwp_dataset, Merge_model_disc, load_model, append_component, testPerform


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
        dataset, datasetY, nwp, scalerY, scalerNwp, dates = prepare_emd_data(first_emd_model_disc)
        val_start = first_emd_model_disc.dataset_len
        dataset_val = dataset[val_start: ]
        dates_val = dates[val_start:]
        #datasetY_val = datasetY[val_start: ]
        #nwp_val =  nwp[val_start:]
        imfs= vmd_detrend(datasetY)
        print('imfs.shape', imfs.shape)
        imfs_val = imfs[val_start:, :]
        #sys.exit()
        nwp_imfs = vmd_detrend(nwp)
        nwp_imfs_val = nwp_imfs[val_start:, :]
        print('nwp_imfs_val.shape', nwp_imfs_val.shape)
        pred_val_results = None
        
        for imfs_id in range(imfs_val.shape[1]):
            model_disc = copy.copy(first_emd_model_disc)
            model_disc.set_immfs_idx(imfs_id)

            emd_model = load_model(model_disc)

            dateset_val_app_comp = append_component(dataset_val, imfs_val, imfs_id)
            dateset_val_app_comp = append_component(dateset_val_app_comp, nwp_imfs_val, imfs_id)
            valX, dummy, dummy, Y_dates = create_multifeature_nwp_dataset(dateset_val_app_comp, NWP_START_INDEX, model_disc.window, model_disc.predict_hour, nwp_end=8, Ycolumn_idx=dateset_val_app_comp.shape[1] -2, dates= dates_val )
            #print(valX[0], Y_dates[0])
            #sys.exit()
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

def get_today_obs_tmr_nwp(dataset, today_start):
    obs_today = copy.copy(dataset[today_start: today_start+ DAY_HOURS * 1, :NWP_START_INDEX])
    nwp_tmr = copy.copy(dataset[today_start + DAY_HOURS: today_start+ DAY_HOURS * 2, NWP_START_INDEX:])
    return obs_today, nwp_tmr

def get_history_dataset(dataset, today_start):

    


    dataset_history = copy.copy(dataset[: today_start + DAY_HOURS])
    zeros = numpy.zeros(shape = (DAY_HOURS * 1, NWP_START_INDEX))
    dataset_history[- DAY_HOURS:, : NWP_START_INDEX] = zeros
    print('dataset history',dataset_history.shape)
    return dataset_history  


def add_today_obs(dataset_tr, today_obs):
    dataset_tr[- DAY_HOURS: , :NWP_START_INDEX] = today_obs
    return dataset_tr

def add_tmr_nwp(dataset_tr, tomorrow_nwp):
    zeros = numpy.zeros(shape = (DAY_HOURS * 1, dataset_tr.shape[1]))
    dataset_tr = numpy.append(dataset_tr, zeros, axis= 0)
    dataset_tr[-DAY_HOURS * 1:, NWP_START_INDEX:] = tomorrow_nwp
    return dataset_tr
def load_run_emd_model(first_emd_model_disc, predict_hour, dataset_app_comp,imfs_id, dates = None):
    model_disc = copy.copy(first_emd_model_disc)
    model_disc.set_immfs_idx(imfs_id)
    model_disc.set_predict_hour(predict_hour)

    emd_model = load_model(model_disc)

    print('dataset_app_comp shape', dataset_app_comp.shape)
    valX, dummy, dummy, Y_dates = create_multifeature_nwp_dataset(dataset_app_comp, NWP_START_INDEX, model_disc.window, model_disc.predict_hour, nwp_end=8, Ycolumn_idx=dataset_app_comp.shape[1] -2, dates = dates)
    print('valx shape', valX.shape)
    print(valX[0], Y_dates[0])
    sys.exit()
    #dummy, pred_val = testPerform(model_disc, emd_model, valX, valY, scalerY = None)
    pred_val = emd_model.predict(valX)
    del emd_model
    K.clear_session()
    #gc.collect()
    pred_val = pred_val.reshape(pred_val.shape[0],1)
    return pred_val
def load_run_merge_model(first_emd_model_disc, predict_hour, pred_val_results):
    merge_model_disc = Merge_model_disc(predict_hour, first_emd_model_disc.window, first_emd_model_disc.dataset_len, first_emd_model_disc.epcoh,  site = first_emd_model_disc.site,  train_test_split=first_emd_model_disc.train_test_split)
    merge_model =  load_model(merge_model_disc)

    merge_pred_val = merge_model.predict(pred_val_results)
    merge_pred_val = merge_pred_val.reshape(merge_pred_val.shape[0],1)
    del merge_model
    K.clear_session()
    #gc.collect()
    #save_np_array(first_emd_model_disc, merge_pred_val, 'hour_1_pred_val'+str(predict_hour))
    return merge_pred_val

def load_run_emd_merge_model(first_emd_model_disc, predict_hour,dataset_history, imfs, nwp_imfs, dates = None):
    
    for imfs_id in range(imfs.shape[1]):

        dataset_app_comp = append_component(dataset_history, imfs, imfs_id)
        dataset_app_comp = append_component(dataset_app_comp, nwp_imfs, imfs_id)
        if predict_hour < 24:
            dataset_app_comp = dataset_app_comp[-(DAY_HOURS * 1 + first_emd_model_disc.window): -(DAY_HOURS * 1 - predict_hour), :]
            dates = dates[-(DAY_HOURS * 1 + first_emd_model_disc.window): -(DAY_HOURS * 1 - predict_hour)]
        else:
            dataset_app_comp = dataset_app_comp[-(DAY_HOURS * 1 + first_emd_model_disc.window): , :]
            dates = dates[-(DAY_HOURS * 1 + first_emd_model_disc.window):]
        pred_val = load_run_emd_model(first_emd_model_disc, predict_hour, dataset_app_comp, imfs_id, dates)

        if imfs_id == 0:
            pred_val_results =pred_val
        else:
            pred_val_results = numpy.append(pred_val_results, pred_val, axis=1)
            print('pred_val_results', pred_val_results.shape)

    merge_pred_val = load_run_merge_model(first_emd_model_disc, predict_hour, pred_val_results)
    
    return merge_pred_val
#feed 24 hours of data, predict the next 24 hours
def run_emd_model_on_newday(first_emd_model_disc, dataset_history, scalerY, today_obs, tomorrow_nwp, dates_upto_tmr = None):
    
    # hour_24_pred = load_np_array(first_emd_model_disc, 'hour_24_pred')
    # if hour_24_pred is not None:
    #     return hour_24_pred
    #print('dataset_histrory', dataset_history.shape)
    dataset_history = add_today_obs(dataset_history, today_obs)
    #print('dataset_histrory after add obs', dataset_history.shape)
    dataset_history = add_tmr_nwp(dataset_history, tomorrow_nwp)
    #print('dataset_histrory after add nwp', dataset_history.shape)
    #sys.exit()
    datasetY_history = dataset_history[: - DAY_HOURS * 1 ,0]
    nwp = dataset_history[:, NWP_START_INDEX]
    imfs = vmd_detrend(datasetY_history)
    zeros = numpy.zeros(shape = (DAY_HOURS * 1, imfs.shape[1]))
    imfs =  numpy.append(imfs, zeros, axis= 0)

    nwp_imfs = vmd_detrend(nwp)

    hour_24_pred = []
    for predict_hour in range(1,25):

        merge_pred_val = load_run_emd_merge_model(first_emd_model_disc, predict_hour, dataset_history, imfs, nwp_imfs, dates_upto_tmr)
        merge_pred_val = scalerY.inverse_transform(merge_pred_val)
        hour_24_pred.append(merge_pred_val[0,0])
    hour_24_pred = numpy.array(hour_24_pred)
    print(hour_24_pred.shape)
    hour_24_pred = hour_24_pred.reshape(24,1)
    #save_np_array(first_emd_model_disc, hour_24_pred, 'hour_24_pred')
    return hour_24_pred
    # score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up = compare_nwp_dpl(obs_val, nwp_val, merge_pred_val)


def prepare_emd_data(first_emd_model_disc:Emd_model_disc, data_start = 0, data_end = 'end'):
    dataset, datasetY = read_dataframe(first_emd_model_disc.wind_tprh_nwp_file, data_start, data_end)

    xticts = read_csv(first_emd_model_disc.wind_tprh_nwp_file, usecols=['data-time'], engine='python')
    xticts = xticts.values[:,0]

    scaler = MinMaxScaler(feature_range=(0, 1))
    
    dataset = scaler.fit_transform(dataset)
   
    scalerY =  MinMaxScaler(feature_range=(0, 1))
    datasetY = scalerY.fit_transform(datasetY)

    dummy, nwp = read_dataframe(first_emd_model_disc.wind_tprh_nwp_file, 0, 'end', ycolumn='nwp_wind')
    scalerNwp = MinMaxScaler(feature_range=(0, 1))
    nwp = scalerNwp.fit_transform(nwp)

    return dataset, datasetY, nwp, scalerY, scalerNwp, xticts



#train emdModels and essemble emd models with a dense layer
def emdModels(first_emd_model_disc:Emd_model_disc, forceTraind = False):
    dataset, datasetY, nwp, scalerY, scalerNwp = prepare_emd_data(first_emd_model_disc)

    test_start = first_emd_model_disc.dataset_len*first_emd_model_disc.train_test_split
    dataset_tr = dataset[: test_start]
    datasetY_tr = datasetY[: test_start]
    nwp_tr =  nwp[: test_start]
    imfs_tr = vmd_detrend(datasetY_tr)
    nwp_imfs_tr = vmd_detrend(nwp_tr)


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
    

def emd_day_by_day_test(first_emd_model_disc):
    #today_start = int(first_emd_model_disc.dataset_len * first_emd_model_disc.train_test_split)
    today_start = first_emd_model_disc.dataset_len
    dataset, dummy, dummy, scalerY, scalerNwp, dates = prepare_emd_data(first_emd_model_disc)
   

    pred = None
    obs = None
    nwp = None
    days = 10
    offset = 0
    obs = copy.copy(dataset[today_start + offset: today_start +offset + DAY_HOURS * days, 0])
    nwp = copy.copy(dataset[today_start  + offset: today_start + offset + DAY_HOURS * days, NWP_START_INDEX])
    obs = scalerY.inverse_transform([obs]).T
    nwp = scalerNwp.inverse_transform([nwp]).T
    pred = load_np_array(first_emd_model_disc, 'pred' + str(days)+'days')
    pred = None
    if pred is None:
        for day in range(days):
            dataset_history = get_history_dataset(dataset, today_start)
            obs_today, nwp_tmr = get_today_obs_tmr_nwp(dataset, today_start)
            dates_upto_tmr = dates[:today_start + 48]
            pred_tmr = run_emd_model_on_newday(first_emd_model_disc, dataset_history, scalerY, obs_today, nwp_tmr , dates_upto_tmr)
            today_start += 24
            if day == 0:
                pred = copy.copy(pred_tmr)
                #obs = obs_tmr[:, 0]
                #nwp = nwp_tmr[:, 0]
            else:
                pred = numpy.append(pred,pred_tmr, axis = 0)
                #obs = numpy.append(obs ,obs_tmr[:, 0], axis = 0)
                #nwp = numpy.append(nwp ,nwp_tmr[:, 0], axis = 0)

        save_np_array(first_emd_model_disc, pred, 'pred' + str(days)+'days')
    print('obs.shape', obs.shape)
    compare_nwp_dpl(obs, nwp , pred)
    plotlines([obs,nwp, pred], ['obs', 'nwp', 'dpl'], show = True)
    pass
    
def test_incemental_imfs(model_disc: Emd_model_disc):
    dataset, dummy, dummy, scalerY, scalerNwp, dates = prepare_emd_data(model_disc)
    start = model_disc.dataset_len
    y = dataset[:, 0]
    ori_imfs = vmd_detrend(y)[:, 0]
    today = start + 24
    for day in range(1):
        y_tillNow = dataset[: today, 0]        
        tillNow_imfs = vmd_detrend(y_tillNow)[:, 0]
        incremental = tillNow_imfs[-24:]
        if day == 0:
            incremental_ifms = incremental
        else:
            incremental_ifms = numpy.append(incremental_ifms, incremental, axis= 0)
        today += 24
    plotlines([ori_imfs[start:today], incremental_ifms], ['all_time', 'till_now'], show = True)
    pass

def imfs_model(trainX, trainY):

    trainX = trainX.reshape((trainX.shape[0], trainY.shape[1], 1))
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
    model.fit(trainX, trainY, epochs= 50 , batch_size = 128, verbose= 1)
    return model
    pass


def train_imfs_model(model_disc):
    dataset, dummy, dummy, scalerY, scalerNwp, dates = prepare_emd_data(model_disc)
    start = 30
    y = dataset[:, 0]
    ori_imfs = vmd_detrend(y)[:, 0]
    day = start + 1 
    train_size = 80
    train_test_size = 100

    trainX = load_np_array(model_disc, 'imfs_trainX' + str(train_size))
    trainY = load_np_array(model_disc, 'imfs_trainY' + str(train_size))
    if trainX is None or trainY is None:
        trainX = []
        trainY = []
        while day < train_test_size:
            now = day * 24
            y_tillNow = dataset[: now, 0]        
            tillNow_imfs = vmd_detrend(y_tillNow)[:, 0]
            incremental = tillNow_imfs[-24:]
            
            trainX.append(incremental)
            trainY.append(ori_imfs[now-24: now])
            day += 1


        trainX = numpy.array(trainX)
        print('trainX', trainX.shape)
        trainY = numpy.array(trainY)
        print('trainY', trainY.shape)
        save_np_array(model_disc, trainX, 'imfs_trainX' + str(train_size))
        save_np_array(model_disc, trainX, 'imfs_trainY' + str(train_size))  
    #plotlines([trainX.flatten(), trainY.flatten()], ['till_now','alltime'], show = True)
  
    model = imfs_model(trainX[:train_size], trainY[:train_size])
    
    testX = trainX[:train_size]
    testX = testX.reshape((testX.shape[0], testX.shape[1], 1))
    testY = model.predict(testX)
    plotlines([trainY[train_size:].flatten(), testY.flatten()], ['alltime','till_now'], show = True)
    pass

if __name__ == '__main__':
    
    
    #lstm(DATAPATH + WIND_CSV, DATAPATH + NWP_CSV, window, predict_hour)

    pass