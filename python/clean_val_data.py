from constants import *
from clean_data import clean_nwp_data, clean_tprh_data, clean_wind_data, combine_wind_tprh
import pandas
from pandas import read_csv
import numpy
from sklearn.preprocessing import MinMaxScaler


def clean_nwp_val_data(nwp_val_txt, nwp_val_csv):
    clean_nwp_data(nwp_val_txt, nwp_val_csv)
    
    pass

def clean_wind_val_data(wind_val_txt, wind_val_csv):
    clean_wind_data(wind_val_txt,wind_val_csv)
    pass

def clean_tprh_val_data(tprh_val_txt, tprh_val_csv):
    clean_tprh_data(tprh_val_txt, tprh_val_csv)
    pass

def combine_wind_tprh_val(wind_val_csv, tprh_val_csv, wind_tprh_val_csv):
    combine_wind_tprh(tprh_val_csv, wind_val_csv, wind_tprh_val_csv)

def get_val_testX(wind_tprh_val_csv, nwp_val_csv):
    dataX_24_hours = [] #result

    windTprhdataframe = read_csv(wind_tprh_val_csv, usecols = ['wind','ap','tmp','humi'], engine='python')
    print(wind_tprh_val_csv)
    nwpdataframe = read_csv(nwp_val_csv, usecols = ['wind','dir','u', 'v', 't', 'rh', 'psfc', 'slp'], engine='python')
    wt_dataset = windTprhdataframe.values[:]
    wt_dataset = wt_dataset.astype('float32')

    nwp_dataset = nwpdataframe.values[:]
    nwp_dataset = nwp_dataset.astype('float32')

    wt_scaler = MinMaxScaler(feature_range=(0, 1))
    nwp_scaler = MinMaxScaler(feature_range=(0, 1))

    wt_dataset = wt_scaler.fit_transform(wt_dataset)
    nwp_dataset = nwp_scaler.fit_transform(nwp_dataset)
    
    days = len(wt_dataset) / 24
    window = 24
    predict_hour = 1
    #note that nwp contains all days data, while wind and trph only contains data every other day
    while predict_hour < 25:
        dataX = []
        day = 0
        while day < days:
            obs_features = wt_dataset[window * day : window * (day + 1), :]
            #nwp_features = nwp_dataset[(2*day+1) * window + predict_hour - 1, :]
            nwp_features = nwp_dataset[(day+1) * window + predict_hour - 1, :]
            obs_nwp_featue = []
            for row in obs_features:
                tmp = numpy.append(row, nwp_features)
                obs_nwp_featue.append(tmp)
            
            dataX.append(obs_nwp_featue)

            day += 1
        
        dataX_24_hours.append(dataX)
        predict_hour += 1
    result = numpy.array(dataX_24_hours)
    #print(len(result))
    #print(result[-1][-1])
    return result





if __name__ == '__main__':
    site0 = 'site0/' 
    wind_val_txt = DATAPATH + site0 + WIND_VAL_TXT
    wind_val_csv = DATAPATH + site0 + WIND_VAL_CSV
    tprh_val_txt = DATAPATH + site0 + TPRH_VAL_TXT
    tprh_val_csv = DATAPATH + site0 + TPRH_VAL_CSV
    nwp_val_txt = DATAPATH + site0 + NWP_VAL_TXT
    nwp_val_csv = DATAPATH + site0 + NWP_VAL_CSV
    wind_tprh_val_csv = DATAPATH + site0 + WIND_TPRH_VAL_CSV

    #clean_nwp_val_data(nwp_val_txt, nwp_val_csv)
    #clean_wind_val_data(wind_val_txt, wind_val_csv)
    #clean_tprh_val_data(tprh_val_txt, tprh_val_csv)
    #combine_wind_tprh_val(wind_val_csv, tprh_val_csv, wind_tprh_val_csv)
    get_val_testX(wind_tprh_val_csv, nwp_val_csv)
    
    pass
