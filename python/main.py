import logging
from lstm import *
from clean_data import *
from clean_val_data import *
from constants import *

DATAPATH = '../data/'
RESULTPATH = '../results/'
MODEL_PATH = '../models/'
WIND_TXT = 'wind.txt'
WIND_CSV = 'wind.csv'

TPRH_CSV = 'tprh.csv'

NWP_TXT = 'nwp.txt'
NWP_CSV = 'nwp.csv'

WIND_TPRH_CSV = 'wind_tprh.csv'
WIND_TPRH_NWP_CSV = 'wind_tprh_nwp.csv'


TPRH_TXT = 'tprh.txt'
TPRH_CSV = 'tprh.csv'


def clean_data(site):
    #clean train data
    wind_raw =  DATAPATH + site + WIND_TXT
    wind_csv = DATAPATH + site + WIND_CSV
    nwp_raw = DATAPATH + site + NWP_TXT
    nwp_csv = DATAPATH + site + NWP_CSV
    tprh_raw = DATAPATH + site + TPRH_TXT
    tprh_csv = DATAPATH + site + TPRH_CSV
    wind_tprh_csv = DATAPATH + site + WIND_TPRH_CSV
    wind_tprh_nwp_csv = DATAPATH + site + WIND_TPRH_NWP_CSV

    clean_wind_data(wind_raw,wind_csv)
    clean_tprh_data(tprh_raw, tprh_csv)
    clean_nwp_data(nwp_raw,nwp_csv)
    combine_wind_tprh(tprh_csv,wind_csv,wind_tprh_csv)
    combine_wind_tprh_nwp(wind_tprh_csv,nwp_csv,wind_tprh_nwp_csv)

    
    #clean validation data
    wind_val_txt = DATAPATH + site + WIND_VAL_TXT
    wind_val_csv = DATAPATH + site + WIND_VAL_CSV
    tprh_val_txt = DATAPATH + site + TPRH_VAL_TXT
    tprh_val_csv = DATAPATH + site + TPRH_VAL_CSV
    nwp_val_txt = DATAPATH + site + NWP_VAL_TXT
    nwp_val_csv = DATAPATH + site + NWP_VAL_CSV
    wind_tprh_val_csv = DATAPATH + site + WIND_TPRH_VAL_CSV
    wind_tprh_nwp_val_csv = DATAPATH + site + WIND_TPRH_NWP_VAL_CSV
    
    clean_nwp_val_data(nwp_val_txt, nwp_val_csv)
    clean_wind_val_data(wind_val_txt, wind_val_csv)
    clean_tprh_val_data(tprh_val_txt, tprh_val_csv)
    combine_wind_tprh_val(wind_val_csv, tprh_val_csv, wind_tprh_val_csv)
    combine_wind_tprh_nwp(wind_tprh_val_csv, nwp_val_csv, wind_tprh_nwp_val_csv)    

    wind_tprh_nwp_allyear_csv = DATAPATH + site + WIND_TPRH_NWP_ALLYEAR_CSV
    #clean data

    #connect val_data with train data
    combine_two_wind_tprh_nwp(wind_tprh_nwp_csv,wind_tprh_nwp_val_csv, wind_tprh_nwp_allyear_csv)

def train_classifcation_model(site, window, epcoh, dropout, wind_bar):
        #predict_hour, window, datalen, epcoh
    try:
        first_model = Classification_model_disc(1, window, TRAIN_TEST_SIZE, epcoh, dropout, site, wind_bar)
        
        keras.models.load_model(MODEL_PATH + first_model.site + first_model.model_name + '.h5', custom_objects={'f1_m': f1_m, 'precision_m':precision_m, 'recall_m':recall_m})
        return first_model
    
        pass
    except:
        first_model = None

        for predict_hour in range(1,25):        
            dataset_len = TRAIN_TEST_SIZE
            model_disc = Classification_model_disc(predict_hour, window, dataset_len, epcoh, dropout, site, wind_bar)
            if predict_hour == 1:
                first_model = model_disc
            
            lstm_classfier(model_disc)
    
        return first_model

def train_model(site, window, epcoh, dropout, min_wind, max_wind):
    print('im here')

    

    try:
        first_model = Lstm_model_disc(1, window, TRAIN_TEST_SIZE, epcoh, dropout, site, min_wind, max_wind)
        #return first_model
        print("loading" + first_model.model_name + '.h5')
        keras.models.load_model(MODEL_PATH + first_model.site + first_model.model_name + '.h5')
        print('succeed')
        return first_model
        
    except:
        print('failed')
        #sys.exit()
    #predict_hour, window, datalen, epcoh
        first_model = None

        for predict_hour in range(1,25):        
            dataset_len = TRAIN_TEST_SIZE
            model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh, dropout, site, min_wind, max_wind)
            if predict_hour == 1:
                first_model = model_disc
            
            lstm_multifeature(model_disc)
    
        return first_model
    


if __name__ == '__main__':

    LOG_FILENAME = "logfile.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)    

    sites = ['site0/', 'site1/','site2/', 'site3/','site4/', 'site5/','site6/', 'site7/','site8/']
    #site_idx = int(sys.argv[1])

    site_idx = int(sys.argv[1])
    if site_idx == 0:
        results = [['site','validationScore','nwpScore', 'bw_validationScore', 'bw_nwpScore', 'sw_validationScore', 'sw_nwpScore']]
    else:
        results = []
    #for site_idx in range(0,9):
    site = str(sites[site_idx])
    logging.info('traing site' + site)
    clean_data(site)
    epcoh = 50
    classifciation_epcoh = 300
    
    window = 24
    

    dropout = 0.4
    min_wind = 0
    wind_bar = 6
    max_wind = 100
    # first_model =  train_model(site,window, epcoh, dropout, min_wind, max_wind)
    # load_and_validate_model(first_model)


    small_wind_model = train_model(site,window, epcoh, dropout, min_wind, wind_bar)
    big_wind_model = train_model(site,window, epcoh, dropout, wind_bar, max_wind)
    clsfi_model = train_classifcation_model(site,window, classifciation_epcoh, dropout, wind_bar)

    combin_two_models(small_wind_model, big_wind_model, clsfi_model)
    # results.append([site,testScore, npwScore, bw_testScore, bw_npwScore, sw_testScore, sw_npwScore])
    # print(site,testScore, npwScore, bw_testScore, bw_npwScore, sw_testScore, sw_npwScore)

    # with open(RESULTPATH + 'window' + str(window)+'final_result.csv', 'a') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(results)
