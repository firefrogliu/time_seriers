import logging
from lstm import *

from emd_lstm import *
from clean_data import *
from clean_val_data import *
from constants import *
from detrend import *

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
    update_obs_raw = DATAPATH + site + UPDATE_OBS_RAW
    update_obs_val_raw = DATAPATH + site + UPDATE_OBS_VAL_RAW
    update_obs_csv = DATAPATH + site + UPDATE_OBS_CSV
    
    update_obs_nwp_allyear_csv = DATAPATH + site + UPDATA_OBS_NWP_ALLYEAR_CSV

    #clean data

    #connect val_data with train data
    combine_two_wind_tprh_nwp(wind_tprh_nwp_csv,wind_tprh_nwp_val_csv, wind_tprh_nwp_allyear_csv)
    clean_update_obs_data(update_obs_raw, update_obs_val_raw, update_obs_csv)
    combine_updateobs_nwp_allyear(update_obs_csv, wind_tprh_nwp_allyear_csv, update_obs_nwp_allyear_csv)

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

def train_model(site, window, epcoh, dropout, min_wind, max_wind, forceTrain):
    # print('im here')

    
    if forceTrain:

        first_model = None

        for predict_hour in range(1,25):        
            dataset_len = TRAIN_TEST_SIZE
            model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh, dropout, site, min_wind, max_wind)
            if predict_hour == 1:
                first_model = model_disc
            
            lstm_multifeature(model_disc)
    
        return first_model, True
    
    else:
        try:
            first_model = Lstm_model_disc(24, window, TRAIN_TEST_SIZE, epcoh, dropout, site, min_wind, max_wind)
            #return first_model
            print("loading" + first_model.model_name + '.h5')
            keras.models.load_model(MODEL_PATH + first_model.site + first_model.model_name + '.h5')
            print('succeed')
            return first_model, False
            
        except:

            first_model = None

            for predict_hour in range(1,25):        
                dataset_len = TRAIN_TEST_SIZE
                model_disc = Lstm_model_disc(predict_hour, window, dataset_len, epcoh, dropout, site, min_wind, max_wind)
                if predict_hour == 24:
                    first_model = model_disc
                
                lstm_multifeature(model_disc)
        
            return first_model, True
    


def test_seq2seq_model(site_idx, note):
    sites = ['site0/', 'site1/','site2/', 'site3/','site4/', 'site5/','site6/', 'site7/','site8/']
    


    #for site_idx in range(0,9):
    site = str(sites[site_idx])
    clean_data(site)

    epcoh = 1
    
    window = 24
    look_forword = 24
    dropout = 0.4
    
    seq2seq_model_disc = Seq2seq_model_disc(look_forword, window, TRAIN_TEST_SIZE, epcoh, dropout, site)
    seq2seqModel(seq2seq_model_disc)
    sys.exit()


def test_ma_maRes_model(site_idx, NotADirectoryError):
    sites = ['site0/', 'site1/','site2/', 'site3/','site4/', 'site5/','site6/', 'site7/','site8/']
    site = str(sites[site_idx])
    clean_data(site)

    epcoh = 100
    window = 72
    dropout = 0.4
    ma_window = 168

    ma_model_disc = Moving_ave_model_disc(24, window, TRAIN_TEST_SIZE, epcoh, dropout, site, ma_window)
    ma_testY, ma_predY = ma_model(ma_model_disc)

    ma_res_model_disc = Ma_res_model_disc(24, window, TRAIN_TEST_SIZE, epcoh , dropout, site, ma_window)
    ma_res_testY, ma_res_predY = ma_res_model(ma_res_model_disc)

    test_ma_maRes_combined(ma_testY, ma_predY, ma_res_testY, ma_res_predY)


def testBaselineModel(site_idx, note):

    sites = ['site0/', 'site1/','site2/', 'site3/','site4/', 'site5/','site6/', 'site7/','site8/']
    #for site_idx in range(0,9):
    site = str(sites[site_idx])
    clean_data(site)

    epcoh = 50
    window = 24
    dropout = 0.4
    min_wind = 0
    max_wind = 100
    forceTrain = False

    first_model, newlyTrained =  train_model(site,window, epcoh, dropout, min_wind, max_wind, forceTrain)
    site,testScore, npwScore, bw_testScore, bw_npwScore, sw_testScore, sw_npwScore = load_and_validate_model(first_model, newlyTrained)

    if site_idx == 0:
        results= [[first_model.model_name, note]]
        results.append(['site','validationScore','nwpScore', 'bw_validationScore', 'bw_nwpScore', 'sw_validationScore', 'sw_nwpScore'])
    else:
        results = []

    results.append([site,testScore, npwScore, bw_testScore, bw_npwScore, sw_testScore, sw_npwScore])
    with open(RESULTPATH + 'final_result.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(results)

def testClassification_fusionModel(site_idx, note):

    sites = ['site0/', 'site1/','site2/', 'site3/','site4/', 'site5/','site6/', 'site7/','site8/']
    

    #for site_idx in range(0,9):
    site = str(sites[site_idx])
    logging.info('traing site' + site)
    clean_data(site)

    epcoh = 100
    classifciation_epcoh = 100
    window = 72
    
    dropout = 0.4
    min_wind = 0
    wind_bar = 6
    max_wind = 100
    forceTrain = False
 
    small_wind_model = train_model(site,window, epcoh, dropout, min_wind, wind_bar, forceTrain)
    big_wind_model = train_model(site,window, epcoh, dropout, wind_bar, max_wind, forceTrain)
    clsfi_model = train_classifcation_model(site,window, classifciation_epcoh, dropout, wind_bar)

    combin_two_models(small_wind_model, big_wind_model, clsfi_model)

def testCombindeEmdModels(site):
    epcoh = 50
    window = 24     
    
    #predict_hour = 1
    dropout = 0.4
    train_test_split = 0.99

    imfs_id = 4
    model_disc = Emd_model_disc(predict_hour, window, TRAIN_TEST_SIZE, epcoh, dropout, imfs_id,site, train_test_split= train_test_split)
    score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up, final_24_val = comb_24_emd_models(model_disc)
    
    return final_24_val.reshape(final_24_val.shape[0],1)

    results = []
    if predict_hour == 0:
        results = [[model_disc.model_name]]
        results.append(['site','predict_hour','score_nwp', 'score_pre', 'score_up', 'score_bw_nwp', 'score_bw_pre', 'score_bw_up'])
        results.append([site, predict_hour, score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up])
    else:
        results.append([site, predict_hour, score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up])
    with open(RESULTPATH + 'vmd_results_combined.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(results)
    sys.exit()

def get_allsite_prediction(site_list: list):
    sites = ['site0/', 'site1/','site2/', 'site3/','site4/', 'site5/','site6/', 'site7/','site8/']
    site_real_name = ['01', '18L', '18R', '19', '36L', '36R', 'MID1', 'MID2', 'MID3']
    itr = 0
    for real_name in site_list:
        idx = site_real_name.index(real_name)
        site = sites[idx]
        if itr == 0:
            final_24_val = testCombindeEmdModels(site)
            # print(final_24_val.shape)
            # print(final_24_val[:10])
            # sys.exit()
        else:
            final_24_val = numpy.append(final_24_val, testCombindeEmdModels(site), axis=1)

        itr += 1
    
    file_text = []
    
    even_day = False
    for i in range(final_24_val.shape[0]):        

        even_day = not even_day
        row = final_24_val[i, :]
        row = row.tolist()

        row_text = ' '.join([str(x) for x in row])
        file_text.append(row_text)

    hour = 0
    even_day = False
    with open('emd_fcst.dat', 'w') as the_file:
        for row in file_text:
            if even_day:
                the_file.write(row + '\n')
            hour = hour + 1
            if hour == 24:
                even_day = not even_day
                hour = 0
    pass


def runEmdPerDay(site):
    epcoh = 50
    window = 24     
    
    predict_hour = 1
    dropout = 0.4
    train_test_split = 0.99
    forceTrain = False

    imfs_id = 4


    #for predict_hour in range(24,25):
    model_disc = Emd_model_disc(predict_hour, window, TRAIN_TEST_SIZE, epcoh, dropout, imfs_id,site, train_test_split= train_test_split)
    #emd_day_by_day_test(model_disc)

    pass

def test_imfs_model(site):
    epcoh = 50
    window = 24     
    
    predict_hour = 1
    dropout = 0.4
    train_test_split = 0.99
    forceTrain = False

    imfs_id = 4


    #for predict_hour in range(24,25):
    model_disc = Emd_model_disc(predict_hour, window, TRAIN_TEST_SIZE, epcoh, dropout, imfs_id,site, train_test_split= train_test_split)
    
    #test_incemental_imfs(model_disc)
    train_imfs_model(model_disc)


def testEmdModel(site, predict_hour):
    epcoh = 50
    window = 24     
    
    #predict_hour = 1
    dropout = 0.4
    train_test_split = 0.99
    forceTrain = False

    imfs_id = 4


    #for predict_hour in range(24,25):
    model_disc = Emd_model_disc(predict_hour, window, TRAIN_TEST_SIZE, epcoh, dropout, imfs_id,site, train_test_split= train_test_split)
    
    one_newlyTrained = True
    #one_newlyTrained = emdModels(model_disc, forceTraind = False)
    
    score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up = val_emd_models(model_disc, one_newlyTrained)

    results = []
    print(score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up)
    if predict_hour == 1:
        results = [[model_disc.model_name]]
        results.append(['site','predict_hour','score_nwp', 'score_pre', 'score_up', 'score_bw_nwp', 'score_bw_pre', 'score_bw_up'])
        results.append([site, predict_hour, score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up])
    else:
        results.append([site, predict_hour, score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up])
    with open(RESULTPATH + 'vmd_results_24hour.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(results)
    sys.exit()


if __name__ == '__main__':

    # site_idx = int(sys.argv[1])
    # note = sys.argv[2]

    #site_idx = 0
    #note = 'vmd'
    
    sites = ['site0/', 'site1/','site2/', 'site3/','site4/', 'site5/','site6/', 'site7/','site8/']
    
    site_idx = int(sys.argv[1])
    predict_hour = int(sys.argv[2])

    site = str(sites[site_idx])

    #testEmdModel(site, predict_hour)
    #testCombindeEmdModels(site)
    #get_allsite_prediction(['18R','36L','MID1','18L','36R','MID2','01','19','MID3'])
    #runEmdPerDay(site)
    test_imfs_model(site)
