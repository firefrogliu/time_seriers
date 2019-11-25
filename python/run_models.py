from lstm import load_models_and_run, cal_val_data
from constants import *
import csv
from clean_val_data import *

if __name__ == '__main__':
    results = [['site','testScore','nwpScore', 'bw_testScore', 'bw_nwpScore']]
    window = 24
    for site_idx in range(7,8):
        site = 'site' + str(site_idx) + '/'
        wind_val_txt = DATAPATH + site + WIND_VAL_TXT
        wind_val_csv = DATAPATH + site + WIND_VAL_CSV
        tprh_val_txt = DATAPATH + site + TPRH_VAL_TXT
        tprh_val_csv = DATAPATH + site + TPRH_VAL_CSV
        nwp_val_txt = DATAPATH + site + NWP_VAL_TXT
        nwp_val_csv = DATAPATH + site + NWP_VAL_CSV
        wind_tprh_val_csv = DATAPATH + site + WIND_TPRH_VAL_CSV
        
        clean_nwp_val_data(nwp_val_txt, nwp_val_csv)
        clean_wind_val_data(wind_val_txt, wind_val_csv)
        clean_tprh_val_data(tprh_val_txt, tprh_val_csv)
        combine_wind_tprh_val(wind_val_csv, tprh_val_csv, wind_tprh_val_csv)

        print('dealing' + site)
        testScore, npwScore, bw_testScore, bw_npwScore = load_models_and_run(site, window)
        results.append([site,testScore, npwScore, bw_testScore, bw_npwScore])
        #cal_val_data(site,window,wind_tprh_val_csv,nwp_val_csv)
    
    # with open(RESULTPATH + 'final_result.csv', 'w') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(results)