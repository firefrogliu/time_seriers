
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
WIND_MIN = 0
WIND_MAX = 30

WIND_VAL_TXT = 'wind_val.txt'
TPRH_VAL_TXT = 'tprh_val.txt'
NWP_VAL_TXT = 'nwp_val.txt'

WIND_VAL_CSV = 'wind_val.csv'
TPRH_VAL_CSV = 'tprh_val.csv'
NWP_VAL_CSV = 'nwp_val.csv'

WIND_TPRH_VAL_CSV = 'wind_tprh_val.csv'
WIND_TPRH_NWP_VAL_CSV = 'wind_tprh_nwp_val.csv'

WIND_TPRH_NWP_ALLYEAR_CSV = 'wind_tprh_nwp_allyear.csv'

TRAIN_TEST_SIZE = 5904

NWP_START_INDEX = 6 #tell where the nwp info start in the wind_tprh_nwp.csv file

UPDATE_OBS_RAW = 'obs_update.txt'
UPDATE_OBS_VAL_RAW = 'obs_val_update.txt'
UPDATE_OBS_CSV = 'obs_update_allyear.csv'

UPDATA_OBS_NWP_ALLYEAR_CSV = 'obs_update_nwp_allyear.csv'

CSV_COLUMNS = ['wind','dir','slp','t2', 'rh2', 'td2', 'nwp_wind','nwp_dir','nwp_u', 'nwp_v', 'nwp_t', 'nwp_rh', 'nwp_psfc', 'nwp_slp', 'residual']

VMD_MODES = 5