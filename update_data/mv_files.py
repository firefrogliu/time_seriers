import os
#move update date to data

appendix = [
    '_01', '_18L', '_18R', '_19', '_36L', '_36R', '_MID1', '_MID2', '_MID3'
]
for site_idx in range(9):
    site = 'site' + str(site_idx)  + '/'
    cmd_mv_nwp = 'cp NWP1_validate_2018030112_2018103112.txt ' + '../data/' + site + 'nwp.txt'  
    cmd_mv_nwp_val = 'cp NWP1_validate_2018110112_2018123012.txt ' + '../data/' + site + 'nwp_val.txt'  
    cmd_mv_tprh  = 'cp obs_tprh_2018030112_2018103112.txt ' + '../data/' + site + 'tprh.txt'
    cmd_mv_tprh_val  = 'cp obs_tprh_2018110112_2018123012.txt ' + '../data/' + site + 'tprh_val.txt'
    cmd_mv_wind = 'cp obs_wind_2018030112_2018103112' + appendix[site_idx] + '.txt ' + '../data/' + site + 'wind.txt'
    cmd_mv_wind_val = 'cp obs_wind_2018110112_2018123012' + appendix[site_idx] + '.txt ' + '../data/' + site + 'wind_val.txt'
    
    os.system(cmd_mv_nwp)
    os.system(cmd_mv_nwp_val)
    os.system(cmd_mv_tprh)
    os.system(cmd_mv_tprh_val)
    os.system(cmd_mv_wind)
    os.system(cmd_mv_wind_val)