import os
#move update date to data

appendix = [
    '_01', '_18L', '_18R', '_19', '_36L', '_36R', '_MID1', '_MID2', '_MID3'
]
for site_idx in range(9):
    site = 'site' + str(site_idx)  + '/'

    cmd_mv_update_obs = 'cp qinghua_2018030200_2018110100_site' + appendix[site_idx] + '.txt ' + '../../data/' + site + 'obs_update.txt'
    cmd_mv_update_val_obs = 'cp qinghua_2018110200_2018123100_site' + appendix[site_idx] + '.txt ' + '../../data/' + site + 'obs_val_update.txt'
    
    os.system(cmd_mv_update_obs)
    os.system(cmd_mv_update_val_obs)