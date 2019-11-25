import os

#create folder
folders = ['data/', 'results/', 'models/']

for folder in folders:
    for site_idx in range(2,10):
        site = 'site' + str(site_idx)  + '/'
        path = folder + site
        os.mkdir(path)

