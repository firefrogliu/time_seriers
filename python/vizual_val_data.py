from pandas import read_csv
import pandas
import numpy
import matplotlib.pyplot as plt

nwpdataframe = read_csv('/home/cheffer/workplace/wind_blow/lstm/data/site8/nwp_val.csv', usecols = ['wind','dir','u', 'v', 't', 'rh', 'psfc', 'slp'], engine='python')
nwp_dataset = nwpdataframe.values[:]
nwp_dataset = nwp_dataset.astype('float32')
preddataframe = read_csv('/home/cheffer/workplace/wind_blow/lstm/results/site8/predict_val.csv', usecols = ['wind'], engine='python')

days = len(preddataframe)/24

nwp_odd_days = []
window = 24
for day in range(int(days)):
    for hour in range(window):
        nwp_wind = nwp_dataset[(2*day + 1)*window + hour][0]
        nwp_odd_days.append(nwp_wind)

nwp_odd_days = numpy.array(nwp_odd_days)

line1, = plt.plot(preddataframe, label = 'deep learning')
line2, = plt.plot(nwp_odd_days, label = 'nwp')

plt.legend(handles = [line1, line2])
plt.show()
