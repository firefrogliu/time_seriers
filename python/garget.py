import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.ticker as ticker
from pandas import read_csv
from constants import *


def plotlines(data: list, lengeds: list, xlabel: str = None,  ylabel: str = None, xticks = None, xtick_space = 1, display_lenth = -1, savepath: str = None, show = False, figsize = (16,9)):
    plt.clf()
    line_legends = []
    fig = plt.figure(figsize= figsize)
    ax =  fig.add_subplot()  
    for i in range(len(data)):
        if xticks is not None:
            line, = ax.plot(xticks[:display_lenth], data[i][:display_lenth], label = lengeds[i])
        else:
            line, = ax.plot(data[i][:display_lenth], label = lengeds[i])
        line_legends.append(line)

    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles = line_legends)
    if xtick_space != 1:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_space))
    #ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)    
    if show:
        plt.show()

def plotlines_multifigue(data: list, lengeds: list = None, xlabel: str = None,  ylabel: str = None, xticks = None, xtick_space = 1, display_lenth = -1,savepath: str = None, show = False, figsize = (16,9)):
    plt.clf()
    print('datalen is', len(data))
    fig, axs = plt.subplots(len(data), figsize= figsize)
    
    print('lengeds is None')
    if lengeds is None:
        lengeds = []
        for i in range(len(data)):
            legend = []
            for j in range(len(data[i])):
                legend.append('None')
            lengeds.append(legend)

    for i in range(len(data)):
        line_legends = []    
        ax = axs[i]
        for j in range(len(data[i])):
            if xticks is not None:
                
                line, = ax.plot(xticks[:display_lenth], data[i][j][:display_lenth], label = lengeds[i][j])
            else:
                line, = ax.plot(data[i][j], label = lengeds[i][j])
            line_legends.append(line)
        ax.legend(handles = line_legends)
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_space))

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')    
    plt.grid(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)    
    if show:
        plt.show()

def read_dataframe(csvfile, datastart, dataend, ycolumn = 'wind'):
    print('reawding', csvfile)
    dataframe = read_csv(csvfile, usecols = CSV_COLUMNS,engine='python')
    
    dataset = dataframe.values[:]
    dataset = dataset.astype('float32')
    
    dataframeY = read_csv(csvfile, usecols=[ycolumn], engine='python')
    datasetY = dataframeY.values[:]
    datasetY = datasetY.astype('float32')

    if dataend == 'end':
        dataset = dataset[datastart:, :]
        datasetY = datasetY[datastart:, :]
    else:   
        dataset = dataset[datastart:dataend, :]
        datasetY = datasetY[datastart:dataend, :]

    return dataset, datasetY

def cal_metrics(a,b, wind_bar = 4):
    bw_a = []
    bw_b = []
    for i in range(len(a)):
        if a[i] > wind_bar:
            bw_a.append(a[i])
            bw_b.append(b[i])

    testScore = math.sqrt(mean_squared_error(a, b))
    print('Test Score: %.3f RMSE' % (testScore))

    bw_testScore = 0.01
    if len(bw_a) > 0:
        bw_testScore = math.sqrt(mean_squared_error(bw_a, bw_b))
        print('Big wind Test Score: %.3f RMSE' % (bw_testScore))

    return testScore, bw_testScore

def compare_nwp_dpl(obs, nwp, dpl):
    score_nwp, score_bw_nwp = cal_metrics(obs, nwp)
    score_pre, score_bw_pre = cal_metrics(obs, dpl)

    score_up = (score_nwp - score_pre)/score_nwp
    score_bw_up = (score_bw_nwp - score_bw_pre)/score_bw_nwp
    
    print('score_nwp, score_pre, up', score_nwp, score_pre, score_up * 100)
    print('score_bw_nwp, score_bw_pre, up_bw', score_bw_nwp, score_bw_pre, score_bw_up * 100)

    return score_nwp, score_pre, score_up, score_bw_nwp, score_bw_pre, score_bw_up


def cal_correlation(a,b):    
    coffs = np.corrcoef(a, b)
    #print(coffs)
    #plotlines([a,b],['a','b'], show = True )
    return coffs[0,1]

if __name__ == '__main__':

    # a = [1,2,4,8,10]
    # b = [9, 12, 3, 8, 11]

    # a = np.array(a)
    # print(a)
    # a = np.insert(a, 0, [0] * 5)
    # print(a)

    #plotlines([a,b], ['a','b'], xlabel = 'time', ylabel = 'wind', savepath= None, show = True)

    import numpy as np
    import matplotlib.pyplot as plt


    x = ['a','b','c','d','e']
    y = [0,1,2,3,4]
    y1 = [2,5,6,6,8,]
    z = [1.2,3.3,4,5,7]
    z0 = [2,8,2,3,4]
    z1 = z0[:-1]
    print(z1)
    #plotlines_multifigue([[y, y1],[z,z0]], [['y', 'y1'],['z','z0']], xlabel = 'time', ylabel = 'wind', xticks=x, xtick_space=1, show = True)
