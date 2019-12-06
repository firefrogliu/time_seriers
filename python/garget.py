import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


def plotlines(data: list, lengeds: list, xlabel: str = None, ylabel: str = None, savepath: str = None, show = False):
    plt.clf()
    line_legends = []
    for i in range(len(data)):
        line, = plt.plot(data[i], label = lengeds[i])
        
        line_legends.append(line)

    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles = line_legends)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()




def cal_metrics(a,b, wind_bar = 4):
    bw_a = []
    bw_b = []
    for i in range(len(a)):
        if a[i] > wind_bar:
            bw_a.append(a[i])
            bw_b.append(b[i])

    testScore = math.sqrt(mean_squared_error(a, b))
    print('Test Score: %.3f RMSE' % (testScore))

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

    a = [1,2,4,8,10]
    b = [9, 12, 3, 8, 11]

    a = np.array(a)
    print(a)
    a = np.insert(a, 0, [0] * 5)
    print(a)

    #plotlines([a,b], ['a','b'], xlabel = 'time', ylabel = 'wind', savepath= None, show = True)
