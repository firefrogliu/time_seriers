#!/usr/bin/env python
# encoding: utf-8
"""
@author Yu Xing
for calculation of RMSE/STD/BIAS/CORR/MAE  and plt
fcst VS obs  
"""
import sys
import os 
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from   pylab  import mpl
from   datetime import datetime 
import time
import matplotlib.dates as mdate
from   matplotlib import font_manager
import matplotlib
import math
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题
def read_file(infile):
    data=np.loadtxt(infile,float)
    data_all=data.reshape(-1,1)
    data_all=pd.DataFrame(data_all)
    data_all.columns=['all']
    data=pd.DataFrame(data)
    data.index=ti
    data.columns=columns
    return data,data_all

def _get_dataframe_metrics(df1, df2):
    delta = (df2 - df1).mean()
    mean1=df1.mean()
    mean2=df2.mean()
    mae=np.abs(df1-df2).mean()
    rmse = np.sqrt(((df2 - df1) ** 2).mean())
    stdv = (rmse**2.-delta**2.)**0.5
    df1.columns=['obs']
    df2.columns=['fcst']
    df3=df1.join(df2)
    corre=df3.corr().iloc[0][1]
    return delta,rmse,corre,mean1,mean2,stdv,mae
   
    
def plot_obs(data,outfig):
    ymax=18
    fig,ax=plt.subplots(1,1,figsize=(15,10))
    data.iloc[:,0].plot(kind='line',style='r',use_index=True)
    data.iloc[:,1].plot(kind='line',style='b',use_index=True)
    ax.legend(loc='upper left',fontsize=25)
    ax.set_title(testname+"  "+list(data)[0],fontsize=30)
    ax.set_ylabel('WindSpeed(m/s)',fontsize=25)
    ax.set_xlabel('11/12 2018',fontsize=25)
    ax.set_ylim(0, ymax)
    ax.tick_params(axis='x',labelsize=20)
    ax.tick_params(axis='y',labelsize=20)
    ax.grid(True, which='both',linestyle='-.')
    ix=20
    ax.text(data.index[ix+65],14,clegend1,fontsize=25)
    ax.text(data.index[ix],13,clegend[0],fontsize=25)
    ax.text(data.index[ix],12,clegend[1],fontsize=25)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d %H'))
    ax.set_xticks(pd.date_range(data.index[0],data.index[-1],freq='2d'))
    tii=data.index.to_series().astype(str)
    xti = [ x[8:10] for x in tii ] 
    ax.set_xticklabels(xti[0:len(data):24],rotation=0)
    plt.show()
    plt.savefig(outfig,dpi=500)
    
plt.switch_backend('agg')
#testname="EC"
testname="EMD"
threshold=[0,3]
number_of_days=29
start_time=datetime(2018,11,3,0)
ti = pd.date_range(start_time, periods=24 * number_of_days, freq='2h')
columns=['18R','36L','MID1','18L','36R','MID2','01','19','MID3']
columnsfcst=['fcst_18R','fcst_36L','fcst_MID1','fcst_18L','fcst_36R','fcst_MID2','fcst_01','fcst_19','fcst_MID3']
spd_obs,spd_obs_all=read_file('./'+testname+'_obs.dat')
spd_fcst,spd_fcst_all=read_file('./'+testname+'_fcst.dat')
obs_ave=pd.DataFrame(spd_obs.mean(axis=1))
fcst_ave=pd.DataFrame(spd_fcst.mean(axis=1))
obs_ave.columns=['mean']
fcst_ave.columns=['fcst_mean']
pltmean=obs_ave.join(fcst_ave)

jloc=1
clegend=[]
for i in [0,1]:
    ths=threshold[i]
    spd_obs_all.replace( spd_obs_all < ths, np.nan,inplace = True)
    obs_spd = spd_obs_all [ spd_obs_all >= ths ]
    fcst_spd=spd_fcst_all [ spd_obs_all >= ths ]
    
    obs_spd.dropna (inplace=True)
    fcst_spd.dropna(inplace=True)
    delta,rmse,corre,mean1,mean2,std,mae=_get_dataframe_metrics(obs_spd, fcst_spd)
    clegend1="RMSE   STDV   BIAS  CORR  MAE  MOBS  MFCT"
    clegend.append(">%1.0fm/s%6.3f%6.3f%7.3f%6.3f%6.3f%6.2f%6.2f" %(ths,rmse,std,delta,corre,mae,mean1,mean2))
    print(clegend1)
    print(clegend[i])
plot_obs(pltmean,testname+"_mean.png")
with open(testname+'_score_all.txt', 'w') as f:
    f.writelines(clegend[0]+'\n')
    f.writelines(clegend[1]+'\n')
plt.close()
sys.exit()
