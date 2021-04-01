#!/usr/bin/python3

#kalman filter implementation is from: 
#https://stackoverflow.com/questions/48739169/how-to-apply-a-rolling-kalman-filter-to-a-column-in-a-dataframe


import pandas as pd
from pykalman import KalmanFilter
import numpy as np



df = pd.read_csv('metrics.csv')
#-------------------------------------kalman filtering----------------------------------
def rolling_window(a, step):
    shape   = a.shape[:-1] + (a.shape[-1] - step + 1, step)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_kf_value(y_values):
    kf = KalmanFilter()
    Kc, Ke = kf.em(y_values, n_iter=1).smooth(0)
    return Kc
wsize = 3
arr = rolling_window(df.P.values, wsize)
zero_padding = np.zeros(shape=(wsize-1,wsize))
arrst = np.concatenate((zero_padding, arr))
arrkalman = np.zeros(shape=(len(arrst),1))

for i in range(len(arrst)):
    arrkalman[i] = get_kf_value(arrst[i])

kalmandf_p = pd.DataFrame(arrkalman, columns=['P_k'])



arr = rolling_window(df.R.values, wsize)
zero_padding = np.zeros(shape=(wsize-1,wsize))
arrst = np.concatenate((zero_padding, arr))
arrkalman = np.zeros(shape=(len(arrst),1))
for i in range(len(arrst)):
    arrkalman[i] = get_kf_value(arrst[i])

kalmandf_r = pd.DataFrame(arrkalman, columns=['R_k'])
kalmandf_r= pd.concat([kalmandf_p,kalmandf_r], axis=1)


arr = rolling_window(df.map05.values, wsize)
zero_padding = np.zeros(shape=(wsize-1,wsize))
arrst = np.concatenate((zero_padding, arr))
arrkalman = np.zeros(shape=(len(arrst),1))
for i in range(len(arrst)):
    arrkalman[i] = get_kf_value(arrst[i])

kalmandf_05 = pd.DataFrame(arrkalman, columns=['map05_k'])
kalmandf_05 = pd.concat([kalmandf_r,kalmandf_05], axis=1)

arr = rolling_window(df.map9.values, wsize)
zero_padding = np.zeros(shape=(wsize-1,wsize))
arrst = np.concatenate((zero_padding, arr))
arrkalman = np.zeros(shape=(len(arrst),1))
for i in range(len(arrst)):
    arrkalman[i] = get_kf_value(arrst[i])

kalmandf_9 = pd.DataFrame(arrkalman, columns=['map90_k'])
kalmandf_last = pd.concat([kalmandf_05,kalmandf_9], axis=1)



emptydf = np.zeros(shape=(len(arrst),1))
kalmandf_dif_p = pd.DataFrame(kalmandf_last["P_k"]-df["P"], columns=['P dif'])
kalmandf_dif_r = pd.DataFrame(kalmandf_last["R_k"]-df["R"], columns=['R dif'])
kalmandf_dif_05 = pd.DataFrame(kalmandf_last["map05_k"]-df["map05"], columns=['map05 dif'])
kalmandf_dif_90 = pd.DataFrame(kalmandf_last["map90_k"]-df["map9"], columns=['map90 dif'])
kalmandf_differences = pd.concat([kalmandf_dif_p,kalmandf_p], axis=1)
kalmandf_differences = pd.concat([kalmandf_dif_r,kalmandf_differences], axis=1)
kalmandf_differences = pd.concat([kalmandf_dif_05,kalmandf_differences], axis=1)
kalmandf_differences = pd.concat([kalmandf_dif_90,kalmandf_differences], axis=1)
#----------------------------------kalman filtering------------------------------------------------------
print(kalmandf_last)
print(df)
print(kalmandf_differences)
df.to_csv("/content/metrics_kalman.csv")