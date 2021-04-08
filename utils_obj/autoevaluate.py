#!/usr/bin/python3
import pandas as pd
from tslearn.metrics import dtw, dtw_path
import statistics


def autoeval(path_new, path_old):
#path_new = "/content/metrics_new.csv"
#path_old = "/content/metrics_old.csv"
#----------------------------------------moving average to smoothen curves
    csv_new = pd.read_csv(path_new)
    csv_new['P_s_new'] = csv_new['P'].rolling(7,min_periods=1).sum()
    csv_new['R_s_new'] = csv_new['R'].rolling(7,min_periods=1).sum()
    csv_new['map05_s_new'] = csv_new['map05'].rolling(7,min_periods=1).sum()
    csv_new['map9_s_new'] = csv_new['map9'].rolling(7,min_periods=1).sum()

    metrics_smooth_new = csv_new[['P_s_new', 'R_s_new', 'map05_s_new','map9_s_new']].copy()


    csv_old = pd.read_csv(path_old)
    csv_old['P_s_old'] = csv_old['P'].rolling(7,min_periods=1).sum()
    csv_old['R_s_old'] = csv_old['R'].rolling(7,min_periods=1).sum()
    csv_old['map05_s_old'] = csv_old['map05'].rolling(7,min_periods=1).sum()
    csv_old['map9_s_old'] = csv_old['map9'].rolling(7,min_periods=1).sum()

    metrics_smooth_old = csv_old[['P_s_old', 'R_s_old', 'map05_s_old','map9_s_old']].copy()

    p_list_n = metrics_smooth_new['P_s_new']
    p_list_o = metrics_smooth_old['P_s_old']

    r_list_n = metrics_smooth_new['R_s_new']
    r_list_o = metrics_smooth_old['R_s_old']

    m05_list_n = metrics_smooth_new['map05_s_new']
    m05_list_o = metrics_smooth_old['map05_s_old']

    map9_list_n = metrics_smooth_new['map9_s_new']
    map9_list_o = metrics_smooth_old['map9_s_old']

#-------------------------moving average-----------------------------

#---------------differences-------------------------
    dif = []

    p_dtw = dtw(p_list_n, p_list_o)
    dif.append(p_dtw)
    r_dtw = dtw(r_list_n, r_list_o)
    dif.append(r_dtw)

    m05_dtw = dtw(m05_list_n, m05_list_o)
    dif.append(m05_dtw)
    map9_dtw = dtw(map9_list_n, map9_list_o)
    dif.append(map9_dtw)

    avg_score = statistics.mean(dif)
#------------------differences---------------------
    treshold = 0.1

    if avg_score > treshold and m05_list_n.iat[-1] > m05_list_o.iat[-1]:
      print("Use new model because new model's acurracy and its curve is better")
    else:
      print("Use previous model")