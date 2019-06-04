import numpy as np
from cal_metric import calculate_metric, calculate_metric_hidden

def calculate_eval_rec(prefix):
    pre1, rec1 = calculate_metric(prefix, 0, 250)
    pre2, rec2 = calculate_metric(prefix, 250, 500)
    pre3, rec3 = calculate_metric(prefix, 500, 750)
    pre4, rec4 = calculate_metric(prefix, 750, 1000)

    precision = pre1 + pre2 + pre3 + pre4
    recall = rec1 + rec2 + rec3 + rec4
    print(len(precision), len(recall))

    print('precision', np.mean(precision), np.std(precision))
    print('recall', np.mean(recall), np.std(recall))

def calculate_eval_rec_hidden(prefix, is_observed=False):
    pre1, rec1 = calculate_metric_hidden(prefix, 0, 250, is_observed)
    pre2, rec2 = calculate_metric_hidden(prefix, 250, 500, is_observed)
    pre3, rec3 = calculate_metric_hidden(prefix, 500, 750, is_observed)
    pre4, rec4 = calculate_metric_hidden(prefix, 750, 1000, is_observed)

    precision = pre1 + pre2 + pre3 + pre4
    recall = rec1 + rec2 + rec3 + rec4
    print(len(precision), len(recall))

    print('precision', np.mean(precision), np.std(precision))
    print('recall', np.mean(recall), np.std(recall))

def calculate_eval_rec_observed(prefix, is_observed=True):
    pre1, rec1 = calculate_metric_hidden(prefix, 0, 250, is_observed)
    pre2, rec2 = calculate_metric_hidden(prefix, 250, 500, is_observed)
    pre3, rec3 = calculate_metric_hidden(prefix, 500, 750, is_observed)
    pre4, rec4 = calculate_metric_hidden(prefix, 750, 1000, is_observed)

    precision = pre1 + pre2 + pre3 + pre4
    recall = rec1 + rec2 + rec3 + rec4
    print(len(precision), len(recall))

    print('precision', np.mean(precision), np.std(precision))
    print('recall', np.mean(recall), np.std(recall))
