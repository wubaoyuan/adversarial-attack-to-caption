=================================================================
---------------------some useful tools---------------------------
=================================================================

path: cal_metric.py
      1) func: calculate_metric
               calculate precision and recall for one subset dataset about target complete caption
      2) func: calculate_metric_hidden
               calculate precision and recall for one subset dataset about target partial caption
      3) func: calculate_metric_rate
               calculate the number of successful samples

path: calculate_pre_rec_all.py
      1) func: calculate_eval_rec, calculate_eval_rec_hidden, calculate_eval_rec_observed
               calculate precision and recall for all 1000 val data

path: calculate.py
      1) func: print_word_rate, print_word_rate_observed
               calculate word rate on different word position

