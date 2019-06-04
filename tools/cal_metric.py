import torch
import json
from eval_step import eval_pre_rec

def calculate_metric(prefix, start_num, end_num):
    target_file = 'save_dir/{}_{}_{}/preds_target.json'.format(prefix, start_num, end_num)
    origin_file = 'save_dir/{}_{}_{}/preds_origin.json'.format(prefix, start_num, end_num)
    attack_file = 'save_dir/{}_{}_{}/preds_attack.json'.format(prefix, start_num, end_num)

    with open(target_file, 'r') as f:
        preds_target = json.load(f)

    with open(origin_file, 'r') as f:
        preds_origin = json.load(f)

    with open(attack_file, 'r') as f:
        preds_attack = json.load(f)

    attack = [torch.tensor(ix['caption_ix']).unsqueeze(0) for ix in preds_attack]
    attack = torch.cat(attack, 0)
    target = [torch.tensor(ix['caption_ix']).unsqueeze(0) for ix in preds_target]
    target = torch.cat(target, 0)

    precision = []
    recall = []

    for idx in range(len(target)):
        prec, rec = eval_pre_rec(target[idx, :], attack[idx, :], [])
        precision.append(prec)
        recall.append(rec)

    #precision = precision / len(target)
    #recall = recall / len(target)

    return precision, recall

def calculate_metric_hidden(prefix, start_num, end_num, is_observed=False):
    target_file = 'save_dir/{}_{}_{}/preds_target.json'.format(prefix, start_num, end_num)
    origin_file = 'save_dir/{}_{}_{}/preds_origin.json'.format(prefix, start_num, end_num)
    attack_file = 'save_dir/{}_{}_{}/preds_attack.json'.format(prefix, start_num, end_num)
    hidden_wordpos_file = 'save_dir/{}_{}_{}/hidden_wordpos.json'.format(prefix, start_num, end_num)
    
    with open(target_file, 'r') as f:
        preds_target = json.load(f)

    with open(origin_file, 'r') as f:
        preds_origin = json.load(f)

    with open(attack_file, 'r') as f:
        preds_attack = json.load(f)

    with open(hidden_wordpos_file, 'r') as f:
        hidden_wordpos = json.load(f)

    attack = [torch.tensor(ix['caption_ix']).unsqueeze(0) for ix in preds_attack]
    attack = torch.cat(attack, 0)
    target = [torch.tensor(ix['caption_ix']).unsqueeze(0) for ix in preds_target]
    target = torch.cat(target, 0)
    hidden_pos = [ix['wordpos_ix'] for ix in hidden_wordpos]
    
    precision = []
    recall = []
    for idx in range(len(target)):
        hidden_pos[idx] = [ix - 1 for ix in hidden_pos[idx]]
        prec, rec = eval_pre_rec(target[idx, :], attack[idx, :], hidden_pos[idx], is_observed)
        precision.append(prec)
        recall.append(rec)

    return precision, recall


def calculate_metric_rate(prefix, start_num, end_num):
    target_file = 'save_dir/{}_{}_{}/preds_target.json'.format(prefix, start_num, end_num)
    origin_file = 'save_dir/{}_{}_{}/preds_origin.json'.format(prefix, start_num, end_num)
    attack_file = 'save_dir/{}_{}_{}/preds_attack.json'.format(prefix, start_num, end_num)
    hidden_wordpos_file = 'save_dir/{}_{}_{}/hidden_wordpos.json'.format(prefix, start_num, end_num)

    with open(target_file, 'r') as f:
        preds_target = json.load(f)

    with open(origin_file, 'r') as f:
        preds_origin = json.load(f)

    with open(attack_file, 'r') as f:
        preds_attack = json.load(f)

    with open(hidden_wordpos_file, 'r') as f:
        hidden_wordpos = json.load(f)

    attack = [np.array(ix['caption_ix']) for ix in preds_attack]
    target = [np.array(ix['caption_ix']) for ix in preds_target]
    hidden = [np.array([0] + ix['wordpos_ix']) for ix in hidden_wordpos]

    count = 0.
    for idx in range(len(attack)):
        target_idx = target[idx]
        attack_idx = attack[idx]
        hidden_idx = hidden[idx]

        target_idx[hidden_idx] = 0
        tmp = attack_idx * (target_idx > 0)

        if np.sum(target_idx - tmp) == 0:
            count += 1

    return count
