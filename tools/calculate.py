import torch
import json
from eval_step import eval_pre_rec

def calculate_word_rate(prefix, start_num, end_num):
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

    res1 = (attack - target) != 0
    res2 = target != 0

    numerator = torch.sum(res1, dim=0)
    denominator = torch.sum(res2, dim=0)

    return numerator, denominator

def print_word_rate(prefix):
    num1, den1 = calculate_word_rate(prefix, 0, 250)
    num2, den2 = calculate_word_rate(prefix, 250, 500)
    num3, den3 = calculate_word_rate(prefix, 500, 750)
    num4, den4 = calculate_word_rate(prefix, 750, 1000)

    num = num1 + num2 + num3 + num4
    den = den1 + den2 + den3 + den4
    
    error = num
    num = den - num
    right = num

    num = num.data.cpu().numpy().tolist()
    den = den.data.cpu().numpy().tolist()

    right = right.data.cpu().numpy().tolist()
    error = error.data.cpu().numpy().tolist()

    result = []
    for i in range(len(num)):
        if den[i] == 0:
            result.append(0)
        else:
            result.append(round(num[i] * 1.0 / den[i], 4))

    print(right)
    print(error)
    print(den)
    print(result)

def calculate_word_rate_observed(prefix, start_num, end_num):
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
    hidden = [torch.tensor([0] + ix['wordpos_ix']) for ix in hidden_wordpos]

    for idx in range(len(hidden)):
        target[idx, hidden[idx]] = 0
        attack[idx ,:] = attack[idx ,:] * (target[idx ,:] > 0).long()

    res1 = (attack - target) != 0
    res2 = target != 0

    numerator = torch.sum(res1, dim=0)
    denominator = torch.sum(res2, dim=0)

    return numerator, denominator

def print_word_rate_observed(prefix):
    num1, den1 = calculate_word_rate_observed(prefix, 0, 250)
    num2, den2 = calculate_word_rate_observed(prefix, 250, 500)
    num3, den3 = calculate_word_rate_observed(prefix, 500, 750)
    num4, den4 = calculate_word_rate_observed(prefix, 750, 1000)

    num = num1 + num2 + num3 + num4
    den = den1 + den2 + den3 + den4
    
    error = num
    num = den - num
    right = num

    num = num.data.cpu().numpy().tolist()
    den = den.data.cpu().numpy().tolist()

    right = right.data.cpu().numpy().tolist()
    error = error.data.cpu().numpy().tolist()

    result = []
    for i in range(len(num)):
        if den[i] == 0:
            result.append(0)
        else:
            result.append(round(num[i] * 1.0 / den[i], 4))

    print(right)
    print(error)
    print(den)
    print(result)
