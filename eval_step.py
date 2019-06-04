from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
import random
import string
import time
import os
import misc.utils as utils

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_print(loader, data, seq, predictions, n, eval_kwargs={}):
    beam_size = eval_kwargs.get('beam_size', 1)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    verbose = eval_kwargs.get('verbose', True)

    # print beam search
    if beam_size > 1 and verbose_beam:
        for i in range(loader.batch_size):
            print('\n'.join(utils.decode_sequence(loader.get_vocab(),
                    _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]))
            print('---' * 10)
    sents = utils.decode_sequence(loader.get_vocab(), seq)

    for k, sent in enumerate(sents):
        entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
        predictions.append(entry)
        if eval_kwargs.get('dump_images', 0) == 1:
            # dump the raw image to vis/ folder
            cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
            print(cmd)
            os.system(cmd)

        if verbose:
            print('image %s: %s' %(entry['image_id'], entry['caption']))

    # if we wrapped around the split or used up val imgs budget then bail
    ix0 = data['bounds']['it_pos_now']
    ix1 = data['bounds']['it_max']
    if num_images != -1:
        ix1 = min(ix1, num_images)
    for i in range(n - ix1):
        predictions.pop()

    if verbose:
        if ix0 % 200 == 0: 
            print('evaluating validation preformance... %d/%d' %(ix0, ix1)) if ix0 else \
                print('evaluating validation preformance... %d/%d' %(ix1, ix1))

    if data['bounds']['wrapped']:
        return True
    if num_images >= 0 and n >= num_images:
        return True

    return False

def eval_print_caption(loader, info, seq, predictions):
    # TODO: beam search > 1
    sent = utils.decode_sequence(loader.get_vocab(), seq)[0]
    seq_ = seq.squeeze().data.cpu().numpy()
    entry = {'image_id': info['id'], 'caption': sent, 'caption_ix': seq_.tolist()}
    predictions.append(entry)
    print('image %s: %s' %(entry['image_id'], entry['caption']))

def eval_print_wordpos(loader, info, wordpos, observed, seq_f, hidden_wordpos, res, 
                                                noun, verb, adjective, adverb):
    entry = {'image_id': info['id'], 'word': [], 'wordpos': [], 'word_f': [], 'wordpos_ix': []}
    for pos in wordpos:
        word_idx = observed[pos + 1]
        word_idx_f = seq_f.squeeze()[pos + 1]
        res[pos + 1] = 0
        word_idx_ = word_idx.squeeze().data.cpu().numpy()
        word_idx_f_ = word_idx_f.squeeze().data.cpu().numpy()
        word = loader.ix_to_word[str(word_idx_)]
        if word_idx_f_:
            word_f = loader.ix_to_word[str(word_idx_f_)]
        else:
            word_f = '<eos>'
        if word in noun:
            print('pos_ix: ', int(pos) + 1, ', ', word, ': noun  --> ', word_f, end=', ')
            entry['word'].append(word)
            entry['word_f'].append(word_f)
            entry['wordpos'].append(0)
            entry['wordpos_ix'].append(int(pos) + 1)
        elif word in verb:
            print('pos_ix: ', int(pos) + 1, ', ', word, ': verb  --> ', word_f, end=', ')
            entry['word'].append(word)
            entry['word_f'].append(word_f)
            entry['wordpos'].append(1)
            entry['wordpos_ix'].append(int(pos) + 1)
        elif word in adjective:
            print('pos_ix: ', int(pos) + 1, ', ', word, ': adjective  --> ', word_f, end=', ')
            entry['word'].append(word)
            entry['word_f'].append(word_f)
            entry['wordpos'].append(2)
            entry['wordpos_ix'].append(int(pos) + 1)
        elif word in adverb:
            print('pos_ix: ', int(pos) + 1, ', ', word, ': adverb  --> ', word_f, end=', ')
            entry['word'].append(word)
            entry['word_f'].append(word_f)
            entry['wordpos'].append(3)
            entry['wordpos_ix'].append(int(pos) + 1)
        else:
            print('pos_ix: ', int(pos) + 1, ', ', word, ': others  --> ', word_f, end=', ')
            entry['word'].append(word)
            entry['word_f'].append(word_f)
            entry['wordpos'].append(4)
            entry['wordpos_ix'].append(int(pos) + 1)
    hidden_wordpos.append(entry)

def eval_hamming_dist(s_observed, seq_f, wordpos):
    observed = s_observed.clone()
    obs_idx = torch.nonzero(s_observed).squeeze()
    seq = seq_f.clone()
    for pos in wordpos:
        observed[pos + 1] = 0
        seq[pos + 1] = 0
    if torch.sum(observed - seq) == 0:
        return 1.
    else:
        sum_ = torch.sum((observed - seq) == 0)
        return float(sum_) / len(obs_idx)

def eval_pre_rec(s_observed, seq_f, wordpos, is_observed=False):
    observed = s_observed.clone()
    len_obs = torch.max(torch.nonzero(observed).squeeze()) + 1
    seq = seq_f.clone()
    len_seq = torch.max(torch.nonzero(seq).squeeze()) + 1
    count = 0.
    if len(wordpos) != 0 and observed[wordpos[-1] + 1] == 0:
        wordpos.pop()
    for pos in wordpos:
        observed[pos + 1] = 0
        if pos + 1 <= len_seq - 1:
            seq[pos + 1] = 0
            count += 1
    if is_observed:  # first word is always hidden
        observed[0] = 0
        seq[0] = 0
        count += 1
    res = (observed - seq) == 0
    len_seq = len_obs if len_seq > len_obs else len_seq
    if not is_observed:
        assert (torch.sum(res[:len_seq]) - count) == (torch.sum(res[:len_obs]) - len(wordpos)), print(torch.sum(res[:len_seq]), count, torch.sum(res[:len_obs]), len(wordpos), s_observed, seq_f, len_obs, len_seq, wordpos, res)
        pre = (torch.sum(res[:len_seq]) - count).float() / (len_seq - count).float()
        rec = (torch.sum(res[:len_obs]) - len(wordpos)).float() / (len_obs - len(wordpos)).float()
    else:
        assert (torch.sum(res[:len_seq]) - count) == (torch.sum(res[:len_obs]) - len(wordpos) - 1)
        if len_seq - count != 0:
            pre = (torch.sum(res[:len_seq]) - count).float() / (len_seq - count).float()
        else:
            pre = torch.tensor(0.)
        rec = (torch.sum(res[:len_obs]) - len(wordpos) - 1).float() / (len_obs - len(wordpos) - 1).float()
    
    assert pre >= rec
    return pre.data.cpu().numpy(), rec.data.cpu().numpy()
