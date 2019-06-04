from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as numpy
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import os

import opts
import models
from dataloader import *
from eval_step import eval_print_caption, eval_pre_rec

import skimage.io
from skimage.transform import resize
from misc import resnet_utils, inception_utils
import misc.resnet as resnet
import misc.inception as inception

def latent_structural_svm(model, optimizer, img, epsilon, s_observed, lamda, max_iters):
    for i in range(max_iters):
        # latent variable completion
        s_hidden = torch.zeros_like(s_observed)

        # structural SVM
        for j in range(max_iters):
            # (1) loss augmented inference
            with torch.no_grad():
                s_hidden_, s_observed_ = model(img, epsilon, s_observed=s_observed, 
                                                            mode='inference_lai', slack=opt.slack)
            # (2) update epsilon by gradient descent
            loss1 = model(img, epsilon, s_observed=s_observed_, s_hidden=s_hidden_, mode='inference_gd')
            loss2 = model(img, epsilon, s_observed=s_observed, s_hidden=s_hidden, mode='inference_gd')
            loss3 = lamda * torch.sum(epsilon * epsilon)
            loss = loss1 - loss2 + loss3

            # gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return epsilon

def maxloglikelihood_e_step(model, img, epsilon, s_observed, topk=5):
    obs_idx = torch.nonzero(s_observed).squeeze(1)
    # if last obs_idx is end of caption, need add a <eos>
    max_idx = torch.max(obs_idx)
    if max_idx != len(s_observed) - 1 and s_observed[max_idx + 1] == 0:
        tmp = s_observed.clone()
        tmp[max_idx + 1] = -1
        obs_idx = torch.nonzero(tmp).squeeze(1)
        max_idx = torch.max(obs_idx)
    max_step = max_idx + 1
    qs_list = []
    qs_inds = []
    
    for t in range(max_step):
        if t != max_step - 1:
            with torch.no_grad():
                qs = model(img, epsilon, qs_list=qs_list, qs_inds=qs_inds, mode='inference_e')
        if t in obs_idx:
            qs = torch.zeros_like(qs)
            qs[:, s_observed[t]] = 1.
            qs_inds.append([s_observed[t]])
            qs_list.append([1.])
        else:
            qs_topk, qs_ind = torch.topk(qs, topk, dim=1)
            qs_inds.append(qs_ind.squeeze(0))
            qs_list.append(qs_topk.squeeze(0))

    return qs_list, qs_inds

def maxloglikelihood_m_step(model, optimizer, img, epsilon, s_observed, qs, qs_inds, lamda):
    obs_idx = torch.nonzero(s_observed).squeeze(1)
    # if last obs_idx is end of caption, need add a <eos>
    max_idx = torch.max(obs_idx)
    if max_idx != len(s_observed) - 1 and s_observed[max_idx + 1] == 0:
        tmp = s_observed.clone()
        tmp[max_idx + 1] = -1
        obs_idx = torch.nonzero(tmp).squeeze(1)
        max_idx = torch.max(obs_idx)
    max_step = max_idx + 1
    
    loss = 0.
    for t in range(max_step):
        loss_value = model(img, epsilon, qs_list=qs, qs_inds=qs_inds, t=t, mode='inference_m')
        loss += loss_value
    loss = lamda * torch.sum(epsilon * epsilon) - loss

    # gradient ascent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return epsilon

def attack(opt):
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    rnn_net = models.setup(opt).cuda()
    rnn_net.load_state_dict(torch.load(opt.pretrain_model))

    if not opt.caption_model == 'show_tell':
        net = getattr(resnet, 'resnet101')()
        net.load_state_dict(torch.load('data/imagenet_weights/resnet101.pth'))
        model = resnet_utils.combineNet(net, rnn_net)
        model.cuda()
        model.resnet.eval()
        model.rnn_net.train()
    else:
        net = getattr(inception, 'inception_v3')()
        net.load_state_dict(torch.load('data/imagenet_weights/inception_v3.pth'))
        model = inception_utils.combineNet(net, rnn_net)
        model.cuda()
        model.inception.eval()
        model.rnn_net.train()

    eval_kwargs = {'split': 'attack', 'dataset': opt.input_json}
    eval_kwargs.update(vars(opt))
    split = eval_kwargs.get('split', 'attack')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    n = 0
    preds_origin = []
    preds_attack = []
    preds_target = []
    max_iters = 50 if not opt.is_ssvm else 10
    lr = 1e-3
    lamda = opt.lamda
    count = 0
    eps_l2 = 0.
    precision = 0.
    recall = 0.
    total_time = 0.
    print('lambda: ', opt.lamda, '  zeta: ', opt.slack)

    val_data = loader.val_data[opt.start_num: opt.end_num]
    print('using EM algorithm...') if not opt.is_ssvm else print('using structural SVM...')
    for i, info in enumerate(val_data):
        # attack image
        I = skimage.io.imread(os.path.join('data/images', info['file_path']))
        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)
        if opt.caption_model == 'show_tell':
            I = resize(I, (299, 299))
            I = torch.from_numpy(I.transpose([2, 0, 1])).float().cuda()
        else:
            I = I.astype('float32') / 255.0
            I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
        img = I.clone()
        eps = torch.zeros_like(I).cuda()
        with torch.no_grad():
            seq, _, attention = model(I, eps, opt=eval_kwargs, mode='sample')
        print('attack caption: ')
        eval_print_caption(loader, info, seq[:, 1:], preds_origin)
        
        # target images
        target_data = info['target']
        target_seq = []
        target_img = []
        fake_attentions = []
        epsilons = []
        print('target caption: ')
        for i_t, info_t in enumerate(target_data):
            I_t = skimage.io.imread(os.path.join('data/images', info_t['file_path']))
            # handle grayscale input images
            if len(I_t.shape) == 2:
                I_t = I_t[:, :, np.newaxis]
                I_t = np.concatenate((I_t, I_t, I_t), axis=2)
            if opt.caption_model == 'show_tell':
                I_t = resize(I_t, (299, 299))
                I_t = torch.from_numpy(I_t.transpose([2, 0, 1])).cuda().float()
            else:
                I_t = I_t.astype('float32') / 255.0
                I_t = torch.from_numpy(I_t.transpose([2, 0, 1])).cuda()
            target_img.append(I_t.clone())
            eps_t = torch.zeros_like(I_t).cuda()
            with torch.no_grad():
                seq_t, _, _ = model(I_t, eps_t, opt=eval_kwargs, mode='sample')
            eval_print_caption(loader, info_t, seq_t[:, 1:], preds_target)
            target_seq.append(seq_t)
        target_seq = torch.cat(target_seq, 0)

        # attack
        s_observed = target_seq[:, 1:]
        print('evaluate after attack: ')
        for idx in range(len(target_data)):
            epsilon = torch.ones_like(img).cuda() / 255.
            img = Variable(img, requires_grad=True)
            epsilon = Variable(epsilon, requires_grad=True)
            optimizer = optim.Adam([epsilon], lr=lr)
            tmp_observed = s_observed[idx, :].clone()
	        
            if not opt.is_ssvm:
                start_time = time.time()
                for _ in range(max_iters):
                    qs, qs_inds = maxloglikelihood_e_step(model, img, epsilon, tmp_observed) 
                    epsilon = maxloglikelihood_m_step(model, optimizer, img, epsilon, 
                                                        tmp_observed, qs, qs_inds, lamda)
                end_time = time.time()
                total_time += end_time - start_time
            else:
                start_time = time.time()
                epsilon = latent_structural_svm(model, optimizer, img, epsilon,
                                                        tmp_observed, lamda, max_iters)
                end_time = time.time()
                total_time += end_time - start_time

            # evaluate after attack
            eps_l2 += torch.norm(epsilon.view(-1), p=2)
            img = Variable(img, requires_grad=False)
            eps = Variable(epsilon.clone(), requires_grad=False)
            with torch.no_grad():
                seq_f, _, fake_attention = model(img, eps, opt=eval_kwargs, mode='sample')
            if torch.sum(seq_f[:, 1:] - tmp_observed) == 0:
                count += 1
            prec, rec = eval_pre_rec(tmp_observed, seq_f[:, 1:].squeeze(), [])
            print(prec, rec)
            precision += prec
            recall += rec
            fake_attentions.append(fake_attention)
            epsilons.append(eps.clone())
            eval_print_caption(loader, info, seq_f[:, 1:], preds_attack)
        tmp_total = (i + 1) * len(target_data)
        print('{}/{}, average l2_norm: {}'.format(count, tmp_total, eps_l2 / tmp_total))
        print('precision: {}, recall: {}'.format(precision / tmp_total, recall / tmp_total))
        print('average time: {}'.format(total_time / ((i + 1) * len(target_data))))
        torch.cuda.empty_cache()

        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        with open(os.path.join(opt.save_dir, 'preds_origin.json'), 'w') as f:
            json.dump(preds_origin, f)
        with open(os.path.join(opt.save_dir, 'preds_target.json'), 'w') as f:
            json.dump(preds_target, f)
        with open(os.path.join(opt.save_dir, 'preds_attack.json'), 'w') as f:
            json.dump(preds_attack, f)

        if opt.save_as_images:
            tmp_real = img.permute([1, 2, 0]).data.cpu().numpy()[:, :, [2, 1, 0]]
            #real_att_ = cv2.resize(attention[1].view(14, 14).data.cpu().numpy(),
            #                                                        tmp_real.shape[:2][::-1])

            for idx in range(len(target_data)):
                #tmp_target = target_img[idx].permute([1, 2, 0]).data.cpu().numpy()
                tmp_fake = torch.clamp(img + epsilons[idx], 0, 1).permute([1, 2, 0]).data.cpu().numpy()[:, :, [2, 1, 0]]
                noise = torch.abs(epsilons[idx]).permute([1, 2, 0]).data.cpu().numpy()
                #fake_att_ = cv2.resize(fake_attention[1].view(14, 14).data.cpu().numpy(),
                #                                                    tmp_real.shape[:2][::-1])

                cv2.imwrite('images/real_{}.png'.format(i), (255 * tmp_real).astype(int))
                cv2.imwrite('images/fake_{}.png'.format(i), (255 * tmp_fake).astype(int))

                max_value, min_value = np.max(noise), np.min(noise)
                noise = (noise - min_value) / (max_value - min_value)
                plt.imsave('images/noise_{}.png'.format(i), noise, format='png')

                for ix in range(len(fake_attention)):
                    real_att_ = cv2.resize(fake_attention[ix].view(14, 14).data.cpu().numpy(),
                                                                            tmp_real.shape[:2][::-1])
                    fig = plt.figure()
                    plt.imshow(tmp_real)
                    plt.imshow(255 * real_att_, alpha=0.85, cmap='viridis')
                    plt.axis('off')
                    plt.savefig('images/img_{}_attention_{}.png'.format(i + 1, ix),
                                                  format='png', bbox_inches='tight', pad_inches=0)

opt = opts.parse_opt()
attack(opt)
