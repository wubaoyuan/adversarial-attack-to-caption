# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import itertools
from functools import reduce

from .CaptionModel import CaptionModel

class OldModel(CaptionModel):
    def __init__(self, opt):
        super(OldModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.linear = nn.Linear(self.fc_feat_size, self.num_layers * self.rnn_size) # feature to rnn_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
        if self.rnn_type == 'lstm':
            return (image_map, image_map)
        else:
            return image_map

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)
        outputs = []

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            xt = self.embed(it)

            output, state, _ = self.core(xt, fc_feats, att_feats, state)
            output = F.log_softmax(self.logit(self.dropout(output)), dim=1)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state, _ = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)), dim=1)

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, self.fc_feat_size)
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            
            state = self.init_hidden(tmp_fc_feats)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
            done_beams = []
            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(it)

                output, state, _ = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output)), dim=1)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)

        seq = []
        #seqLogprobs = []
        logprobs = []
        attentions = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
		seq.append(it)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprob.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprob.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprob by temperature
                    prob_prev = torch.exp(torch.div(logprob.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprob.gather(1, it) # gather the logprob at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            xt = self.embed(it)

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                #if unfinished.sum() == 0:
                #    break
                it = it * unfinished.type_as(it) # batchsize
                seq.append(it) #seq[t] the input of t+2 time step 
                #seqLogprobs.append(sampleLogprobs.view(-1)) # batchsize

            output, state, attention = self.core(xt, fc_feats, att_feats, state) # batchsize * hiddensize
            logprob = F.log_softmax(self.logit(self.dropout(output)), dim=1) # batchsize * vocab_size
            logprobs.append(logprob)
            attentions.append(attention)

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in logprobs], 1), attentions

    # lvc means latent variable completion
    def _inference_lvc(self, fc_feats, att_feats, s_observed):
        obs_idx = torch.nonzero(s_observed).squeeze(1)
        max_idx = torch.max(obs_idx)
        # if last obs_idx is end of caption, need add a <eos>
        max_idx = torch.max(obs_idx)
        if max_idx != len(s_observed) - 1 and s_observed[max_idx + 1] == 0:
            tmp = s_observed.clone()
            tmp[max_idx + 1] = -1
            obs_idx = torch.nonzero(tmp).squeeze(1)
            max_idx = torch.max(obs_idx)
        max_step = max_idx + 1

        s_hidden = torch.zeros_like(s_observed)
        state = self.init_hidden(fc_feats)
        batch_size = fc_feats.size(0)
        it = fc_feats.data.new(batch_size).long().zero_()

        for t in range(max_step):
            xt = self.embed(it.view(-1).long())
            output, state, _ = self.core(xt, fc_feats, att_feats, state)
            logprob = F.log_softmax(self.logit(self.dropout(output)), dim=1)
            if t in obs_idx:
                it = s_observed[t]
            else:
                _, it = torch.max(logprob, 1)
                s_hidden[t] = it

        return s_hidden

    def _inference_lai(self, fc_feats, att_feats, s_observed, slack=1.):
        obs_idx = torch.nonzero(s_observed).squeeze(1)
        max_idx = torch.max(obs_idx)
        # if last obs_idx is end of caption, need add a <eos>
        if max_idx != len(s_observed) - 1 and s_observed[max_idx + 1] == 0:
            tmp = s_observed.clone()
            tmp[max_idx + 1] = -1
            obs_idx = torch.nonzero(tmp).squeeze(1)
            max_idx = torch.max(obs_idx)
        max_step = max_idx + 1

        s_hidden_ = torch.zeros_like(s_observed)
        s_observed_ = torch.zeros_like(s_observed)

        state = self.init_hidden(fc_feats)
        batch_size = fc_feats.size(0)
        it = fc_feats.data.new(batch_size).long().zero_()

        for t in range(max_step):
            xt = self.embed(it.view(-1).long())
            output, state, _ = self.core(xt, fc_feats, att_feats, state)
            logprob = F.log_softmax(self.logit(self.dropout(output)), dim=1)
            _, it = torch.max(logprob, 1)
            if t in obs_idx:
                logprob += slack
                logprob[:, s_observed[t]] -= slack
                _, it = torch.max(logprob, 1)
                s_observed_[t] = it
            else:
                s_hidden_[t] = it

        return s_hidden_, s_observed_

    def _inference_gd(self, fc_feats, att_feats, s_observed, s_hidden):
        obs_idx = torch.nonzero(s_observed).squeeze(1)
        max_idx = torch.max(obs_idx)
        # if last obs_idx is end of caption, need add a <eos>
        if max_idx != len(s_observed) - 1 and s_observed[max_idx + 1] == 0:
            tmp = s_observed.clone()
            tmp[max_idx + 1] = -1
            obs_idx = torch.nonzero(tmp).squeeze(1)
            max_idx = torch.max(obs_idx)
        max_step = max_idx + 1

        state = self.init_hidden(fc_feats)
        batch_size = fc_feats.size(0)
        it = fc_feats.data.new(batch_size).long().zero_()
        loss = 0.

        for t in range(max_step):
            xt = self.embed(it.view(-1))
            output, state, _ = self.core(xt, fc_feats, att_feats, state)
            logprob = F.log_softmax(self.logit(self.dropout(output)), dim=1)
            if t in obs_idx:
                it = s_observed[t]
            else:
                it = s_hidden[t]
            loss += logprob[:, it].clone()

        return loss

    def _inference_e(self, fc_feats, att_feats, qs_topk, qs_inds):
        state = self.init_hidden(fc_feats)
        batch_size = fc_feats.size(0)

        it = fc_feats.data.new(batch_size).long().zero_()
        xt = self.embed(it.view(-1))
        output, state, _ = self.core(xt, fc_feats, att_feats, state)
	
        if len(qs_topk) == 0:
            logprob = F.log_softmax(self.logit(self.dropout(output)), dim=1)
            qs = F.softmax(torch.exp(logprob - 1), dim=1)
            #nn_softmax = torch.nn.Softmax()
            #qs = nn_softmax(torch.exp(logprob - 1), dim=1)
        else:
            combinations = itertools.product(*qs_inds)
            qs_prob_comb = list(itertools.product(*qs_topk))
            qs = torch.zeros(1, self.vocab_size + 1).cuda()

            for i, comb in enumerate(combinations):
                for t in range(len(qs_inds)):
                    it = comb[t]
                    xt = self.embed(it.view(-1))
                    output, state, _ = self.core(xt, fc_feats, att_feats, state)

                logprob = F.log_softmax(self.logit(self.dropout(output)), dim=1)
                qs_mult = reduce(lambda x, y: x * y, qs_prob_comb[i])
                qs = qs + float(qs_mult) * logprob
            qs = F.softmax(torch.exp(logprob - 1), dim=1)
            #qs = torch.softmax(torch.exp(qs - 1), dim=1)

        return qs


    def _inference_m(self, fc_feats, att_feats, qs_topk, qs_inds, t_step):
        state = self.init_hidden(fc_feats)
        batch_size = fc_feats.size(0)
	
        it = fc_feats.data.new(batch_size).long().zero_()
        xt = self.embed(it.view(-1))
        output, state_init, _ = self.core(xt, fc_feats, att_feats, state)
        loss = 0.

        if t_step == 0:
            tmp = torch.tensor(qs_topk[t_step]).cuda()
            logprob = F.log_softmax(self.logit(self.dropout(output)), dim=1)
            logprob_topk = logprob[0, qs_inds[t_step]].clone()
            loss = torch.sum(logprob_topk * tmp)
        else:
            combinations = itertools.product(*qs_inds[:t_step + 1])
            qs_prob_comb = list(itertools.product(*qs_topk[:t_step + 1]))

            for i, comb in enumerate(combinations):
                state = state_init
                for t in range(t_step):
                    it = comb[t]
                    xt = self.embed(it.view(-1))
                    output, state, _ = self.core(xt, fc_feats, att_feats, state)
                logprob = F.log_softmax(self.logit(self.dropout(output)), dim=1)
                logprob_topk = logprob[0, comb[t_step]].clone()
                qs_mult = reduce(lambda x, y: x * y, qs_prob_comb[i])
                loss += float(qs_mult) * logprob_topk

        return loss


class ShowAttendTellCore(nn.Module):
    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size, 
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1)
        else:
            self.ctx2att = nn.Linear(self.att_feat_size, 1)
            self.h2att = nn.Linear(self.rnn_size, 1)

    def forward(self, xt, fc_feats, att_feats, state):
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = att_feats.view(-1, self.att_feat_size)
        if self.att_hid_size > 0:
            att = self.ctx2att(att)                             # (batch * att_size) * att_hid_size
            att = att.view(-1, att_size, self.att_hid_size)     # batch * att_size * att_hid_size
            att_h = self.h2att(state[0][-1])                    # batch * att_hid_size
            att_h = att_h.unsqueeze(1).expand_as(att)           # batch * att_size * att_hid_size
            dot = att + att_h                                   # batch * att_size * att_hid_size
            dot = torch.tanh(dot)                               # batch * att_size * att_hid_size
            dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
            dot = self.alpha_net(dot)                           # (batch * att_size) * 1
            dot = dot.view(-1, att_size)                        # batch * att_size
        else:
            att = self.ctx2att(att)(att)                        # (batch * att_size) * 1
            att = att.view(-1, att_size)                        # batch * att_size
            att_h = self.h2att(state[0][-1])                    # batch * 1
            att_h = att_h.expand_as(att)                        # batch * att_size
            dot = att_h + att                                   # batch * att_size

	weight = F.softmax(dot, dim=1)
        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        output, state = self.rnn(torch.cat([xt, att_res], 1).unsqueeze(0), state)
        return output.squeeze(0), state, weight

class AllImgCore(nn.Module):
    def __init__(self, opt):
        super(AllImgCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.fc_feat_size, 
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

    def forward(self, xt, fc_feats, att_feats, state):
        output, state = self.rnn(torch.cat([xt, fc_feats], 1).unsqueeze(0), state)
        return output.squeeze(0), state

class ShowAttendTellModel(OldModel):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)

class AllImgModel(OldModel):
    def __init__(self, opt):
        super(AllImgModel, self).__init__(opt)
        self.core = AllImgCore(opt)

