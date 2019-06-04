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

class ShowTellModel(CaptionModel):
    def __init__(self, opt):
        super(ShowTellModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i-1].data.clone()
                        #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                        #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                        prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = seq[:, i-1].clone()                
                # break if all the sequences end
                if i >= 2 and seq[:, i-1].data.sum() == 0:
                    break
                xt = self.embed(it)

            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        # 'it' contains a word index
        xt = self.embed(it)
                
        output, state = self.core(xt.unsqueeze(0), state)
        logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k+1]).expand(beam_size, self.input_encoding_size)
                elif t == 1: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(it)

                output, state = self.core(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)

            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
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
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)

            output, state = self.core(xt.unsqueeze(0), state)
            logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing
            
            #if t >= 0:
            if t == 0:
                seq[:,t] = it #seq[t] the input of t+2 time step
                seqLogprobs[:,t] = sampleLogprobs.view(-1)
                continue
            # stop when all finished
            if t == 1:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it #seq[t] the input of t+2 time step
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs, None

    
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
        batch_size = fc_feats.size(0)

        xt = self.img_embed(fc_feats)
        state = self.init_hidden(batch_size)
        output, state = self.core(xt.unsqueeze(0), state)
        it = fc_feats.data.new(batch_size).long().zero_()

        for t in range(max_step):
            xt = self.embed(it.view(-1).long())
            output, state = self.core(xt.unsqueeze(0), state)
            logprob = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
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
        batch_size = fc_feats.size(0)

        xt = self.img_embed(fc_feats)
        state = self.init_hidden(batch_size)
        output, state = self.core(xt.unsqueeze(0), state)
        it = fc_feats.data.new(batch_size).long().zero_()

        for t in range(max_step):
            xt = self.embed(it.view(-1).long())
            output, state = self.core(xt.unsqueeze(0), state)
            logprob = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
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

        batch_size = fc_feats.size(0)
        xt = self.img_embed(fc_feats)
        state = self.init_hidden(batch_size)
        output, state = self.core(xt.unsqueeze(0), state)
        it = fc_feats.data.new(batch_size).long().zero_()
        loss = 0.

        for t in range(max_step):
            xt = self.embed(it.view(-1))
            output, state = self.core(xt.unsqueeze(0), state)
            logprob = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
            if t in obs_idx:
                it = s_observed[t]
            else:
                it = s_hidden[t]
            loss += logprob[:, it].clone()

        return loss
    

    def _inference_acl(self, fc_feats, att_feats, s_observed, s_hidden, slack=10000.):
        obs_idx = torch.nonzero(s_observed).squeeze(1)
        max_idx = torch.max(obs_idx)
        # if last obs_idx is end of caption, need add a <eos>
        if max_idx != len(s_observed) - 1 and s_observed[max_idx + 1] == 0:
            tmp = s_observed.clone()
            tmp[max_idx + 1] = -1
            obs_idx = torch.nonzero(tmp).squeeze(1)
            max_idx = torch.max(obs_idx)
        max_step = max_idx + 1

        batch_size = fc_feats.size(0)
        xt = self.img_embed(fc_feats)
        state = self.init_hidden(batch_size)
        output, state = self.core(xt.unsqueeze(0), state)
        it = fc_feats.data.new(batch_size).long().zero_()
        logits = []
        cap_logits = []
        loss = 0.

        for t in range(max_step):
            xt = self.embed(it.view(-1))
            output, state = self.core(xt.unsqueeze(0), state)
            logit_tmp = self.logit(output.squeeze(0))
            if t in obs_idx:
                it = s_observed[t]
            else:
                it = s_hidden[t]
            cap_logits.append(logit_tmp[:, it])
            logit_tmp[:, it] -= slack
            max_prob, _ = torch.max(logit_tmp, 1)
            max_prob = torch.max(max_prob, torch.tensor(-1.).cuda())
            logits.append(max_prob)

        logits = torch.cat(logits, 0)
        cap_logits = torch.cat(cap_logits, 0)

        loss = torch.sum(logits - cap_logits)

        return loss

    def _inference_acl_logprob(self, fc_feats, att_feats, s_observed, s_hidden):
        obs_idx = torch.nonzero(s_observed).squeeze(1)
        max_idx = torch.max(obs_idx)
        # if last obs_idx is end of caption, need add a <eos>
        if max_idx != len(s_observed) - 1 and s_observed[max_idx + 1] == 0:
            tmp = s_observed.clone()
            tmp[max_idx + 1] = -1
            obs_idx = torch.nonzero(tmp).squeeze(1)
            max_idx = torch.max(obs_idx)
        max_step = max_idx + 1

        batch_size = fc_feats.size(0)
        xt = self.img_embed(fc_feats)
        state = self.init_hidden(batch_size)
        output, state = self.core(xt.unsqueeze(0), state)
        it = fc_feats.data.new(batch_size).long().zero_()
        loss = 0.

        for t in range(max_step):
            xt = self.embed(it.view(-1))
            output, state = self.core(xt.unsqueeze(0), state)
            logit_tmp = self.logit(output.squeeze(0))
            p_logit = torch.softmax(logit_tmp, dim=1)
            loss += p_logit[:, s_observed[t]].clone()
        
        #print(loss)
        return loss

    def _inference_e(self, fc_feats, att_feats, qs_topk, qs_inds):
        batch_size = fc_feats.size(0)
        xt = self.img_embed(fc_feats)
        state = self.init_hidden(batch_size)
        output, state = self.core(xt.unsqueeze(0), state)

        it = fc_feats.data.new(batch_size).long().zero_()
        xt = self.embed(it.view(-1))
        output, state = self.core(xt.unsqueeze(0), state)

        if len(qs_topk) == 0:
            logprob = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
            #qs = torch.softmax(torch.exp(logprob - 1), dim=1)
            qs = F.softmax(torch.exp(logprob - 1), dim=1)
        else:
            combinations = itertools.product(*qs_inds)
            qs_prob_comb = list(itertools.product(*qs_topk))
            qs = torch.zeros(1, self.vocab_size + 1).cuda()

            for i, comb in enumerate(combinations):
                for t in range(len(qs_inds)):
                    it = comb[t]
                    xt = self.embed(it.view(-1))
                    output, state = self.core(xt.unsqueeze(0), state)

                logprob = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
                qs_mult = reduce(lambda x, y: x * y, qs_prob_comb[i])
                qs = qs + float(qs_mult) * logprob
            #qs = torch.softmax(torch.exp(qs - 1), dim=1)
            qs = F.softmax(torch.exp(qs - 1), dim=1)

        return qs


    def _inference_m(self, fc_feats, att_feats, qs_topk, qs_inds, t_step):
        batch_size = fc_feats.size(0)
        xt = self.img_embed(fc_feats)
        state = self.init_hidden(batch_size)
        output, state = self.core(xt.unsqueeze(0), state)

        it = fc_feats.data.new(batch_size).long().zero_()
        xt = self.embed(it.view(-1))
        output, state_init = self.core(xt.unsqueeze(0), state)
        loss = 0.

        if t_step == 0:
            tmp = torch.tensor(qs_topk[t_step]).cuda()
            logprob = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
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
                    output, state = self.core(xt.unsqueeze(0), state)
                logprob = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
                logprob_topk = logprob[0, comb[t_step]].clone()
                qs_mult = reduce(lambda x, y: x * y, qs_prob_comb[i])
                loss += float(qs_mult) * logprob_topk

        return loss


















