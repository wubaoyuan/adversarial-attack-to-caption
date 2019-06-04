import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

class myInception(nn.Module):
    def __init__(self, net):
        super(myInception, self).__init__()
        self.inception = net

    def forward(self, img):
        x = img.unsqueeze(0)
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)

        fc = x.mean(3).mean(2).squeeze().unsqueeze(0)
        att = x.squeeze().permute(1, 2, 0).unsqueeze(0)

        return fc, att

class combineNet(nn.Module):
    """
    Combine the feature extracted network and rnn network together.
    """
    def __init__(self, inception, rnn_net):
        super(combineNet, self).__init__()
        self.inception = inception
        self.rnn_net = rnn_net
        self.mean = mean.unsqueeze(1).unsqueeze(2).cuda()
        self.std = std.unsqueeze(1).unsqueeze(2).cuda()

    def forward(self, img, eps, att_size=8, qs_list=None, qs_inds=None, t=None, 
                    s_observed=None, s_hidden=None, opt={}, slack=1., mode='sample'):
        # normalize
        img = img + eps
        img = torch.clamp(img, 0, 1)
        img = (img - self.mean) / self.std
        x = img.unsqueeze(0)
        
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        
        fc = x.mean(3).mean(2).squeeze().unsqueeze(0)
        att = x.squeeze().permute(1, 2, 0).unsqueeze(0)

        if mode == 'sample': 
            seq, logprobs, attentions = self.rnn_net(fc, att, opt=opt, mode=mode)
            return seq, logprobs, attentions
        elif mode == 'inference_e':
            qs = self.rnn_net(fc, att, qs_topk=qs_list, qs_inds=qs_inds, mode=mode)
            return qs
        elif mode == 'inference_m':
            loss = self.rnn_net(fc, att, qs_topk=qs_list, qs_inds=qs_inds, t_step=t, mode=mode)
            return loss
        elif mode == 'inference_lvc':
            s_hidden = self.rnn_net(fc, att, s_observed=s_observed, mode=mode)
            return s_hidden
        elif mode == 'inference_lai':
            s_hidden, s_observed = self.rnn_net(fc, att, s_observed=s_observed, slack=slack, mode=mode)
            return s_hidden, s_observed
        elif mode == 'inference_gd':
            loss = self.rnn_net(fc, att, s_observed=s_observed, s_hidden=s_hidden, mode=mode)
            return loss
        elif mode == 'inference_acl':
            loss = self.rnn_net(fc, att, s_observed=s_observed, s_hidden=s_hidden, slack=slack, mode=mode)
            return loss
        elif mode == 'inference_acl_logprob':
            loss = self.rnn_net(fc, att, s_observed=s_observed, s_hidden=s_hidden, mode=mode)
            return loss
        else:
            raise Exception("Invaild mode {}.".format(mode))




        
