import torch
import torch.nn as nn
import torch.nn.functional as F

from models.OldModel import ShowAttendTellModel

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)
        
        return fc, att

class combineNet(nn.Module):
    """
    Combine the feature extracted network and rnn network together.
    """
    def __init__(self, resnet, rnn_net):
        super(combineNet, self).__init__()
        self.resnet = resnet
        self.rnn_net = rnn_net
        self.mean = mean.unsqueeze(1).unsqueeze(2).cuda()
        self.std = std.unsqueeze(1).unsqueeze(2).cuda()

    def forward(self, img, eps, att_size=14, qs_list=None, qs_inds=None, t=None, 
                    s_observed=None, s_hidden=None, opt={}, slack=1., mode='sample'):
        # normalize
        img = img + eps
        img = torch.clamp(img, 0, 1)
        img = (img - self.mean) / self.std
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze().unsqueeze(0)
        att = F.adaptive_avg_pool2d(x, (att_size,att_size)).squeeze().permute(1, 2, 0).unsqueeze(0)
       
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
        else:
            raise Exception("Invaild mode {}.".format(mode))




        
