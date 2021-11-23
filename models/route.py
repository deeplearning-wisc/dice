import torch
import torch.nn as nn
import numpy as np


class RouteDICE(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=90, conv1x1=False, info=None):
        super(RouteDICE, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.topk = topk
        self.info = info[0]
        self.masked_w = None

    def calculate_mask_weight(self):
        self.thresh = np.percentile(self.info, self.topk)
        mask = torch.Tensor((self.info > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out

