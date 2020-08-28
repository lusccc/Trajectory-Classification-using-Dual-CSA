from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger
from torch import device
from torchsummary import summary

from netwok_torch.Conv1D_AE import Conv1D_AE
from netwok_torch.Conv2D_AE import Conv2D_AE


class Dual_CA_Softmax(nn.Module):
    def __init__(self, n_channels, RP_emb_dim, FS_emb_dim, pretrain=True):
        super(Dual_CA_Softmax, self).__init__()
        self.RP_AE = Conv2D_AE(n_channels, RP_emb_dim)
        self.FS_AE = Conv1D_AE(n_channels, FS_emb_dim)
        self.softmax = nn.Softmax()
        self.pretrain = pretrain


    def forward(self, RP_mat, FS):
        RP_recon, RP_emb = self.RP_AE(RP_mat)
        FS_recon, FS_emb = self.FS_AE(FS)
        if not self.pretrain:
            concat_emb = torch.cat((RP_emb, FS_emb), dim=1)
            soft_label = self.softmax(concat_emb)
        else:
            soft_label = None
        return RP_recon, soft_label, FS_recon


if __name__ == '__main__':
    model = Dual_CA_Softmax(5, 2, 2)
    print(model)
    summary(model, [(5, 184, 184), (5, 200)])
