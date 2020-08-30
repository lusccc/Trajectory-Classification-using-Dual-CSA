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
    def __init__(self, n_channels, RP_emb_dim, FS_emb_dim, pretrained=False):
        super(Dual_CA_Softmax, self).__init__()
        self.RP_AE = Conv2D_AE(n_channels, RP_emb_dim)
        self.FS_AE = Conv1D_AE(n_channels, FS_emb_dim)
        # note: softmax is contained in n.CrossEntropyLoss(), here we dont need to do this
        # self.softmax = nn.Softmax()
        self.pretrained = pretrained

    def set_pretrained(self, petrained):
        self.pretrained = petrained

    def forward(self, RP, FS):
        RP_recon, RP_emb = self.RP_AE(RP)
        FS_recon, FS_emb = self.FS_AE(FS)
        concat_emb = torch.cat((RP_emb, FS_emb), dim=1) if self.pretrained else None
        return {'recon_ori': [(RP_recon, RP), (FS_recon, FS)], 'pred': concat_emb, 'emb': concat_emb}


if __name__ == '__main__':
    model = Dual_CA_Softmax(5, 2, 2)
    print(model)
    summary(model, [(5, 184, 184), (5, 200)])
