import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger
from torch import device
from torchsummary import summary

from network_torch.Conv1D_AE import Conv1D_AE
from network_torch.Conv2D_AE import Conv2D_AE
from network_torch.Dual_CSA import PCC_Layer


class CSA_RP(nn.Module):
    def __init__(self, n_channels, RP_emb_dim, centroid, pretrained=False):
        super(CSA_RP, self).__init__()
        self.RP_AE = Conv2D_AE(n_channels, RP_emb_dim)
        self.centroid = centroid
        self.PCC = PCC_Layer(self.centroid, 1)
        self.pretrained = pretrained

    def set_pretrained(self, petrained):
        self.pretrained = petrained

    def cuda(self, device=None):
        self.centroid = self.centroid.cuda(device)
        self.PCC = PCC_Layer(self.centroid, 1)
        return super().cuda(device)

    def forward(self, RP):
        RP_recon, RP_emb = self.RP_AE(RP)
        soft_label = self.PCC(RP_emb) if self.pretrained else None
        return {'recon_ori': [(RP_recon, RP), ], 'pred': soft_label, 'emb': RP_emb}


if __name__ == '__main__':
    ces = torch.rand(5, 152)
    model = CSA_RP(5, 152, ces)
    model.set_pretrained(True)
    print(model)
    summary(model, (5, 184, 184))
