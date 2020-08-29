import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger
from torch import device
from torchsummary import summary

from netwok_torch.Conv1D_AE import Conv1D_AE
from netwok_torch.Conv2D_AE import Conv2D_AE
from netwok_torch.Dual_CSA import PCC_Layer


class CSA_FS(nn.Module):
    def __init__(self, n_channels, FS_emb_dim, centroid, pretrained=False):
        super(CSA_FS, self).__init__()
        self.FS_AE = Conv2D_AE(n_channels, FS_emb_dim)
        self.centroid = centroid
        self.PCC = PCC_Layer(self.centroid, 1)
        self.pretrained = pretrained

    def set_pretrained(self, petrained):
        self.pretrained = petrained

    def cuda(self, device=None):
        self.centroid = self.centroid.cuda(device)
        self.PCC = PCC_Layer(self.centroid, 1)
        return super().cuda(device)

    def forward(self, FS):
        FS_recon, FS_emb = self.FS_AE(FS)
        soft_label = self.PCC(FS_emb) if self.pretrained else None
        return {'recon_ori': [(FS_recon, FS), ], 'pred': soft_label, 'emb': {FS_emb}}


if __name__ == '__main__':
    ces = torch.tensor(
        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [4, 4, 4, 4]]
    )
    model = CSA_FS(5, 4, ces)
    model.pretrain = False
    print(model)
    summary(model, (5, 184, 184))
