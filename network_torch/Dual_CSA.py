import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger
from torch import device
from torchsummary import summary

from network_torch.Conv1D_AE import Conv1D_AE
from network_torch.Conv2D_AE import Conv2D_AE


class PCC_Layer(nn.Module):

    def __init__(self, centroid, alpha=1.):
        """
        :param centroid: tensor, predefined centroid, shape: (n_class, emb_dim)
        """
        super(PCC_Layer, self).__init__()
        self.alpha = alpha
        self.centroid = centroid

    def forward(self, emb):
        """
        see paper `Unsupervised Deep Embedding for Clustering Analysis`

        qij = 1/(1+dist(zi, uj)^2), then normalize it.

        qij: each row represent the probability of a sample belong to each class (centroid)

        Examples
        --------
         >>> z1=[1,2,3]  z2=[4,5,6]
         >>> u1=[1,1,1]  u2=[3,3,3]  u3=[6,6,6]
         we will calc q11, q12, q13 and q21, q22, q23 at a time.
         here we calc them all at once following tensor calc rules, the results will be:
         >>> qij=[[q11, q12, q13],[q21, q22, q23]]
         after normalization, we return results, i.e., the probability of assigning sample i to class j
        :param emb: (n, emb_dim)
        """
        z = emb.unsqueeze(1)
        u = self.centroid
        qij = (1. + torch.sum((z - u) ** 2, dim=2) / self.alpha) ** -1
        qij_normalize = qij.T / torch.sum(qij, dim=1)  # transpose op here for easy calc
        qij_normalize = qij_normalize.T  # transpose back
        return qij_normalize


class Dual_CSA(nn.Module):
    def __init__(self, n_channels, RP_emb_dim, FS_emb_dim, centroid, pretrained=False):
        super(Dual_CSA, self).__init__()
        self.RP_AE = Conv2D_AE(n_channels, RP_emb_dim)
        self.FS_AE = Conv1D_AE(n_channels, FS_emb_dim)
        self.centroid = centroid
        self.PCC = PCC_Layer(self.centroid, 1)
        self.pretrained = pretrained

    def set_pretrained(self, petrained):
        self.pretrained = petrained

    def cuda(self, device=None):
        self.centroid = self.centroid.cuda(device)
        self.PCC = PCC_Layer(self.centroid, 1)
        return super().cuda(device)

    def forward(self, RP, FS):
        RP_recon, RP_emb = self.RP_AE(RP)
        FS_recon, FS_emb = self.FS_AE(FS)
        concat_emb = torch.cat((RP_emb, FS_emb), dim=1)
        soft_label = self.PCC(concat_emb) if self.pretrained else None
        return {'recon_ori': [(RP_recon, RP), (FS_recon, FS)], 'pred': soft_label, 'emb': concat_emb}


if __name__ == '__main__':
    ces = torch.rand(5, 304)
    model = Dual_CSA(5, 152, 152, ces)
    model.pretrained = True
    print(model)
    summary(model, [(5, 184, 184), (5, 200)])
