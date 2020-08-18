import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from netwok_torch.Conv1D_AE import Conv1D_AE
from netwok_torch.Conv2D_AE import Conv2D_AE


class PCC_Layer(nn.Module):

    def __init__(self, centroids, alpha=1.):
        """
        :param centroids: tensor, predefined centroids, shape: (n_class, emb_dim)
        """
        super(PCC_Layer, self).__init__()
        self.alpha = alpha
        self.centroids = centroids

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
        u = self.centroids
        qij = (1. + torch.sum((z - u) ** 2, dim=2) / self.alpha) ** -1
        qij_normalize = qij.T / torch.sum(qij, dim=1)  # transpose op here for easy calc
        qij_normalize = qij_normalize.T  # transpose back
        return qij_normalize


class Dual_CSA(nn.Module):
    def __init__(self, n_channels, RP_emb_dim, FS_emb_dim, centroids):
        super(Dual_CSA, self).__init__()
        self.RP_AE = Conv2D_AE(n_channels, RP_emb_dim)
        self.FS_AE = Conv1D_AE(n_channels, FS_emb_dim)
        self.centroids = centroids
        self.PCC = PCC_Layer(self.centroids, 1)

    def forward(self, RP_mat, features_seg):
        RP_recon, RP_emb = self.RP_AE(RP_mat)
        FS_recon, FS_emb = self.FS_AE(features_seg)
        concat_emb = torch.cat((RP_emb, FS_emb), dim=1)
        soft_label = self.PCC(concat_emb)
        return RP_recon, soft_label, FS_recon

if __name__ == '__main__':
    ces = torch.tensor(
        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [4, 4, 4, 4]]
    )
    model = Dual_CSA(5, 2, 2, ces)
    print(model)
    summary(model, [(5, 184, 184), (5, 200)])

