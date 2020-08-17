import torch
import torch.nn as nn
from torchsummary import summary
# Clustering layer definition (see DCEC article for equations)
class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=5, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x1 = x.unsqueeze(1) - self.weight
        x2 = torch.mul(x1, x1)
        x3 = torch.sum(x2, dim=2)
        x4 = 1.0 + (x3 / self.alpha)
        x5 = 1.0 / x4
        x6 = x5 ** ((self.alpha +1.0) / 2.0)
        x7 = torch.t(x6) / torch.sum(x6, dim=1)
        x8 = torch.t(x7)
        return x8

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)

model = ClusterlingLayer()
print(model)
summary(model, (10,))