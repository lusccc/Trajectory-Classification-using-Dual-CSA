
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Conv2D_AE(nn.Module):
    # refer to https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv.ipynb
    def __init__(self, n_ori_channels, emb_dim):
        super(Conv2D_AE, self).__init__()
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        ### ENCODER
        # 184x184xn_ori_channels => 184x184x32
        self.conv_1 = nn.Conv2d(in_channels=n_ori_channels,
                                out_channels=32,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                # (1*(184-1) - 184 + 3) / 2 = 1
                                padding=1)
        # 184x184x32 => 92x92x32
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2),
                                   stride=(2, 2),
                                   # (2*(92-1) - 184 + 2) / 2 = 0
                                   padding=0,
                                   return_indices=True)
        # 92x92x32 => 92x92x64
        self.conv_2 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                # (1*(92-1) - 92 + 3) / 2 = 1
                                padding=1)
        # 92x92x64 => 46x46x64
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2),
                                   stride=(2, 2),
                                   # (2*(46-1) - 92 + 2) / 2 = 0
                                   padding=0,
                                   return_indices=True)
        # 46x46x64 => 46x46x128
        self.conv_3 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                # (1*(46-1) - 46 + 3) / 2 = 1
                                padding=1)
        # 46x46x128 => 23x23x128
        self.pool_3 = nn.MaxPool2d(kernel_size=(2, 2),
                                   stride=(2, 2),
                                   # (2*(23-1) - 46 + 2) / 2 = 0
                                   padding=0,
                                   return_indices=True)
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(128 * 23 * 23, emb_dim)

        ### DECODER
        self.recon_flatten = nn.Linear(emb_dim, 128 * 23 * 23)  # reconstruct flatten in encoder
        self.unpool_1 = nn.MaxUnpool2d(kernel_size=(2, 2),
                                       stride=(2, 2),
                                       padding=0)
        self.deconv_1 = nn.ConvTranspose2d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=(3, 3),
                                           stride=(1, 1),
                                           padding=1)
        self.unpool_2 = nn.MaxUnpool2d(kernel_size=(2, 2),
                                       stride=(2, 2),
                                       padding=0)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=64,
                                           out_channels=32,
                                           kernel_size=(3, 3),
                                           stride=(1, 1),
                                           padding=1)
        self.unpool_3 = nn.MaxUnpool2d(kernel_size=(2, 2),
                                       stride=(2, 2),
                                       padding=0)
        self.deconv_3 = nn.ConvTranspose2d(in_channels=32,
                                           out_channels=n_ori_channels,
                                           kernel_size=(3, 3),
                                           stride=(1, 1),
                                           padding=1)

    def forward(self, x):
        ### ENCODER
        x = self.conv_1(x)
        x = F.leaky_relu(x)
        x, pool_1_indices = self.pool_1(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        x, pool_2_indices = self.pool_2(x)
        x = self.conv_3(x)
        x = F.leaky_relu(x)
        x, pool_3_indices = self.pool_3(x)

        x = self.flatten(x)
        emb = self.embedding(x)

        # ### DECODER
        x = self.recon_flatten(emb)
        x = x.view(-1, 128, 23, 23) # reshape
        x = self.unpool_1(x, indices=pool_3_indices) # indices: max val index in feature map passed through
        x = self.deconv_1(x)
        x = F.leaky_relu(x)
        x = self.unpool_2(x, indices=pool_2_indices)
        x = self.deconv_2(x)
        x = F.leaky_relu(x)
        x = self.unpool_3(x, indices=pool_1_indices)
        x = self.deconv_3(x)
        x = F.leaky_relu(x)
        return x, emb




if __name__ == '__main__':
    model = Conv2D_AE(5, 152)
    print(model)
    summary(model, (5, 184, 184))