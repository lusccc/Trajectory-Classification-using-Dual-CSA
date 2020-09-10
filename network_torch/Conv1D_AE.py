
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Conv1D_AE(nn.Module):
    # refer to https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv.ipynb
    def __init__(self, n_ori_channels, emb_dim):
        super(Conv1D_AE, self).__init__()
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        ### ENCODER
        # 200xn_ori_channels => 200x32
        self.conv_1 = nn.Conv1d(in_channels=n_ori_channels,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                # (1*(200-1) - 200 + 3) / 2 = 1
                                padding=1)
        # 200x32 => 100x32
        self.pool_1 = nn.MaxPool1d(kernel_size=2,
                                   stride=2,
                                   # (2*(100-1) - 200 + 2) / 2 = 0
                                   padding=0,
                                   return_indices=True)
        # 100x100x32 => 100x100x64
        self.conv_2 = nn.Conv1d(in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                # (1*(100-1) - 100 + 3) / 2 = 1
                                padding=1)
        # 100x64 => 50x64
        self.pool_2 = nn.MaxPool1d(kernel_size=2,
                                   stride=2,
                                   # (2*(50-1) - 100 + 2) / 2 = 0
                                   padding=0,
                                   return_indices=True)
        # 50x64 => 50x128
        self.conv_3 = nn.Conv1d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                # (1*(50-1) - 50 + 3) / 2 = 1
                                padding=1)
        # 50x128 => 25x128
        self.pool_3 = nn.MaxPool1d(kernel_size=2,
                                   stride=2,
                                   # (2*(25-1) - 50 + 2) / 2 = 0
                                   padding=0,
                                   return_indices=True)
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(128 * 25, emb_dim)

        ### DECODER
        self.recon_flatten = nn.Linear(emb_dim, 128 * 25)  # reconstruct flatten in encoder
        self.unpool_1 = nn.MaxUnpool1d(kernel_size=2,
                                       stride=2,
                                       padding=0)
        self.deconv_1 = nn.ConvTranspose1d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.unpool_2 = nn.MaxUnpool1d(kernel_size=2,
                                       stride=2,
                                       padding=0)
        self.deconv_2 = nn.ConvTranspose1d(in_channels=64,
                                           out_channels=32,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.unpool_3 = nn.MaxUnpool1d(kernel_size=2,
                                       stride=2,
                                       padding=0)
        self.deconv_3 = nn.ConvTranspose1d(in_channels=32,
                                           out_channels=n_ori_channels,
                                           kernel_size=3,
                                           stride=1,
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
        x = x.view(-1, 128, 25) # reshape
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
    model = Conv1D_AE(1, 152)
    print(model)
    summary(model, (1, 200))