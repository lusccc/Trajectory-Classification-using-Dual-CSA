import math
import os
import pathlib
import threading
import timeit

import logzero
import numpy as np
import tables as tb
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from logzero import logger

pathlib.Path(os.environ['RES_PATH']).mkdir(parents=True, exist_ok=True)
logzero.logfile(os.path.join(os.environ['RES_PATH'], 'log.txt'), backupCount=3)

from MF_RP_mat_h5support import H5_NODE_NAME


class Trajectory_Feature_Dataset(Dataset):
    def __init__(self, dataset_name, data_type):
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.multi_feature_segs = torch.from_numpy(
            np.load(f'./data/{self.dataset_name}_features/multi_feature_segs_{data_type}_normalized.npy')).float()
        self.labels = torch.from_numpy(
            np.load(f'./data/{self.dataset_name}_features/multi_feature_seg_labels_{data_type}.npy')).float()

        self.transform = transforms.Compose([
            transforms.Normalize(*calc_RP_data_mean_std(dataset_name, data_type))
        ])

    def __len__(self):
        return self.multi_feature_segs.shape[0]

    def __getitem__(self, index):
        # open h5 file at each iteration to avoid parallel reading error,
        # see https://discuss.pytorch.org/t/hdf5-multi-threaded-alternative/6189/9
        multi_channel_RP_mats_h5_file = tb.open_file(
            f'./data/{self.dataset_name}_features/multi_channel_RP_mats_{self.data_type}.h5',
            mode='r')
        multi_channel_RP_mats = multi_channel_RP_mats_h5_file.get_node('/' + H5_NODE_NAME)
        RP = self.transform(torch.from_numpy(multi_channel_RP_mats[index])).float()
        label = self.labels[index]
        FS = self.multi_feature_segs[index]
        multi_channel_RP_mats_h5_file.close()
        return RP, FS, label


def calc_RP_data_mean_std(dataset_name, data_type, use_gpu=False):
    # TODO: only use train set to calc mean std, though currently indeed use train set, we should make it clear in code
    if os.path.exists(f'./data/{dataset_name}_features/mean.npy') and os.path.exists(
            f'./data/{dataset_name}_features/std.npy'):
        mean, std = np.load(f'./data/{dataset_name}_features/mean.npy'), np.load(
            f'./data/{dataset_name}_features/std.npy')
        logger.info(f'loading calculated mean and std from file, mean: {mean}, std: {std}')
        return mean, std
    logger.info(f'calc_RP_data_mean_std for {dataset_name}...')

    class RP_Data(Dataset):
        def __init__(self, dataset_name, data_type):
            self.dataset_name = dataset_name
            self.multi_channel_RP_mats_h5_file = tb.open_file(
                f'./data/{self.dataset_name}_features/multi_channel_RP_mats_{data_type}.h5',
                mode='r')
            self.multi_channel_RP_mats = self.multi_channel_RP_mats_h5_file.get_node('/' + H5_NODE_NAME)
            logger.info(self.multi_channel_RP_mats.shape)
            self.n_channels = self.multi_channel_RP_mats.shape[1]

        def __len__(self):
            return self.multi_channel_RP_mats.shape[0]

        def __getitem__(self, index):
            RPs = torch.from_numpy(self.multi_channel_RP_mats[index])
            return RPs

    # https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/23
    device = torch.device('cuda' if use_gpu else 'cpu')
    dataset = RP_Data(dataset_name, data_type)

    start = timeit.time.perf_counter()
    # calc on ram:
    # data = dataset_name.RP_mats_ndarray.to(device)
    # logger.info("Mean:", torch.mean(data, dim=(0, 2, 3)))
    # logger.info("Std:", torch.std(data, dim=(0, 2, 3)))
    # logger.info("Elapsed time: %.3f seconds" % (timeit.time.perf_counter() - start))
    # logger.info()

    start = timeit.time.perf_counter()
    mean = 0.
    for data in dataset:
        data = data.to(device)
        mean += torch.mean(data, dim=(1, 2))
    mean /= len(dataset)
    logger.info(f"Mean:{mean}")

    var = 0.  # variance
    nb_samples = 0.
    for data in dataset:
        data = data.to(device)
        var += ((data.view(dataset.n_channels, -1) - mean.unsqueeze(1)) ** 2).sum(dim=1)
        nb_samples += np.prod(data.size()[1:])  # width*height
    std = torch.sqrt(var / nb_samples)
    logger.info(f"Std:{std}")
    logger.info("calc_RP_data_mean_std elapsed time: %.3f seconds" % (timeit.time.perf_counter() - start))
    dataset.multi_channel_RP_mats_h5_file.close()
    logger.info('saving calculated mean and std')
    np.save(f'./data/{dataset_name}_features/mean.npy', mean.numpy())
    np.save(f'./data/{dataset_name}_features/std.npy', std.numpy())
    return mean, std


if __name__ == '__main__':
    TFD = Trajectory_Feature_Dataset('SHL', 'train')
    a = TFD[0]
