import math
import threading
import timeit

import numpy as np
import tables as tb
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from logzero import logger

import utils
from MF_RP_mat_h5support import H5_NODE_NAME


class Trajectory_Feature_Dataset(Dataset):
    def __init__(self, dataset_name, data_type, ):
        self.dataset_name = dataset_name
        self.data_type = data_type
        # self.multi_channel_RP_mats = utils.synchronized_open_file(lock,
        #                                                           f'./data/{self.dataset_name}_features/multi_channel_RP_mats_{data_type}.h5',
        #                                                           mode='r').get_node('/' + H5_NODE_NAME)
        # self.multi_channel_RP_mats = tb.open_file(
        #     f'./data/{self.dataset_name}_features/multi_channel_RP_mats_{data_type}.h5',
        #     mode='r').get_node('/' + H5_NODE_NAME)
        self.multi_feature_segs = torch.from_numpy(
            np.load(f'./data/{self.dataset_name}_features/multi_feature_segs_{data_type}_normalized.npy'))
        self.labels = torch.from_numpy(
            np.load(f'./data/{self.dataset_name}_features/multi_feature_seg_labels_{data_type}.npy'))

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
        RP = self.transform(torch.from_numpy(multi_channel_RP_mats[index]))
        label = self.labels[index]
        FS = self.multi_feature_segs[index]
        multi_channel_RP_mats_h5_file.close()
        return RP, FS, label


def calc_RP_data_mean_std(dataset_name, data_type, use_gpu=False):
    class RP_Data(Dataset):
        def __init__(self, dataset_name, data_type):
            self.dataset_name = dataset_name
            self.multi_channel_RP_mats_h5_file = tb.open_file(
                f'./data/{self.dataset_name}_features/multi_channel_RP_mats_{data_type}.h5',
                mode='r')
            self.multi_channel_RP_mats = self.multi_channel_RP_mats_h5_file.get_node('/' + H5_NODE_NAME)
            # logger.info(self.multi_channel_RP_mats.shape)

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
        var += ((data.view(5, -1) - mean.unsqueeze(1)) ** 2).sum(dim=1)
        nb_samples += np.prod(data.size()[1:])  # width*height
    std = torch.sqrt(var / nb_samples)
    logger.info(f"Std:{std}")
    logger.info("calc_RP_data_mean_std elapsed time: %.3f seconds" % (timeit.time.perf_counter() - start))
    dataset.multi_channel_RP_mats_h5_file.close()
    return mean, std


if __name__ == '__main__':
    # data_set = Dataset('geolife')
    # x_RP_train = data_set.x_RP_train
    # x_RP_test = data_set.x_RP_test
    # x_features_series_train = data_set.x_features_series_train
    # x_features_series_test = data_set.x_features_series_test
    # x_centroids_train = data_set.x_centroids_train
    # x_centroids_test = data_set.x_centroids_test
    # y_train = data_set.y_train
    # y_test = data_set.y_test
    # RP_mats_h5file = synchronized_open_file('data\geolife_features\RP_mats_train.h5', mode='a')
    # multi_channel_RP_mats = RP_mats_h5file.get_node('/' + 'RP_data')
    # a = multi_channel_RP_mats[0]
    # calc_RP_data_mean_std('geolife', 'train', False)
    TFD = Trajectory_Feature_Dataset('geolife', 'train')
    a = TFD[0]
