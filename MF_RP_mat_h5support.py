import argparse
import multiprocessing
import sys
import threading
import time
from math import cos, pi

import numpy as np
import tables as tb
from numpy.lib.stride_tricks import as_strided
from numpy.linalg import norm
from pyts.image.recurrence import _trajectories

from params import DIM, TAU, FEATURES_SET_1
from utils import scale_any_shape_data, scale_RP_each_feature, synchronized_open_file, synchronized_close_file

#  Settings for the embedding

# Distance metric in phase space ->
# Possible choices ("manhattan","euclidean","supremum")
METRIC = "euclidean"

EPS = 0.05  # Fixed recurrence threshold

lock = threading.Lock()
DATA_NAME = 'RP_data'


def __gen_multiple_RP_mats(multi_series, scale=False, dim=DIM, tau=TAU):
    """
    deprecated!
    generate a RP mat for each series, note: large memory required
    """

    n_series = len(multi_series)

    print('calc distance_mats ....')
    # each series generates n_vectors phase space trajectories
    stri = multi_series.strides
    its = multi_series.itemsize
    print(stri, its)
    phase_vectorss = _trajectories(multi_series, dimension=dim, time_delay=tau)
    n_vectors = phase_vectorss.shape[1]
    # calc distances mats, 1 mat for each series.
    # shape:(n_samples, n_vectors, n_vectors)
    # from source code in pyts.image.recurrence.RecurrencePlot.transform
    distance_mats = np.sqrt(
        np.sum((phase_vectorss[:, None, :, :] - phase_vectorss[:, :, None, :]) ** 2,
               axis=3, dtype='float32')
    )
    print('end calc distance_mats')

    mycode = False
    if mycode:
        ## my code to calc distances
        RP_mats = []  # !!skipped Heavside function
        count = 0
        for distance_mat, phase_vectors in zip(distance_mats, phase_vectorss):
            print('{}/{}'.format(count, n_series))
            # now no need to use this method since we want to calc sign value
            # https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
            # distances = pdist(phase_vectors, )
            sign_value_mat = np.empty((n_vectors, n_vectors))
            for i in range(n_vectors):
                for j in range(n_vectors):
                    a = phase_vectors[i]
                    b = phase_vectors[j]
                    if i == j:
                        sign_value = 1
                    else:
                        # equation (5) in "Robust Single Accelerometer-Based Activity Recognition
                        # Using Modified Recurrence Plot"
                        sign_value = sign(a, b)
                    sign_value_mat[i][j] = sign_value
            sign_value_mat = np.array(sign_value_mat)
            RP_mat = sign_value_mat * distance_mat
            RP_mats.append(RP_mat)
            count += 1

        RP_mats = np.array(RP_mats)
    else:
        RP_mats = distance_mats  # !!skipped Heavside function
    if scale:
        RP_mats = scale_any_shape_data(RP_mats)
    return RP_mats


def sign(m, n):
    # equation (5) in "Robust Single Accelerometer-Based Activity Recognition
    # Using Modified Recurrence Plot"
    m = np.array(m)
    n = np.array(n)
    dim = m.shape[0]  # vector dim
    v = np.repeat(1, dim)  # base vec
    cos_angle = np.dot(m - n, v) / (norm(m - n) * norm(v))
    if cos_angle < cos(3. / 4. * pi):
        return -1
    else:
        return 1





def gen_single_RP_mat(singel_series, dimension=DIM, time_delay=TAU):
    # The strides of an array tell us how many bytes we have to skip in memory to move to the next position along a certain axis.
    n_timestamps = len(singel_series)
    s = singel_series.strides[0]
    shape_new = (n_timestamps - (dimension - 1) * time_delay, dimension)
    strides_new = (s, time_delay * s)
    phase_space_trjs = as_strided(singel_series, shape_new, strides_new)
    distance_mat = np.sqrt(
        np.sum((phase_space_trjs[None, :, :] - phase_space_trjs[:, None, :]) ** 2,
               axis=2, dtype='float32')
    )
    return distance_mat


def generate_RP_mats(trjs_segs_features, dim, tau, n_features, save_path, h5node_name):
    print('gen_RP_mats...')
    tasks = []
    batch_size = int(len(trjs_segs_features) / n_cpus + 1)
    for i in range(0, n_cpus):
        tasks.append(pool.apply_async(do_generate_RP_mats,
                                      (trjs_segs_features[i * batch_size:(i + 1) * batch_size], dim, tau, n_features, save_path,
                                       h5node_name
                                       )))
    res = [t.get() for t in tasks]


def do_generate_RP_mats(multi_features_segs, dim, tau, n_features, save_path, h5node_name):
    '''
    :param multi_features_segs: (n, seg_size, n_features)
    '''
    RP_mats_h5file = synchronized_open_file(lock, save_path, mode='a')
    RP_mats_h5array = RP_mats_h5file.get_node('/' + h5node_name)
    for multi_features_seg in multi_features_segs:
        multi_channels_RP_mat = []
        for i in range(n_features):
            single_feature_seg = multi_features_seg[:, i]  # (n, seg_size)
            RP_mat = gen_single_RP_mat(single_feature_seg, dim,
                                       tau)  # (n_vec, n_vec)  , n_vec: number of phase space trajectory
            RP_mat = np.expand_dims(RP_mat, 2)  # (n_vec, n_vec, 1)
            multi_channels_RP_mat.append(RP_mat)
        multi_channels_RP_mat = np.stack(multi_channels_RP_mat, axis=2)  # (n_vec, n_vec, n_feature, 1)
        multi_channels_RP_mat = np.squeeze(multi_channels_RP_mat)  # (n_vec, n_vec, n_feature)
        multi_channels_RP_mat = np.expand_dims(multi_channels_RP_mat, axis=0) # (1, n_vec, n_vec, n_feature)
        RP_mats_h5array.append(multi_channels_RP_mat)
    print(RP_mats_h5array.shape)
    synchronized_close_file(lock, RP_mats_h5file)
    print('*end a thread')

if __name__ == '__main__':
    n_cpus = multiprocessing.cpu_count()
    print(f'n_thread:{n_cpus}')
    pool = multiprocessing.Pool(processes=1) # pytables multi-thread support TODO
    start = time.time()
    parser = argparse.ArgumentParser(description='RP_mat')
    parser.add_argument('--dim', default=DIM, type=int)
    parser.add_argument('--tau', default=TAU, type=int)
    parser.add_argument('--feature_set', type=str)
    parser.add_argument('--trjs_segs_features_path', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()
    dim = args.dim
    tau = args.tau
    save_path = args.save_path
    print('dim:{} tau:{}'.format(dim, tau))
    if args.feature_set is None:
        feature_set = FEATURES_SET_1
    else:
        feature_set = [int(item) for item in args.feature_set.split(',')]
    print('feature_set:{}'.format(feature_set))

    trjs_segs_features = np.load(args.trjs_segs_features_path)
    features_RP_mats = []
    n_features = trjs_segs_features.shape[3]
    n_samples = trjs_segs_features.shape[0]
    n_timestamps = trjs_segs_features.shape[2]
    trjs_segs_features = np.squeeze(trjs_segs_features)

    n_vectors = n_timestamps - (dim - 1) * tau
    with tb.open_file(args.save_path, mode='w') as RP_mats_h5:
        RP_mats_h5array = RP_mats_h5.create_earray(
            '/',
            DATA_NAME,  # 数据名称，之后需要通过它来访问数据
            tb.Float32Atom(),  # 设定数据格式（和data1格式相同）
            shape=(0, n_vectors, n_vectors, n_features),  # 第一维的 0 表示数据可沿行扩展
            # filters=tb.Filters(complevel=5, complib='blosc'),
            expectedrows=n_samples
        )
    generate_RP_mats(trjs_segs_features, dim, tau, n_features, save_path, DATA_NAME)
    RP_mats_h5file = synchronized_open_file(lock, save_path, mode='r')
    RP_mats_h5array = RP_mats_h5file.get_node('/' + DATA_NAME)
    print(RP_mats_h5array.shape)
    # for i in range(n_features):
    #     single_feature_segs = trjs_segs_features[:, :, i]  # (n, seg_size)
    #     a1 = gen_single_RP_mat(single_feature_segs[0], dim, tau)
    #     # generate RP mat for each seg
    #     feature_RP_mats = __gen_multiple_RP_mats(multi_series=single_feature_segs[:], scale=False, dim=dim, tau=tau)
    #     feature_RP_mats = np.expand_dims(feature_RP_mats, axis=3)
    #     features_RP_mats.append(feature_RP_mats)
    #     max = np.max(feature_RP_mats)
    #     min = np.min(feature_RP_mats)
    #     print('RP mat value range:', min, max)
    # features_RP_mats = np.concatenate(features_RP_mats, axis=3)
    #
    # print('features_RP_mats.shape:{}'.format(features_RP_mats.shape))
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    # np.save(args.save_path, scale_RP_each_feature(features_RP_mats))  # note scaled!
