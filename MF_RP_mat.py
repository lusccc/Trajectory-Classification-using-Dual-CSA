import argparse
import math
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

import utils
from params import DIM, TAU, FEATURES_SET_1

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
        RP_mats = utils.scale_any_shape_data(RP_mats)
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


def generate_RP_mats(trjs_segs_features, dim, tau, n_features, save_path):
    print('gen_RP_mats...')
    tasks = []
    batch_size = int(len(trjs_segs_features) / n_cpus + 1)
    for i in range(0, n_cpus):
        tasks.append(pool.apply_async(do_generate_RP_mats,
                                      (trjs_segs_features[i * batch_size:(i + 1) * batch_size], dim, tau, n_features,
                                       )))
    res = [t.get() for t in tasks]
    RP_mats = np.concatenate(res)
    RP_mats = utils.scale_RP_each_feature(RP_mats)
    print(f'save to {save_path}')
    np.save(save_path, RP_mats)

def do_generate_RP_mats(multi_features_segs, dim, tau, n_features):
    '''
    :param multi_features_segs: (n, seg_size, n_features)
    '''
    RP_mats = []
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
        # multi_channels_RP_mat = np.expand_dims(multi_channels_RP_mat, axis=0)  # (1, n_vec, n_vec, n_feature)
        RP_mats.append(multi_channels_RP_mat)
    RP_mats = np.array(RP_mats)
    print(RP_mats.shape)
    print('*end a thread')
    return RP_mats

if __name__ == '__main__':
    n_cpus = multiprocessing.cpu_count()
    print(f'n_thread:{n_cpus}')
    pool = multiprocessing.Pool(processes=n_cpus)
    start = time.time()
    parser = argparse.ArgumentParser(description='RP')
    parser.add_argument('--dim', default=DIM, type=int)
    parser.add_argument('--tau', default=TAU, type=int)
    parser.add_argument('--feature_set', type=str)
    parser.add_argument('--trjs_segs_features_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--n_parts', type=int, default=1)  # divide data into n parts to save final RP mats.

    args = parser.parse_args()
    dim = args.dim
    tau = args.tau
    save_path = args.save_path
    n_parts = args.n_parts
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

    part_size = math.ceil(n_samples / n_parts)
    print(f'n_samples:{n_samples}, n_parts:{n_parts}, part_size:{part_size}')
    for i in range(n_parts):
        full_save_path = save_path if n_parts == 1 else save_path+f"_p{i}"
        generate_RP_mats(trjs_segs_features[i * part_size: (i + 1) * part_size], dim, tau, n_features, full_save_path)

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    # np.save(args.save_path, scale_RP_each_feature(features_RP_mats))  # note scaled!
