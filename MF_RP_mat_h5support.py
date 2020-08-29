import argparse
import multiprocessing
import sys
import threading
import time
from math import cos, pi
from logzero import logger

import numpy as np
import tables as tb
from numpy.lib.stride_tricks import as_strided
from numpy.linalg import norm
from pyts.image.recurrence import _trajectories

from params import DIM, TAU, FEATURES_SET_1
from utils import scale_any_shape_data, scale_RP_each_feature, synchronized_open_file, synchronized_close_file

LOCK = threading.Lock()
H5_NODE_NAME = 'RP_data'


def __gen_multiple_RP_mats(multi_series, scale=False, dim=DIM, tau=TAU):
    """
    deprecated!
    generate a RP mat for each series, note: large memory required
    """

    n_series = len(multi_series)

    logger.info('calc distance_mats ....')
    # each series generates n_vectors phase space trajectories
    stri = multi_series.strides
    its = multi_series.itemsize
    logger.info(stri, its)
    phase_vectorss = _trajectories(multi_series, dimension=dim, time_delay=tau)
    n_vectors = phase_vectorss.shape[1]
    # calc distances mats, 1 mat for each series.
    # shape:(n_samples, n_vectors, n_vectors)
    # from source code in pyts.image.recurrence.RecurrencePlot.transform
    distance_mats = np.sqrt(
        np.sum((phase_vectorss[:, None, :, :] - phase_vectorss[:, :, None, :]) ** 2,
               axis=3, dtype='float32')
    )
    logger.info('end calc distance_mats')

    mycode = False
    if mycode:
        ## my code to calc distances
        RP_mats = []  # !!skipped Heavside function
        count = 0
        for distance_mat, phase_vectors in zip(distance_mats, phase_vectorss):
            logger.info('{}/{}'.format(count, n_series))
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
    # The strides of an array tell us how many bytes we have to skip in memory to move to the next position along a
    # certain axis.
    n_timestamps = len(singel_series)
    s = singel_series.strides[0]
    # based on source code in pyts.image.recurrence.RecurrencePlot.transform
    shape_new = (n_timestamps - (dimension - 1) * time_delay, dimension)
    strides_new = (s, time_delay * s)
    phase_space_trjs = as_strided(singel_series, shape_new, strides_new)
    distance_mat = np.sqrt(
        np.sum((phase_space_trjs[None, :, :] - phase_space_trjs[:, None, :]) ** 2,
               axis=2, dtype='float32')
    )
    return distance_mat


def generate_RP_mats(n_thread, multi_feature_segs, dim, tau, n_features, save_path, h5node_name):
    logger.info('gen_RP_mats...')
    tasks = []
    batch_size = int(len(multi_feature_segs) / n_thread + 1)
    for i in range(0, n_thread):
        tasks.append(pool.apply_async(do_generate_RP_mats,
                                      (multi_feature_segs[i * batch_size:(i + 1) * batch_size], dim, tau,
                                       n_features, save_path, h5node_name
                                       )))
    res = [t.get() for t in tasks]


def do_generate_RP_mats(multi_features_segs, dim, tau, n_features, save_path, h5node_name):
    """
    :param multi_features_segs: (n, seg_size, n_features)
    """
    RP_mats_h5file = synchronized_open_file(LOCK, save_path, mode='a')
    RP_mats_h5array = RP_mats_h5file.get_node('/' + h5node_name)
    for multi_features_seg in multi_features_segs:
        multi_channels_RP_mat = []
        for i in range(n_features):
            single_feature_seg = multi_features_seg[i, :]  # (n, seg_size)
            RP_mat = gen_single_RP_mat(single_feature_seg, dim,
                                       tau)  # (n_vec, n_vec)  , n_vec: number of phase space trajectory
            RP_mat = np.expand_dims(RP_mat, 0)  # (1, n_vec, n_vec)
            multi_channels_RP_mat.append(RP_mat)
        multi_channels_RP_mat = np.stack(multi_channels_RP_mat, axis=0)  # ( n_feature, 1, n_vec, n_vec)
        multi_channels_RP_mat = np.squeeze(multi_channels_RP_mat,  axis=1)  # (n_feature, n_vec, n_vec )
        multi_channels_RP_mat = np.expand_dims(multi_channels_RP_mat, axis=0)  # (1, n_feature, n_vec, n_vec)
        RP_mats_h5array.append(multi_channels_RP_mat)
    synchronized_close_file(LOCK, RP_mats_h5file)
    logger.info('*end a thread')


if __name__ == '__main__':
    start = time.time()
    # currently pytables not support multi-thread, hence processes=1
    n_thread = 1
    logger.info(f'n_thread:{n_thread}')
    pool = multiprocessing.Pool(processes=1)

    parser = argparse.ArgumentParser(description='RP')
    parser.add_argument('--dim', default=DIM, type=int)
    parser.add_argument('--tau', default=TAU, type=int)
    parser.add_argument('--multi_feature_segs_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()
    logger.info('dim:{} tau:{}'.format(args.dim, args.tau))

    # !!note load un-normalized data to generate RP
    multi_feature_segs = np.load(args.multi_feature_segs_path)
    n_features = multi_feature_segs.shape[1]
    n_samples = multi_feature_segs.shape[0]
    n_timestamps = multi_feature_segs.shape[2]

    n_vectors = n_timestamps - (args.dim - 1) * args.tau
    with tb.open_file(args.save_path, mode='w') as RP_mats_h5:
        RP_mats_h5array = RP_mats_h5.create_earray(
            '/',
            H5_NODE_NAME,
            tb.Float32Atom(),
            shape=(0, n_features, n_vectors, n_vectors),
            # filters=tb.Filters(complevel=5, complib='blosc'),
            expectedrows=n_samples
        )
    generate_RP_mats(n_thread, multi_feature_segs, args.dim, args.tau, n_features, args.save_path, H5_NODE_NAME)
    end = time.time()
    logger.info('Running time: %s Seconds' % (end - start))
