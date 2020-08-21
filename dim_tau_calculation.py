import argparse
import multiprocessing
import time
from math import ceil

import numpy as np
from scipy.signal import argrelextrema

from fnn_mi import mutualInformation, false_nearest_neighours
from params import FEATURES_SET_1


def calc_tau(trjs_segs_features, n_features, seg_size):
    tasks = []
    for i in range(n_features):
        single_feature_segs = trjs_segs_features[:, :, i]
        batch_size = int(len(single_feature_segs) / n_cpus + 1)
        for i in range(0, n_cpus):
            tasks.append(pool.apply_async(do_calc_tau, (
                single_feature_segs[i * batch_size:(i + 1) * batch_size], seg_size)))
    print(f'n_task:{len(tasks)}')
    res = np.array([t.get() for t in tasks])
    tau_candidates = np.hstack(res)
    print(tau_candidates.shape)
    final_tau = np.mean(tau_candidates)
    print(f'final_tau:{final_tau}, ceil:{ceil(final_tau)}')
    return ceil(final_tau)


def do_calc_tau(single_feature_segs, seg_size):
    n_bins = int(seg_size / 10.)
    delay_range = int(
        seg_size / 5.)  # in kaggle example, for 100points series, delay range is set to 1-20, i.e. one fifth
    tau_candidates = []
    for single_feature_seg in single_feature_segs:
        mis = []
        for j in range(1, delay_range):
            mi = mutualInformation(single_feature_seg, j, n_bins)
            mis.append(mi)
        mis = np.array(mis)
        local_mins = argrelextrema(mis, np.less)
        if len(local_mins[0]) == 0:  # no local mins
            continue
        first_local_min = local_mins[0][0] + 1  # start from 1, hence+1
        # print(first_local_min)
        tau_candidates.append(first_local_min)
    tau_candidates = np.array(tau_candidates)
    print('* end a thread for calc_tau')
    return tau_candidates


def calc_dim(trjs_segs_features, n_features, seg_size, tau):
    tasks = []
    for i in range(n_features):
        single_feature_segs = trjs_segs_features[:, :, i]
        batch_size = int(len(single_feature_segs) / n_cpus + 1)
        for i in range(0, n_cpus):
            tasks.append(pool.apply_async(do_calc_dim, (
                single_feature_segs[i * batch_size:(i + 1) * batch_size], seg_size, tau)))
    print(f'n_task:{len(tasks)}')
    res = np.array([t.get() for t in tasks])
    dim_candidates = np.hstack(res)
    print(dim_candidates.shape)
    final_dim = np.mean(dim_candidates)
    print(f'final dim:{final_dim}, ceil:{ceil(final_dim)}')


def do_calc_dim(single_feature_segs, seg_size, tau):
    dim_candidates = []
    for single_feature_seg in single_feature_segs:
        for i in range(1, seg_size):  # i is dim try to looking for
            "delay*dimension must be < len(data)"
            if i * tau > seg_size:
                break
            fnn_fraction = false_nearest_neighours(single_feature_seg, tau, i) / seg_size
            '''Until the proportion of the false nearest critical point is less than
             5% or the false nearest critical point no longer decreases with 
             the increase of dim, it can be considered that the chaotic attractor 
             has been fully opened, and dim at this time is the embedded dimension'''
            if fnn_fraction < .05:
                dim_candidates.append(i)
                break
    dim_candidates = np.array(dim_candidates)
    print('* end a thread for calc_dim')
    return dim_candidates


if __name__ == '__main__':
    start = time.time()
    n_cpus = multiprocessing.cpu_count()
    print(f'n_thread:{n_cpus}')
    pool = multiprocessing.Pool(processes=n_cpus)
    parser = argparse.ArgumentParser(description='dimtau')
    parser.add_argument('--feature_set', type=str)
    parser.add_argument('--trjs_segs_features_train_path', type=str,
                        default='data/SHL_features/trjs_segs_features_train.npy')
    parser.add_argument('--trjs_segs_features_test_path', type=str,
                        default='data/SHL_features/trjs_segs_features_test.npy')
    args = parser.parse_args()
    if args.feature_set is None:
        feature_set = FEATURES_SET_1
    else:
        feature_set = [int(item) for item in args.feature_set.split(',')]

    # stack to get all data
    trjs_segs_features = np.vstack(
        [np.load(args.trjs_segs_features_train_path), np.load(args.trjs_segs_features_test_path)])
    trjs_segs_features = np.squeeze(trjs_segs_features)
    n_features = trjs_segs_features.shape[2]
    seg_size = trjs_segs_features.shape[1]

    print('calc_tau...')
    tau = calc_tau(trjs_segs_features[:], n_features, seg_size)
    print('calc_dim...')
    calc_dim(trjs_segs_features[:], n_features, seg_size, tau)
    print()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
