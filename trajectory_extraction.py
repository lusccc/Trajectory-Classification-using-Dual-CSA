# coding=utf-8
# https://heremaps.github.io/pptk/tutorials/viewer/geolife.html
import glob
import os
import os.path

import numpy as np
import pandas as pd
from params import *

from utils import datatime_to_timestamp

import multiprocessing



MODE_NAMES = ['walk', 'bike', 'bus', 'car', 'subway', 'train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
# mode_ids = {s : i + 1 for i, s in enumerate(mode_names)}
modes = {}
for i, s in enumerate(MODE_NAMES):
    if s == 'taxi':
        modes[s] = 3
    elif s == 'train':
        modes[s] = 4
    else:
        modes[s] = i
print('modes:', modes)

print('modes to use:', modes_to_use)


def read_plt(plt_file):
    points = pd.read_csv(plt_file, skiprows=6, header=None,
                         parse_dates=[[5, 6]], infer_datetime_format=True)

    # for clarity rename columns
    points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt'})

    # remove unused columns
    points.drop(inplace=True, columns=[2, 4])

    return points


def read_labels(labels_file):
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)

    # for clarity rename columns
    labels.columns = ['start_time', 'end_time', 'label']

    # replace 'label' column with integer encoding
    labels['label'] = [modes[i] for i in labels['label']]

    return labels


def apply_labels(points, labels):
    # search points's time between which two start_time
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1
    # did not find so the index is 0,
    no_label = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    points['label'] = labels['label'].iloc[indices].values
    points['label'][no_label] = 0


# modified
def read_user(user_folder):
    labels_details = None

    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    points = pd.concat([read_plt(f) for f in plt_files])
    # print points[['lat', 'lon']].values

    labels_file = os.path.join(user_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels_details = read_labels(labels_file)
        trjs, trjs_labels =  extract_trjs_with_labels(points, labels_details)
        return trjs, trjs_labels
    else:
        points['label'] = 0
        return [], []



# my code
def extract_trjs_with_labels(points, labels_details):
    points['time'] = points['time'].apply(datatime_to_timestamp)
    labels_details['start_time'] = labels_details['start_time'].apply(datatime_to_timestamp)
    labels_details['end_time'] = labels_details['end_time'].apply(datatime_to_timestamp)
    trjs = []
    trjs_labels = []
    for idx, label_detail in labels_details.iterrows():
        label = label_detail['label']
        if label not in modes_to_use:
            continue
        st = label_detail['start_time']
        et = label_detail['end_time']
        trj = points[(points['time'] >= st) & (points['time'] <= et)]

        trj = trj[['time', 'lat', 'lon']].values
        trjs.append(trj)
        trjs_labels.append(label)

    return trjs, trjs_labels


def read_users(root_folder, subfolders):
    trjs_res = []
    trjs_labels_res = []
    for i, sf in enumerate(subfolders):
        print(' processing user %s' % (sf))
        trjs, trjs_labels = read_user(os.path.join(root_folder, sf))
        trjs_res.extend(trjs)
        trjs_labels_res.extend(trjs_labels)
    return trjs_res, trjs_labels_res


if __name__ == '__main__':
    n_cpus = multiprocessing.cpu_count()
    print('n_thread:{}'.format(n_cpus))
    pool = multiprocessing.Pool(processes=n_cpus)

    root_folder = './data/geolife_raw'
    user_folders = os.listdir(root_folder)
    n_user_folders = len(user_folders)

    tasks = []
    batch_size = int(n_user_folders / n_cpus + 1)
    for i in range(0, n_cpus):
        tasks.append(pool.apply_async(read_users, (root_folder, user_folders[i:i + batch_size])))

    res = np.array([[t.get()[0], t.get()[0]] for t in tasks])
    print(np.shape(res))
    trjs = np.concatenate(res[:, 0])
    labels = np.concatenate(res[:, 1])
    print(trjs.shape)
    print(labels.shape)

    np.save('./data/geolife_extracted/trjs.npy', trjs)
    np.save('./data/geolife_extracted/labels.npy', labels)
