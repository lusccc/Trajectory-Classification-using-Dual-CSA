# coding=utf-8
# https://heremaps.github.io/pptk/tutorials/viewer/geolife.html
import glob
import os
import os.path
import time

import numpy as np
import pandas as pd

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
# walk, bike, bus, driving, train/subway
modes_to_use = [0, 1, 2, 3, 4]
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
        extract_trjs_with_labels(points, labels_details)
    else:
        points['label'] = 0

    return points


# my code
def extract_trjs_with_labels(points, labels_details):
    points['time'] = points['time'].apply(datatime_to_timestamp)
    labels_details['start_time'] = labels_details['start_time'].apply(datatime_to_timestamp)
    labels_details['end_time'] = labels_details['end_time'].apply(datatime_to_timestamp)
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

        # print(label)
        # print(trj)
        # print('\n')


def read_all_users(folder):
    subfolders = os.listdir(folder)
    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        df = read_user(os.path.join(folder, sf))
        df['user'] = int(sf)
        dfs.append(df)
    return pd.concat(dfs)


def datatime_to_timestamp(dt):
    # print dt
    """时间转换为时间戳，单位秒"""
    # 转换成时间数组
    time_array = time.strptime(str(dt), "%Y-%m-%d %H:%M:%S")
    # 转换成时间戳
    timestamp = time.mktime(time_array)  # 单位 s
    # print timestamp
    return int(timestamp)


if __name__ == '__main__':
    trjs = []
    trjs_labels = []
    df = read_all_users('./Data')
    trjs = np.array(trjs)
    labels = np.array(trjs_labels)

    # 保存带标签的轨迹和 标签
    np.save('./geolife/trjs.npy', trjs)
    np.save('./geolife/labels.npy', labels)
