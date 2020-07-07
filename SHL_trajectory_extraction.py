# coding=utf-8
# reference to https://heremaps.github.io/pptk/tutorials/viewer/geolife.html
import glob
import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from params import *

from utils import datatime_to_timestamp

# Null=0, Still=1, Walking=2, Run=3, Bike=4, Car=5, Bus=6, Train=7, Subway=8
use_modes = [2, 3, 4, 5, 6, 7, 8]
print('use_modes:', use_modes)

def process_labels(labels_file_path, locations_file_path):
    sampled_labels = pd.read_csv(labels_file_path, header=None, sep=' ', usecols=[0, 1])
    sampled_labels.columns = ['timestamp', 'label']

    #https://stackoverflow.com/questions/48997350/pandas-dataframe-groupby-for-separate-groups-of-same-value
    sampled_labels['marker'] = (sampled_labels['label'] != sampled_labels['label'].shift()).cumsum()
    labels_time_range = sampled_labels.groupby('marker').agg({'label': 'first', 'timestamp': lambda x: list(x)})

    gps_records = pd.read_csv(locations_file_path, header=None, sep=' ', usecols=[0, 4, 5])
    gps_records.columns = ['timestamp', 'lat', 'lon']

    for idx, label_detail in labels_time_range.iterrows():
        label = label_detail['label']
        if label not in use_modes:
            continue
        # merge walking&run, train&subway
        if label == 3:
            label = 2
        if label == 8:
            label = 7
        st = label_detail['timestamp'][0]
        et = label_detail['timestamp'][-1]
        trj = gps_records[(gps_records['timestamp'] >= st) & (gps_records['timestamp'] <= et)]
        trj = trj[['timestamp', 'lat', 'lon']].values
        trjs.append(trj)
        trjs_labels.append(label)

def read_all_folders(path):
    folders = os.listdir(path)
    n = len(folders)
    for i, f in enumerate(folders):
        print(f'{i+1}/{n}')
        labels_file_path = os.path.join(path, f, 'Label.txt')
        locations_file_path = os.path.join(path, f, 'Hips_Location.txt')
        process_labels(labels_file_path, locations_file_path)


if __name__ == '__main__':
    trjs = []
    trjs_labels = []
    read_all_folders('/mnt/e/DATASET/SHLDataset_User1Hips_v1/release/User1')
    trjs = np.array(trjs)
    labels = np.array(trjs_labels)
    trjs, labels = shuffle(trjs, labels, random_state=0)

    print('saving files...')
    if not os.path.exists('./data/SHL_extracted/'):
        os.makedirs('./data/SHL_extracted/')

    trjs_train, trjs_test, \
    labels_train, labels_test \
        = train_test_split(
        trjs,
        labels,
        test_size=0.20, random_state=7, shuffle=False  # already shuffled, no shuffle here
    )

    np.save('./data/SHL_extracted/trjs_train.npy', trjs_train)
    np.save('./data/SHL_extracted/trjs_test.npy', trjs_test)
    np.save('./data/SHL_extracted/labels_train.npy', labels_train)
    np.save('./data/SHL_extracted/labels_test.npy', labels_test)