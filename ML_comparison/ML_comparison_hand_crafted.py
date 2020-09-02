import argparse
import os
import pathlib

import logzero
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from logzero import logger
from params import modes_to_use, N_CLASS, MAX_SEGMENT_SIZE, FEATURES_SET_1

pathlib.Path(os.environ['RES_PATH']).mkdir(parents=True, exist_ok=True)
logzero.logfile(os.path.join(os.environ['RES_PATH'], 'log.txt'), backupCount=3)


def calc_handcrafted_features(feature_segments):
    handcrafted_features_segments = []
    HC = 19  # Heading rate threshold
    VS = 3.4  # Stop rate threshold
    VR = 0.26  # VCR threshold
    for single_segment in feature_segments:
        # 0        1     2  3  4  5  6   7    8  9
        # delta_t, hour, d, v, a, h, hc, hcr, s, tn
        delta_ts = single_segment[:, 0]
        dists = single_segment[:, 2]
        vs = single_segment[:, 3]
        accs = single_segment[:, 4]
        hcs = single_segment[:, 6]
        hcrs = single_segment[:, 7]

        length = np.sum(dists)
        avg_v = np.sum(dists) / np.sum(delta_ts)
        exp_v = np.mean(vs)
        var_v = np.var(vs)

        sorted_vs = np.sort(vs)[::-1]  # descending order
        max_v1s = sorted_vs[0]
        max_v2s = sorted_vs[1]
        max_v3s = sorted_vs[2]

        sorted_accs = np.sort(accs)[::-1]  # descending order
        max_a1s = sorted_accs[0]
        max_a2s = sorted_accs[1]
        max_a3s = sorted_accs[2]

        sorted_hcrs = np.sort(hcrs)[::-1]  # descending order
        max_h1s = sorted_hcrs[0]
        max_h2s = sorted_hcrs[1]
        max_h3s = sorted_hcrs[2]

        avg_hcrs = np.sum(hcs) / np.sum(delta_ts)
        exp_hcrs = np.mean(hcrs)
        var_hcrs = np.var(hcrs)

        # Heading change rate (HCR)
        Pc = sum(1 for item in list(hcrs) if item > HC)
        # Stop Rate (SR)
        Ps = sum(1 for item in list(vs) if item < VS)
        # Velocity Change Rate (VCR)
        Pv = sum(1 for item in list(accs) if item > VR)

        # length, avg_v, exp_v, var_v, max_v1s, max_v2s, max_v3s, max_a1s, max_a2s, max_a3s, max_h1s, max_h2s,
        #    max_h3s, avg_hcrs, exp_hcrs, var_hcrs
        handcrafted_features_segments.append(
            [length, avg_v, exp_v, var_v, max_v1s, max_v2s, max_v3s, max_a1s, max_a2s, max_a3s, Pc * 1. / length,
             Ps * 1. / length, Pv * 1. / length])
    return np.array(handcrafted_features_segments)


def svc_classification(dataset, x_handcrafted_train, x_handcrafted_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # for svc, y is not one-hot encoding
    y_train_ = np.argmax(y_train, axis=1)
    clf.fit(x_handcrafted_train, y_train_)
    y_pred = clf.predict(x_handcrafted_test)
    show_confusion_matrix(y_pred, y_test, 'SVC', dataset)


def ml_algorithms(dataset, x_handcrafted_train, x_handcrafted_test, y_train, y_test):
    ml_models = [RandomForestClassifier(), KNeighborsClassifier(), MLPClassifier(), DecisionTreeClassifier()]
    for i, model in enumerate(ml_models):
        print('$$$$ {} $$$$'.format(i))
        model.fit(x_handcrafted_train, y_train)
        y_pred = model.predict(x_handcrafted_test)
        y_pred = np.argmax(y_pred, axis=1)
        show_confusion_matrix(y_pred, y_test, model.__class__.__name__, dataset)


def show_confusion_matrix(y_pred, y_test, algorithm_name, dataset_name):
    print('\n###########')
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    logger.info(cm)
    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    logger.info(re)
    with open(os.path.join(os.environ['RES_PATH'], f'{dataset_name}_{algorithm_name}_classification_results.txt'), 'a') as f:
        print(cm, file=f)
        print(re, file=f)


def load_data(dataset_name, data_type):
    dataset_name = dataset_name
    data_type = data_type
    multi_feature_segs = np.load(f'./data/{dataset_name}_features/multi_feature_segs_{data_type}_all_features_normalized.npy')
    multi_feature_segs = np.swapaxes(multi_feature_segs, 1, 2)
    labels = np.load(f'./data/{dataset_name}_features/multi_feature_seg_labels_{data_type}.npy')
    return multi_feature_segs, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results-path', required=True, type=str)
    args = parser.parse_args()
    dataset = args.dataset

    x_train, y_train = load_data(dataset, 'train')
    x_test, y_test = load_data(dataset, 'test')
    x_handcrafted_train = calc_handcrafted_features(x_train)
    x_handcrafted_test = calc_handcrafted_features(x_test)
    ml_algorithms(dataset, x_handcrafted_train, x_handcrafted_test, y_train, y_test, )
    svc_classification(dataset, x_handcrafted_train, x_handcrafted_test, y_train, y_test, )
