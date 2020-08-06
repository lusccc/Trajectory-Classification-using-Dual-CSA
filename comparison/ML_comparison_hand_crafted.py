import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Geolife_trajectory_extraction import modes_to_use
from params import FEATURES_SET_1

x_features_series_train = np.load('./data/geolife_features/trjs_segs_features_all_features_train.npy', )
x_features_series_test = np.load('./data/geolife_features/trjs_segs_features_all_features_test.npy', )
y_train = np.load('./data/geolife_features/trjs_segs_features_labels_all_features_train.npy', )
y_test = np.load('./data/geolife_features/trjs_segs_features_labels_all_features_test.npy', )

n_features = len(FEATURES_SET_1)
x_train = np.squeeze(x_features_series_train)
x_test = np.squeeze(x_features_series_test)


def calc_handcrafted_features(feature_series):
    handcrafted_features_series = []
    HC = 19  # Heading rate threshold
    VS = 3.4  # Stop rate threshold
    VR = 0.26  # VCR threshold
    for single_serires in feature_series:
        # 0        1     2  3  4  5  6   7    8  9
        # delta_t, hour, d, v, a, h, hc, hcr, s, tn
        delta_ts = single_serires[:, 0]
        dists = single_serires[:, 2]
        vs = single_serires[:, 3]
        accs = single_serires[:, 4]
        hcs = single_serires[:, 6]
        hcrs = single_serires[:, 7]

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
        handcrafted_features_series.append(
            [length, avg_v, exp_v, var_v, max_v1s, max_v2s, max_v3s, max_a1s, max_a2s, max_a3s, Pc * 1. / length,
             Ps * 1. / length, Pv * 1. / length])
    return np.array(handcrafted_features_series)


x_handcrafted_train = calc_handcrafted_features(x_train)
x_handcrafted_test = calc_handcrafted_features(x_test)


def show_confusion_matrix(y_pred):
    print('\n###########')
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    print(cm)
    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    print(re)


# ml_models = [RandomForestClassifier(), KNeighborsClassifier(), MLPClassifier(), DecisionTreeClassifier()]
# for i, model in enumerate(ml_models):
#     print('$$$$ {} $$$$'.format(i))
#     model.fit(x_handcrafted_train, y_train)
#     y_pred = model.predict(x_handcrafted_test)
#     y_pred = np.argmax(y_pred, axis=1)
#     show_confusion_matrix(y_pred)


def svc_classification():
    # model = SVC()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # for svc, y is not one-hot encoding
    y_train_ = np.argmax(y_train, axis=1)
    clf.fit(x_handcrafted_train, y_train_)
    y_pred = clf.predict(x_handcrafted_test)
    show_confusion_matrix(y_pred)


svc_classification()
