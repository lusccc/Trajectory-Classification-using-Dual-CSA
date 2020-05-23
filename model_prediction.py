import os

from keras.engine.saving import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from Dual_SAE import student_t, N_CLASS, modes_to_use
from dataset import *


def predict(model_path, x_RP_test,x_features_series_test,y_test):
    sae = load_model(os.path.join(model_path, 'sae_check_point.model'),
                     custom_objects={'student_t': student_t, 'N_CLASS': N_CLASS})
    pred = sae.predict([x_RP_test, x_centroids_test, x_features_series_test])
    y_pred = np.argmax(pred[1], axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    print(cm)

    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    print(re)


def random_drop_samples(percentage=0.1):
    new_RPs = []
    new_fss = []
    new_ys = []

    for RP, fs, y in zip(x_RP_test,x_features_series_test,y_test):
        n = len(RP)
        n_drop = int(n * percentage)
        random_idx = np.random.choice(n, n_drop, replace=False)
        new_RPs.append(np.delete(RP, random_idx, axis=0))
        new_fss.append(np.delete(fs, axis=0))
        new_ys.append(np.delete(y, axis=0))

    return np.array(new_trjs)