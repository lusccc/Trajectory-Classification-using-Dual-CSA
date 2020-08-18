import os
import time

from keras.engine.saving import load_model
from sklearn.metrics import confusion_matrix, classification_report

from dataset_factory import *
from keras_support.network_keras import student_t, N_CLASS, modes_to_use

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# x_RP_test = np.load('./data/geolife_features/RP_mats_test_rdp0.8.npy', )
# x_features_series_test = np.load('./data/geolife_features/trjs_segs_features_test_rdp0.8.npy', )
# x_centroids_test = np.load('./data/geolife_features/centroids_test_rdp0.8.npy', )
# y_test = np.load('./data/geolife_features/trjs_segs_features_labels_test_rdp0.8.npy', )
print()

x_RP_test = x_RP_test[:1000]
x_features_series_test = x_features_series_test[:1000]
x_centroids_test = x_centroids_test[:1000]
y_test = y_test[:1000]


def predict(model_path):
    sae = load_model(model_path,
                     custom_objects={'student_t': student_t, 'N_CLASS': N_CLASS})
    t0 = time.time()
    pred = sae.predict([x_RP_test, x_centroids_test, x_features_series_test])
    print(f'predicting time:{time.time() - t0}')
    y_pred = np.argmax(pred[1], axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    print(cm)

    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    print(re)


if __name__ == '__main__':
    predict(os.path.join('results/default', 'sae_check_point.model'))
