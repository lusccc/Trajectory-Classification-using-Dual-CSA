from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from params import N_CLASS, movement_features, MAX_SEGMENT_SIZE, TOTAL_EMBEDDING_DIM

from dataset import *
from trajectory_extraction import modes_to_use

ml_models = [SVC(), MLPClassifier(), ]

x_train = np.squeeze(x_trj_seg_clean_of_train)
x_test = np.squeeze(x_trj_seg_clean_of_test)


def show_confusion_matrix(y_pred):
    print('\n###########')
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    print(cm)
    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    print(re)

for i, model in enumerate(ml_models):
    print('$$$$ {} $$$$'.format(i))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    show_confusion_matrix(y_pred)
