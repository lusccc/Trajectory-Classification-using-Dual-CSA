import argparse
import os
import pathlib
import tensorflow as tf
import logzero
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.python.keras.layers import Input, Dense, LSTM, concatenate, Activation
from tensorflow.python.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model

import numpy as np

from logzero import logger
from params import modes_to_use, N_CLASS, MAX_SEGMENT_SIZE, FEATURES_SET_1

pathlib.Path(os.environ['RES_PATH']).mkdir(parents=True, exist_ok=True)
logzero.logfile(os.path.join(os.environ['RES_PATH'], 'log.txt'), backupCount=3)


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def LSTM_FCN_Softmax(timesteps, embedding_dim, n_features, n_class):
    ip = Input(shape=(timesteps, n_features))

    x = LSTM(int(embedding_dim))(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(n_class, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    plot_model(model, to_file=os.path.join(os.environ['RES_PATH'], 'lstm_fcn_softmax.png'), show_shapes=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train(epochs=100, batch_size=200):
    logger.info('training...')
    model_checkpoint = ModelCheckpoint(os.path.join(os.environ['RES_PATH'], "lstm_fcn_softmax.model"), verbose=1,
                                       monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=2)
    callback_list = [model_checkpoint, early_stopping]

    hist = model.fit(x_train, y_train, epochs=epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(
                         [x_test],
                         [y_test]),
                     callbacks=callback_list)
    score = np.argmax(hist.history['val_accuracy'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[score],
                                                                             np.max(hist.history['val_accuracy'])))


def show_confusion_matrix(model, x_test, y_test):
    pred = model.predict(x_test)
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    logger.info(cm)
    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    logger.info(re)
    with open(os.path.join(os.environ['RES_PATH'], 'classification_results.txt'), 'a') as f:
        print(cm, file=f)
        print(re, file=f)


def load_data(dataset_name, data_type):
    dataset_name = dataset_name
    data_type = data_type
    multi_feature_segs = np.load(f'./data/{dataset_name}_features/multi_feature_segs_{data_type}_normalized.npy')
    multi_feature_segs = np.swapaxes(multi_feature_segs, 1, 2)
    labels = np.load(f'./data/{dataset_name}_features/multi_feature_seg_labels_{data_type}.npy')
    return multi_feature_segs, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM_FCN_Softmax')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results-path', required=True, type=str)
    args = parser.parse_args()
    dataset = args.dataset

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = LSTM_FCN_Softmax(MAX_SEGMENT_SIZE, 32, len(FEATURES_SET_1), N_CLASS)
    patience = 20
    x_train, y_train = load_data(dataset, 'train')
    x_test, y_test = load_data(dataset, 'test')
    print()
    train(3000)
    model = load_model(os.path.join(os.environ['RES_PATH'], "lstm_fcn_softmax.model"),
                       custom_objects={'N_CLASS': N_CLASS})
    show_confusion_matrix(model, x_test, y_test)
