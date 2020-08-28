import argparse
import os
import pathlib
import numpy as np
import logzero
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import SimpleRNN, Dense
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.utils.vis_utils import plot_model
from logzero import logger

from network_comparison.LSTM_FCN_Softmax_keras import show_confusion_matrix, load_data
from params import modes_to_use, N_CLASS, MAX_SEGMENT_SIZE, FEATURES_SET_1
import tensorflow as tf

pathlib.Path(os.environ['RES_PATH']).mkdir(parents=True, exist_ok=True)
logzero.logfile(os.path.join(os.environ['RES_PATH'], 'log.txt'), backupCount=3)

def RNN_Softmax(timesteps, embedding_dim, n_features, n_class):
    model = Sequential()
    model.add(SimpleRNN(
        # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
        # Otherwise, model.evaluate() will get error.
        input_shape=(timesteps, n_features),
        units=int(embedding_dim),
        # unroll=True,
    ))
    model.add(Dense(n_class, activation='softmax'))
    model.summary()
    plot_model(model, to_file=os.path.join(os.environ['RES_PATH'], 'rnn_softmax.png'), show_shapes=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(epochs=100, batch_size=200):
    model_checkpoint = ModelCheckpoint(os.path.join(os.environ['RES_PATH'], "rnn_softmax.model"), verbose=1,
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNN_Softmax')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--results-path', required=True, type=str)
    args = parser.parse_args()
    dataset = args.dataset

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = RNN_Softmax(MAX_SEGMENT_SIZE, 32, len(FEATURES_SET_1), N_CLASS)
    patience = 35
    x_train, y_train = load_data(dataset, 'train')
    x_test, y_test = load_data(dataset, 'test')
    train(3000)
    model = load_model(os.path.join(os.environ['RES_PATH'], "rnn_softmax.model"),
                       custom_objects={'N_CLASS': N_CLASS})
    show_confusion_matrix(model, x_test, y_test)
