import os

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, LSTM, concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report

from dataset import *
from params import FEATURES_SET_1, MAX_SEGMENT_SIZE
from trajectory_extraction import modes_to_use

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

    plot_model(model, to_file='./comparison_results/lstm_fcn_softmax.png', show_shapes=True)

    learning_rate = 1e-3


    optm = Adam(lr=learning_rate)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train(epochs=100, batch_size=200):
    factor = 1. / np.cbrt(2)
    model_checkpoint = ModelCheckpoint("./comparison_results/lstm_fcn_sofrmax.model", verbose=1,
                                       monitor='val_acc', save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=2)
    callback_list = [model_checkpoint, reduce_lr, early_stopping]

    hist = model.fit(np.squeeze(x_features_series_train), y_train, epochs=epochs,
                     batch_size=batch_size, shuffle=True,
                     validation_data=(
                            [np.squeeze(x_features_series_test)],
                            [y_test]),
                     callbacks=callback_list)
    score = np.argmax(hist.history['val_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[score],
                                                                             np.max(hist.history['val_acc'])))

def show_confusion_matrix():
    model = load_model('./comparison_results/lstm_fcn_sofrmax.model', custom_objects={'N_CLASS': N_CLASS})
    pred = model.predict([np.squeeze(x_features_series_test)])
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    print(cm)
    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    print(re)


if __name__ == "__main__":

    model = LSTM_FCN_Softmax(MAX_SEGMENT_SIZE, 32, len(FEATURES_SET_1), N_CLASS)
    patience = 35
    train(3000)
    show_confusion_matrix()