import numpy as np
from keras import backend as K, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Flatten, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Lambda
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split

from PEDCC import LATENT_VARIABLE_DIM, get_centroids, N_CLASS

import os

from movement_features import MAX_SEGMENT_SIZE

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

classification_layer = Lambda(lambda x: student_t(x[0], x[1]), output_shape=lambda x: (x[0][0], N_CLASS))


def student_t(z, u, alpha=1.):
    q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(z, axis=1) - u), axis=2) / alpha))
    q **= (alpha + 1.0) / 2.0
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    return q


def load_data():
    test_n = 100
    x_segs = np.load('./geolife/features_segments.npy')[:]
    n_samples = x_segs.shape[0]
    x_centroids = get_centroids(n_samples)[:]  # shape(n, 10,48)
    y = np.load('./geolife/features_segments_labels.npy')[:]
    print('load_RP_mats x.shape:{} y.shape{}'.format(x_segs.shape, y.shape))
    train_x_segs, test_x_segs, train_y, test_y, \
    train_x_centroids, test_x_centroids, train_y, test_y = train_test_split(x_segs, y,
                                                                            x_centroids, y,
                                                                            test_size=0.20, random_state=7,
                                                                            shuffle=True)
    train_y = to_categorical(train_y, num_classes=N_CLASS)
    test_y = to_categorical(test_y, num_classes=N_CLASS)
    return x_segs, \
           train_x_segs, test_x_segs, train_y, test_y, \
           train_x_centroids, test_x_centroids, train_y, test_y


x_segs, \
train_x_segs, test_x_segs, train_y, test_y, \
train_x_centroids, test_x_centroids, train_y, test_y = load_data()

RP_mat_size = train_x_segs.shape[1]  # 40
n_features = train_x_segs.shape[3]

""" -----auto-encoder------"""
ts_conv_ae = Sequential()
activ = 'relu'
ts_conv_ae.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ, input_shape=(1, MAX_SEGMENT_SIZE, n_features)))
ts_conv_ae.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(MaxPooling2D(pool_size=(1, 2)))
ts_conv_ae.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(MaxPooling2D(pool_size=(1, 2)))
ts_conv_ae.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(MaxPooling2D(pool_size=(1, 2)))

ts_conv_ae.add(Flatten())
ts_conv_ae.add(Dense(units=10, name='embedding'))
ts_conv_ae.add(Dense(units=1 * 25 * 128, activation=activ))
ts_conv_ae.add(Reshape((1, 25, 128)))

ts_conv_ae.add(UpSampling2D(size=(1, 2)))
ts_conv_ae.add(Conv2DTranspose(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(Conv2DTranspose(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(UpSampling2D(size=(1, 2)))
ts_conv_ae.add(Conv2DTranspose(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(Conv2DTranspose(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(UpSampling2D(size=(1, 2)))
ts_conv_ae.add(Conv2DTranspose(32, (1, 3), strides=(1, 1), padding='same', activation=activ))
ts_conv_ae.add(Conv2DTranspose(4, (1, 3), strides=(1, 1), padding='same', activation=activ, name='reconstruction'))

ts_conv_encoder = Model(inputs=ts_conv_ae.input, outputs=ts_conv_ae.get_layer(name='embedding').output)

ts_conv_ae.summary()
plot_model(ts_conv_ae, to_file='ts_conv_ae.png', show_shapes=True)
ts_conv_ae.compile(optimizer='adam', loss='mse')

"""--------SAE----------"""
centroids_input = Input((N_CLASS, LATENT_VARIABLE_DIM))
classification = classification_layer([ts_conv_ae.get_layer('embedding').output, centroids_input],
                                      )

ts_conv_SAE = Model(
    inputs=[ts_conv_ae.get_layer(index=0).input, centroids_input],
    outputs=[classification, ts_conv_ae.get_layer('reconstruction').output])
ts_conv_SAE.summary()
plot_model(ts_conv_SAE, to_file='./results/ts_conv_SAE.png', show_shapes=True)
ts_conv_SAE.compile(loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam')

"""-----------train---------------"""


def pretrain():
    ts_conv_ae.fit(x_segs, x_segs, batch_size=2000, epochs=2000)
    ts_conv_ae.save('./results/ts_conv_ae.model')


def train():
    print('training...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
    hist = ts_conv_SAE.fit([train_x_segs, train_x_centroids], [train_y, train_x_segs],
                           epochs=30,
                           batch_size=2000, shuffle=True,
                           validation_data=(
                            [test_x_segs, test_x_centroids], [test_y, test_x_segs]),
                           callbacks=[early_stopping])
    ts_conv_SAE.save('./results/conv_SAE.model')
    A = np.argmax(hist.history['val_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[A],
                                                                             np.max(hist.history['val_acc'])))


pretrain()
