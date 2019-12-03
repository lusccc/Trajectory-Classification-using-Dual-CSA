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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

classification_layer = Lambda(lambda x: student_t(x[0], x[1]), output_shape=lambda x: (x[0][0], N_CLASS))


def student_t(z, u, alpha=1.):
    q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(z, axis=1) - u), axis=2) / alpha))
    q **= (alpha + 1.0) / 2.0
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    return q


def load_data():
    test_n = 100
    x_RP = np.load('./geolife/features_RP_mats.npy')[:test_n]
    n_samples = x_RP.shape[0]
    x_centroids = get_centroids(n_samples)[:test_n]  # shape(n, 10,48)
    y = np.load('./geolife/features_segments_labels.npy')[:test_n]
    print('load_RP_mats x.shape:{} y.shape{}'.format(x_RP.shape, y.shape))
    train_x_RP, test_x_RP, train_y, test_y, \
    train_x_centroids, test_x_centroids, train_y, test_y = train_test_split(x_RP, y,
                                                                            x_centroids, y,
                                                                            test_size=0.20, random_state=7,
                                                                            shuffle=True)
    train_y = to_categorical(train_y, num_classes=N_CLASS)
    test_y = to_categorical(test_y, num_classes=N_CLASS)
    return x_RP, \
           train_x_RP, test_x_RP, train_y, test_y, \
           train_x_centroids, test_x_centroids, train_y, test_y


x_RP, \
train_x_RP, test_x_RP, train_y, test_y, \
train_x_centroids, test_x_centroids, train_y, test_y = load_data()

RP_mat_size = train_x_RP.shape[1]  # 40
n_features = train_x_RP.shape[3]

""" -----RP mat conv auto-encoder------"""
RP_conv_ae = Sequential()
activ = 'relu'
RP_conv_ae.add(
    Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation=activ,
           input_shape=(RP_mat_size, RP_mat_size, n_features)))
RP_conv_ae.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(MaxPooling2D(pool_size=(2, 2)))
RP_conv_ae.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(MaxPooling2D(pool_size=(2, 2)))
RP_conv_ae.add(Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(MaxPooling2D(pool_size=(2, 2)))

RP_conv_ae.add(Flatten())
RP_conv_ae.add(Dense(units=LATENT_VARIABLE_DIM, name='RP_conv_embedding'))
RP_conv_ae.add(Dense(units=5 * 5 * 128, activation=activ))
RP_conv_ae.add(Reshape((5, 5, 128)))

RP_conv_ae.add(UpSampling2D(size=(2, 2)))
RP_conv_ae.add(Conv2DTranspose(128, (1, 1), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(Conv2DTranspose(128, (1, 1), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(UpSampling2D(size=(2, 2)))
RP_conv_ae.add(Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(UpSampling2D(size=(2, 2)))
RP_conv_ae.add(Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', activation=activ))
RP_conv_ae.add(
    Conv2DTranspose(n_features, (5, 5), strides=(1, 1), padding='same', activation=activ, name='RP_reconstruction'))

RP_conv_encoder = Model(inputs=RP_conv_ae.input, outputs=RP_conv_ae.get_layer(name='RP_conv_embedding').output)

RP_conv_ae.summary()
plot_model(RP_conv_ae, to_file='./results/RP_conv_ae.png', show_shapes=True)
RP_conv_ae.compile(optimizer='adam', loss='mse')

"""--------SAE----------"""
centroids_input = Input((N_CLASS, LATENT_VARIABLE_DIM))
classification = classification_layer([RP_conv_ae.get_layer('RP_conv_embedding').output, centroids_input],
                                      )

conv_SAE = Model(
    inputs=[RP_conv_ae.get_layer(index=0).input, centroids_input],
    outputs=[classification, RP_conv_ae.get_layer('RP_reconstruction').output])
conv_SAE.summary()
plot_model(conv_SAE, to_file='./results/Conv_SAE.png', show_shapes=True)
conv_SAE.compile(loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam')

"""-----------train---------------"""


def pretrain():
    RP_conv_ae.fit(x_RP, x_RP, batch_size=2000, epochs=5000)
    RP_conv_ae.save('./results/ts_conv_ae.model')


def train():
    print('training...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
    hist = conv_SAE.fit([train_x_RP, train_x_centroids], [train_y, train_x_RP],
                        epochs=30,
                        batch_size=2000, shuffle=True,
                        validation_data=(
                            [test_x_RP, test_x_centroids], [test_y, test_x_RP]),
                        callbacks=[early_stopping])
    conv_SAE.save('./results/conv_SAE.model')
    A = np.argmax(hist.history['val_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[A],
                                                                             np.max(hist.history['val_acc'])))


pretrain()
