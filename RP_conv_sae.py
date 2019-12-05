import numpy as np
from keras import backend as K, Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Conv2D, Flatten, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Lambda, \
    BatchNormalization, Activation
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split

from PEDCC import LATENT_VARIABLE_DIM, get_centroids, N_CLASS

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def student_t(z, u, alpha=1.):
    q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(z, axis=1) - u), axis=2) / alpha))
    q **= (alpha + 1.0) / 2.0
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    return q


classification_layer = Lambda(lambda x: student_t(x[0], x[1]), output_shape=lambda x: (x[0][0], N_CLASS))


def load_data():
    test_n = 5000
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
RP_conv_ae.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', input_shape=(RP_mat_size, RP_mat_size, n_features)))
RP_conv_ae.add(BatchNormalization())
RP_conv_ae.add(Activation(activ))
RP_conv_ae.add(MaxPooling2D(pool_size=(2, 2)))
RP_conv_ae.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
RP_conv_ae.add(BatchNormalization())
RP_conv_ae.add(Activation(activ))
RP_conv_ae.add(MaxPooling2D(pool_size=(2, 2)))
RP_conv_ae.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
RP_conv_ae.add(BatchNormalization())
RP_conv_ae.add(Activation(activ))
RP_conv_ae.add(MaxPooling2D(pool_size=(2, 2)))

RP_conv_ae.add(Flatten())
RP_conv_ae.add(Dense(units=LATENT_VARIABLE_DIM, name='RP_conv_embedding'))
RP_conv_ae.add(Dense(units=5 * 5 * 128, activation=activ))
RP_conv_ae.add(Reshape((5, 5, 128)))

RP_conv_ae.add(UpSampling2D(size=(2, 2)))
RP_conv_ae.add(Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same'))
RP_conv_ae.add(BatchNormalization())
RP_conv_ae.add(Activation(activ))
RP_conv_ae.add(UpSampling2D(size=(2, 2)))
RP_conv_ae.add(Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same'))
RP_conv_ae.add(BatchNormalization())
RP_conv_ae.add(Activation(activ))
RP_conv_ae.add(UpSampling2D(size=(2, 2)))
RP_conv_ae.add(Conv2DTranspose(n_features, (3, 3), strides=(1, 1), padding='same'))
RP_conv_ae.add(BatchNormalization())
RP_conv_ae.add(Activation(activ, name='RP_reconstruction'))

RP_conv_encoder = Model(inputs=RP_conv_ae.input, outputs=RP_conv_ae.get_layer(name='RP_conv_embedding').output)

RP_conv_ae.summary()
plot_model(RP_conv_ae, to_file='./results/RP_conv_ae.png', show_shapes=True)
RP_conv_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

"""--------SAE----------"""
centroids_input = Input((N_CLASS, LATENT_VARIABLE_DIM))
classification = classification_layer([RP_conv_ae.get_layer('RP_conv_embedding').output, centroids_input],
                                      )

RP_conv_sae = Model(
    inputs=[RP_conv_ae.get_layer(index=0).input, centroids_input],
    outputs=[classification, RP_conv_ae.get_layer('RP_reconstruction').output])
RP_conv_sae.summary()
plot_model(RP_conv_sae, to_file='./results/RP_conv_sae.png', show_shapes=True)
RP_conv_sae.compile(loss=['kld', 'mse'], loss_weights=[1, 2], optimizer='adam', metrics=['accuracy', 'accuracy'])

"""-----------train---------------"""
tb = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 #                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True,  # 是否可视化梯度直方图
                 write_images=True,  # 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)

checkpoint = ModelCheckpoint('./results/RP_conv_sae_check_point.model', monitor='val_lambda_1_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_lambda_1_loss', patience=200, verbose=2)

def pretrain(epochs=1000):
    print('pretrain')
    checkpoint = ModelCheckpoint('./results/RP_ae_check_point.model', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')

    RP_conv_ae.fit(x_RP, x_RP, batch_size=2000, epochs=epochs, callbacks=[tb, checkpoint])
    RP_conv_ae.save('./results/RP_conv_ae.model')


def continue_pretrain(epochs=500):
    print('continue_pretrain')
    RP_conv_ae = load_model('./results/RP_conv_ae.model')
    RP_conv_ae.fit(x_RP, x_RP, batch_size=2000, epochs=epochs, callbacks=[tb])
    RP_conv_ae.save('./results/RP_conv_ae.model')

classifier_checkpoint = ModelCheckpoint('./results/RP_conv_sae_check_point.model', monitor='val_lambda_1_acc', verbose=1,
                                 save_best_only=True, mode='max')
def train_classifier(epochs=100):
    print('train_classifier...')
    RP_conv_ae = load_model('./results/RP_ae_check_point.model')
    hist = RP_conv_sae.fit([train_x_RP, train_x_centroids], [train_y, train_x_RP],
                           epochs=epochs,
                           batch_size=2000, shuffle=True,
                           validation_data=(
                               [test_x_RP, test_x_centroids], [test_y, test_x_RP]),
                           callbacks=[early_stopping, tb, classifier_checkpoint])
    RP_conv_sae.save('./results/RP_conv_sae.model')
    A = np.argmax(hist.history['val_lambda_1_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[A],
                                                                             np.max(hist.history['val_lambda_1_acc'])))


def continue_train_classifier(epochs=6000):
    print('continue_train_classifier...')
    RP_conv_sae = load_model('./results/RP_conv_sae.model', custom_objects={'student_t': student_t,
                                                                            'N_CLASS': N_CLASS})
    hist = RP_conv_sae.fit([train_x_RP, train_x_centroids], [train_y, train_x_RP],
                           epochs=epochs,
                           batch_size=2000, shuffle=True,
                           validation_data=(
                               [test_x_RP, test_x_centroids], [test_y, test_x_RP]),
                           callbacks=[early_stopping, tb, classifier_checkpoint])
    RP_conv_sae.save('./results/RP_conv_sae.model')
    A = np.argmax(hist.history['val_lambda_1_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[A],
                                                                             np.max(hist.history['val_lambda_1_acc'])))


# pretrain(5000)
continue_pretrain(5000)
train_classifier(5000)
# continue_train_classifier(5000)