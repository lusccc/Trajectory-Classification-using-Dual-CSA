from os.path import exists

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


import os

from PEDCC import get_centroids
from conv_ae import Conv_AE
from params import N_CLASS, LATENT_VARIABLE_DIM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def student_t(z, u, alpha=1.):
    """
student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
    :param z: shape=(n_samples, n_embedding)
    :param u: ground_truth_centroids
    :param alpha:
    :return:student's t-distribution, or soft labels for each sample. shape=(n_samples, n_class)
    """
    q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(z, axis=1) - u), axis=2) / alpha))
    q **= (alpha + 1.0) / 2.0
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    return q


# input: concat(embedding1,embedding2), ground_truth_centroids
# output: predicted q distribution
classification_layer = Lambda(lambda x: student_t(K.concatenate([x[0], x[1]], axis=0), x[2]),
                              output_shape=lambda x: (x[0][0], N_CLASS))


def load_data():
    test_n = 5000
    x_RP = np.load('./geolife/features_RP_mats.npy')[:]
    n_samples = x_RP.shape[0]
    x_centroids = get_centroids(n_samples)[:]  # shape(n, 10,48)
    x_Gram = np.load('./geolife/trjs_Gram_mats.npy')
    y = np.load('./geolife/segments_labels.npy')[:]
    train_x_RP, test_x_RP, train_y, test_y, \
    train_x_centroids, test_x_centroids, train_y, test_y, \
    train_x_Gram, test_x_Gram, train_y, test_y = train_test_split(x_RP, y,
                                                                  x_centroids, y,
                                                                  x_Gram, y,
                                                                  test_size=0.20, random_state=7,
                                                                  shuffle=True)
    train_y = to_categorical(train_y, num_classes=N_CLASS)
    test_y = to_categorical(test_y, num_classes=N_CLASS)
    return x_RP, x_Gram, \
           train_x_RP, test_x_RP, train_y, test_y, \
           train_x_centroids, test_x_centroids, train_y, test_y, \
           train_x_Gram, test_x_Gram, train_y, test_y


x_RP, x_Gram, \
train_x_RP, test_x_RP, train_y, test_y, \
train_x_centroids, test_x_centroids, train_y, test_y, \
train_x_Gram, test_x_Gram, train_y, test_y = load_data()


def RP_CAE():
    """ -----RP_conv_ae------"""
    RP_mat_size = train_x_RP.shape[1]  # 40
    n_RP_features = train_x_RP.shape[3]
    RP_conv_ae = Conv_AE((RP_mat_size, RP_mat_size, n_RP_features), LATENT_VARIABLE_DIM, n_RP_features, 'RP')
    RP_conv_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return RP_conv_ae

def Gram_CAE():
    """ -----Gram_conv_ae------"""
    Gram_mat_size = train_x_Gram.shape[1]  # 40
    n_Gram_features = train_x_Gram.shape[3]
    Gram_conv_ae = Conv_AE((Gram_mat_size, Gram_mat_size, n_Gram_features), LATENT_VARIABLE_DIM, n_Gram_features,
                           'Gram')
    Gram_conv_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return Gram_conv_ae

def SMAE():
    """--------SAE----------"""
    centroids_input = Input((N_CLASS, LATENT_VARIABLE_DIM))
    classification = classification_layer(
        [RP_conv_ae.get_layer('RP_embedding').output, Gram_conv_ae.get_layer('Gram_embedding').output, centroids_input])

    smae = Model(
        inputs=[RP_conv_ae.get_layer(index=0).input, centroids_input, Gram_conv_ae.get_layer(index=0).input],
        outputs=[RP_conv_ae.get_layer('RP_reconstruction').output, classification,
                 Gram_conv_ae.get_layer('Gram_reconstruction').output])
    smae.summary()
    plot_model(smae, to_file='./results/smae.png', show_shapes=True)
    smae.compile(loss=['mse', 'kld', 'mse'], loss_weights=[1, 1, 1], optimizer='adam',
             metrics=['accuracy', 'accuracy', 'accuracy'])
    return smae

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


def pretrain_RP(epochs=1000, batch_size=200):
    print('pretrain_RP')
    cp_path = './results/RP_conv_ae_check_point.model'
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    if exists(cp_path):
        load_model(cp_path)
    RP_conv_ae.fit(x_RP, x_RP, batch_size=batch_size, epochs=epochs, callbacks=[tb, cp])


def pretrain_Gram(epochs=1000, batch_size=200):
    print('pretrain_Gram')
    cp_path = './results/Gram_conv_ae_check_point.model'
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    if exists(cp_path):
        load_model(cp_path)
    Gram_conv_ae.fit(x_Gram, x_Gram, batch_size=batch_size, epochs=epochs, callbacks=[tb, cp])


def train_classifier(epochs=100, batch_size=200):
    print('train_classifier...')
    cp = ModelCheckpoint('./results/smae_check_point.model', monitor='val_lambda_1_acc',
                         verbose=1,
                         save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_lambda_1_loss', patience=200, verbose=2)
    RP_conv_ae = load_model('./results/RP_ae_check_point.model')
    Gram_conv_ae = load_model('./results/Gram_conv_ae_check_point.model')
    hist = smae.fit([train_x_RP, train_x_centroids, train_x_Gram], [train_x_RP, train_y, train_x_Gram],
                    epochs=epochs,
                    batch_size=batch_size, shuffle=True,
                    validation_data=(
                        [test_x_RP, test_x_centroids, test_x_Gram], [test_x_RP, test_y, test_x_Gram]),
                    callbacks=[early_stopping, tb, cp])
    smae.save('./results/smae.model')
    A = np.argmax(hist.history['val_lambda_1_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[A],
                                                                             np.max(hist.history['val_lambda_1_acc'])))

if __name__ == '__main__':
    epochs = 30
    batch_size = 600
    RP_conv_ae = RP_CAE()
    Gram_conv_ae = Gram_CAE()
    smae = SMAE()
    pretrain_RP(100, batch_size)
    pretrain_Gram(100, batch_size)
    train_classifier(1000, batch_size)
