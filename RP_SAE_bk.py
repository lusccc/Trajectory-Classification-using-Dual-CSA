from os.path import exists
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import numpy as np
from keras import backend as K, Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Conv2D, Flatten, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Lambda, \
    BatchNormalization, Activation, concatenate
from keras.layers import Input, Dense
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split

import os

from Conv2D_AE import Conv2D_AE
from params import N_CLASS, TOTAL_EMBEDDING_DIM
from trajectory_extraction import modes_to_use

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


# input: embedding, ground_truth_centroids
# output: predicted q distribution, shape=(n_samples, n_class)
classification_layer = Lambda(lambda x: student_t(x[0], x[1]), output_shape=lambda x: (x[0][0], N_CLASS))


def load_data():
    x_train_RP = np.load('./geolife/train_features_RP_mats.npy')
    x_test_RP = np.load('./geolife/test_features_RP_mats.npy')
    # test data cannot used to train ae, because it will tell the model test data information
    # x_RP = np.concatenate([x_train_RP, x_test_RP], axis=0)

    x_train_centroids = np.load('./geolife/train_centroids.npy')
    x_test_centroids = np.load('./geolife/test_centroids.npy')

    y_train = np.load('./geolife/train_segments_labels.npy')
    y_test = np.load('./geolife/test_segments_labels.npy')

    y_train = to_categorical(y_train, num_classes=N_CLASS)
    y_test = to_categorical(y_test, num_classes=N_CLASS)
    return x_train_RP, x_test_RP, x_train_centroids, x_test_centroids, y_train, y_test


def RP_CAE():
    """ -----RP_conv_ae------"""
    RP_mat_size = x_train_RP.shape[1]  # 40
    n_RP_features = x_train_RP.shape[3]
    RP_conv_ae = Conv2D_AE((RP_mat_size, RP_mat_size, n_RP_features), each_embedding_dim, n_RP_features, 'RP')
    RP_conv_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return RP_conv_ae


def SAE():
    """--------SAE----------"""
    centroids_input = Input((N_CLASS, TOTAL_EMBEDDING_DIM))
    classification = classification_layer([RP_conv_ae.get_layer('RP_embedding').output, centroids_input])
    # classification = Dense(N_CLASS, activation='softmax')(classification)

    sae = Model(
        inputs=[RP_conv_ae.get_layer(index=0).input, centroids_input],
        outputs=[RP_conv_ae.get_layer('RP_reconstruction').output, classification])
    sae.summary()
    plot_model(sae, to_file='./results/sae.png', show_shapes=True)
    sae.compile(loss=['mse', 'kld'], loss_weights=[1, 1], optimizer='adam',
                metrics=['accuracy', categorical_accuracy])
    return sae


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
    """
    pretrain is unsupervised, all data could use to train
    """
    print('pretrain_RP')
    cp_path = './results/RP_conv_ae_check_point.model'
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    RP_conv_ae_ = RP_conv_ae
    if exists(cp_path):
        RP_conv_ae_ = load_model(cp_path)
    RP_conv_ae_.fit(x_train_RP, x_train_RP, batch_size=batch_size, epochs=epochs, callbacks=[tb, cp])


def train_classifier(epochs=100, batch_size=200):
    print('train_classifier...')
    cp = ModelCheckpoint('./results/sae_check_point.model', monitor='val_lambda_1_acc',
                         verbose=1,
                         save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_lambda_1_loss', patience=100, verbose=2)
    RP_conv_ae = load_model('./results/RP_conv_ae_check_point.model')
    hist = sae.fit([x_train_RP, x_train_centroids], [x_train_RP, y_train],
                   epochs=epochs,
                   batch_size=batch_size, shuffle=True,
                   validation_data=(
                       [x_test_RP, x_test_centroids], [x_test_RP, y_test]),
                   callbacks=[early_stopping, tb, cp]
                   )
    #
    sae.save('./results/sae.model')
    A = np.argmax(hist.history['val_lambda_1_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[A],
                                                                             np.max(hist.history['val_lambda_1_acc'])))


def show_confusion_matrix():
    sae = load_model('./results/sae_check_point.model', custom_objects={'student_t': student_t, 'N_CLASS': N_CLASS})
    pred = sae.predict([x_test_RP, x_test_centroids])
    y_pred = np.argmax(pred[1], axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    print(cm)
    f, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax)  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()

    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'])
    print(re)


if __name__ == '__main__':
    x_train_RP, x_test_RP, x_train_centroids, x_test_centroids, y_train, y_test = load_data()

    epochs = 30
    batch_size = 800
    """ note: each autoencoder has same embedding,
     embedding will be concated to match TOTAL_EMBEDDING_DIM, 
    aka. centroids has dim TOTAL_EMBEDDING_DIM"""
    n_ae = 1
    each_embedding_dim = int(TOTAL_EMBEDDING_DIM / n_ae)

    RP_conv_ae = RP_CAE()
    sae = SAE()
    pretrain_RP(100, batch_size)
    train_classifier(5000, batch_size)
    show_confusion_matrix()
