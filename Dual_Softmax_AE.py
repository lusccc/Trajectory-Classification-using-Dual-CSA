from os.path import exists
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Lambda, \
    concatenate, Dense
from keras.layers import Input
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.utils import plot_model, multi_gpu_model

import os

from CONV2D_AE import CONV2D_AE
from dataset import load_data
from params import N_CLASS, TOTAL_EMBEDDING_DIM, MULTI_GPU
from TS_CONV2D_AE import TS_CONV2D_AE
from trajectory_extraction import modes_to_use
from trajectory_features_and_segmentation import MAX_SEGMENT_SIZE

from dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def RP_Conv2D_AE():
    """ -----RP_conv_ae------"""
    RP_mat_size = x_RP_clean_mf_train.shape[1]  # 40
    n_features = x_RP_clean_mf_train.shape[3]
    RP_conv_ae = CONV2D_AE((RP_mat_size, RP_mat_size, n_features), each_embedding_dim, n_features, 'RP')
    if MULTI_GPU:
        RP_conv_ae = multi_gpu_model(RP_conv_ae, gpus=2)
    RP_conv_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return RP_conv_ae


def ts_Conv2d_AE():
    n_features = x_trj_seg_clean_of_train.shape[3]
    ts_conv1d_ae = TS_CONV2D_AE((1, MAX_SEGMENT_SIZE, n_features), each_embedding_dim, n_features, 'spatial')
    if MULTI_GPU:
        ts_conv1d_ae = multi_gpu_model(ts_conv1d_ae, gpus=2)
    ts_conv1d_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return ts_conv1d_ae


def dual_Softmax_AE():
    concat_embedding = concatenate([RP_conv2d_ae.get_layer('RP_embedding').output,
                                    ts_conv2d_ae.get_layer('spatial_embedding').output])
    softmax_classifier = Dense(N_CLASS, activation='softmax')(concat_embedding)

    d_sae = Model(
        inputs=[RP_conv2d_ae.get_layer(index=0).input, ts_conv2d_ae.get_layer(index=0).input],
        outputs=[RP_conv2d_ae.get_layer('RP_reconstruction').output, softmax_classifier,
                 ts_conv2d_ae.get_layer('spatial_reconstruction').output])
    d_sae.summary()
    plot_model(d_sae, to_file='./comparison_results/dual_sae.png', show_shapes=True)
    if MULTI_GPU:
        d_sae = multi_gpu_model(d_sae, gpus=2)
    d_sae.compile(loss=['mse', 'categorical_crossentropy', 'mse'], loss_weights=loss_weights, optimizer='adam',
                  metrics=['accuracy', categorical_accuracy, 'accuracy'])
    return d_sae


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
    cp_path = './comparison_results/RP_conv_ae_check_point.model'
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    RP_conv_ae_ = RP_conv2d_ae
    if exists(cp_path):
        RP_conv_ae_ = load_model(cp_path)
    RP_conv_ae_.fit(x_RP_clean_mf_train, x_RP_clean_mf_train, batch_size=batch_size, epochs=epochs, callbacks=[tb, cp])


def pretrain_ts(epochs=1000, batch_size=200):
    """
    pretrain is unsupervised, all data could use to train
    """
    print('pretrain_ts')
    cp_path = './comparison_results/ts_conv_ae_check_point.model'
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    ts_conv1d_ae_ = ts_conv2d_ae
    if exists(cp_path):
        ts_conv1d_ae_ = load_model(cp_path)
    ts_conv1d_ae_.fit(x_trj_seg_clean_of_train, x_trj_seg_clean_of_train, batch_size=batch_size, epochs=epochs,
                      callbacks=[tb, cp])


def train_classifier(epochs=100, batch_size=200):
    print('train_classifier...')
    cp = ModelCheckpoint('./comparison_results/sae_check_point.model', monitor='val_dense_3_acc',
                         verbose=1,
                         save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_dense_3_loss', patience=patience, verbose=2)
    RP_conv_ae = load_model('./comparison_results/RP_conv_ae_check_point.model')
    spatial_conv1d_ae_ = load_model('./comparison_results/ts_conv_ae_check_point.model')
    hist = dual_sae.fit([x_RP_clean_mf_train, x_trj_seg_clean_of_train],
                        [x_RP_clean_mf_train, y_train, x_trj_seg_clean_of_train],
                        epochs=epochs,
                        batch_size=batch_size, shuffle=True,
                        validation_data=(
                            [x_RP_clean_mf_test, x_trj_seg_clean_of_test],
                            [x_RP_clean_mf_test, y_test, x_trj_seg_clean_of_test]),
                        callbacks=[early_stopping, tb, cp]
                        )
    #
    score = np.argmax(hist.history['val_dense_3_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[score],
                                                                             np.max(hist.history['val_dense_3_acc'])))


def show_confusion_matrix():
    sae = load_model('./comparison_results/sae_check_point.model', custom_objects={'N_CLASS': N_CLASS})
    pred = sae.predict([x_RP_clean_mf_test, x_trj_seg_clean_of_test])
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

    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    print(re)


if __name__ == '__main__':
    epochs = 30
    batch_size = 300
    """ note: each autoencoder has same embedding,
     embedding will be concated to match TOTAL_EMBEDDING_DIM, 
    aka. centroids has dim TOTAL_EMBEDDING_DIM"""
    n_ae = 2  # num of ae
    each_embedding_dim = int(TOTAL_EMBEDDING_DIM / n_ae)
    loss_weights = [1, 3, 1]
    patience = 35

    RP_conv2d_ae = RP_Conv2D_AE()
    ts_conv2d_ae = ts_Conv2d_AE()
    dual_sae = dual_Softmax_AE()
    pretrain_RP(100, batch_size)
    pretrain_ts(350, batch_size)
    train_classifier(3000, batch_size)
    show_confusion_matrix()
