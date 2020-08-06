import os
import pathlib
from os.path import exists

import matplotlib.pyplot as plt
import seaborn as sns
from backup.trajectory_features_and_segmentation import MAX_SEGMENT_SIZE
from dataset_generation import *
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import concatenate, Dense
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.utils import plot_model, multi_gpu_model
from sklearn.metrics import confusion_matrix, classification_report

from Geolife_trajectory_extraction import modes_to_use
from network.CONV2D_AE import CONV2D_AE
from network.TS_CONV2D_AE import TS_CONV1D_AE
from params import MULTI_GPU


def log(info):
    with open(os.path.join(results_path, 'log.txt'), 'a') as f:
        print(info)
        print(info, file=f)


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


def RP_Conv2D_AE():
    """ -----RP_conv_ae------"""
    RP_mat_size = x_RP_train.shape[1]  # 40
    n_features = x_RP_train.shape[3]
    RP_conv_ae = CONV2D_AE((RP_mat_size, RP_mat_size, n_features), each_embedding_dim, n_features, 'RP')
    if MULTI_GPU:
        RP_conv_ae = multi_gpu_model(RP_conv_ae, gpus=2)
    RP_conv_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return RP_conv_ae


def ts_Conv2d_AE():
    n_features = x_features_series_train.shape[3]
    ts_conv1d_ae = TS_CONV1D_AE((1, MAX_SEGMENT_SIZE, n_features), each_embedding_dim, n_features, 'ts')
    if MULTI_GPU:
        ts_conv1d_ae = multi_gpu_model(ts_conv1d_ae, gpus=2)
    ts_conv1d_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return ts_conv1d_ae


def dual_Softmax_AE():
    concat_embedding = concatenate([RP_conv2d_ae.get_layer('RP_embedding').output,
                                    ts_conv2d_ae.get_layer('ts_embedding').output])
    softmax_classifier = Dense(N_CLASS, activation='softmax')(concat_embedding)

    d_sae = Model(
        inputs=[RP_conv2d_ae.get_layer(index=0).input, ts_conv2d_ae.get_layer(index=0).input],
        outputs=[RP_conv2d_ae.get_layer('RP_reconstruction').output, softmax_classifier,
                 ts_conv2d_ae.get_layer('ts_reconstruction').output])
    d_sae.summary()
    plot_model(d_sae, to_file=os.path.join(results_path, 'dual_encoder.png'), show_shapes=True)
    if MULTI_GPU:
        d_sae = multi_gpu_model(d_sae, gpus=2)
    d_sae.compile(loss=['mse', 'categorical_crossentropy', 'mse'], loss_weights=loss_weights, optimizer='adam',
                  metrics=['accuracy', categorical_accuracy, 'accuracy'])
    return d_sae


"""-----------train---------------"""


def pretrain_RP(epochs=1000, batch_size=200):
    """
    pretrain is unsupervised, all data could use to train
    """
    print('pretrain_RP')
    cp_path = os.path.join(results_path, 'RP_conv_ae_check_point.model')
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    RP_conv_ae_ = RP_conv2d_ae
    if exists(cp_path):
        RP_conv_ae_ = load_model(cp_path)
    RP_conv_ae_.fit(x_RP_train, x_RP_train, batch_size=batch_size, epochs=epochs, callbacks=[tb, cp])


def pretrain_ts(epochs=1000, batch_size=200):
    """
    pretrain is unsupervised, all data could use to train
    """
    print('pretrain_ts')
    cp_path = os.path.join(results_path, 'ts_conv_ae_check_point.model')
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    ts_conv1d_ae_ = ts_conv2d_ae
    if exists(cp_path):
        ts_conv1d_ae_ = load_model(cp_path)
    ts_conv1d_ae_.fit(x_features_series_train, x_features_series_train, batch_size=batch_size, epochs=epochs,
                      callbacks=[tb, cp])


def train_classifier(pretrained=True, epochs=100, batch_size=200):
    log('train_classifier...')
    cp = ModelCheckpoint(os.path.join(results_path, 'sae_check_point.model'), monitor='val_dense_3_acc',
                         verbose=1,
                         save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_dense_3_loss', patience=patience, verbose=2)
    if pretrained:
        log('loading trained dual ae...')
        load_model(os.path.join(results_path, 'RP_conv_ae_check_point.model'))
        load_model(os.path.join(results_path, 'ts_conv_ae_check_point.model'))
    hist = dual_sae.fit([x_RP_train, x_features_series_train],
                        [x_RP_train, y_train, x_features_series_train],
                        epochs=epochs,
                        batch_size=batch_size, shuffle=True,
                        validation_data=(
                            [x_RP_test, x_features_series_test],
                            [x_RP_test, y_test, x_features_series_test]),
                        callbacks=[early_stopping, tb, cp]
                        )
    #
    score = np.argmax(hist.history['val_dense_3_acc'])
    log('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[score],
                                                                           np.max(hist.history['val_dense_3_acc'])))


def show_confusion_matrix():
    sae = load_model(os.path.join(results_path, 'sae_check_point.model'),
                     custom_objects={'student_t': student_t, 'N_CLASS': N_CLASS})
    pred = sae.predict([x_RP_test, x_features_series_test])
    y_pred = np.argmax(pred[1], axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    log(cm)
    with open(os.path.join(results_path, 'confusion_matrix.txt'), 'w') as f:
        print(cm, file=f)
    f, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax)  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    # plt.show()

    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    log(re)
    with open(os.path.join(results_path, 'classification_report.txt'), 'w') as f:
        print(re, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSA_soft')
    parser.add_argument('--results_path', default='results/default', type=str)
    parser.add_argument('--alpha', default=ALPHA, type=float)
    parser.add_argument('--beta', default=BETA, type=float)
    parser.add_argument('--gamma', default=GAMMA, type=float)
    parser.add_argument('--no_pre', default=False, type=bool)
    parser.add_argument('--no_joint', default=False, type=bool)
    args = parser.parse_args()
    results_path = args.results_path
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    no_pretrain = args.no_pre
    no_joint_train = args.no_joint
    log('results_path:{} , loss weight:{},{},{}, no_pretrain:{}, no_joint_train:{}'.format(results_path, alpha, beta,
                                                                                           gamma,
                                                                                           no_pretrain, no_joint_train))
    pathlib.Path(os.path.join(results_path, 'visualization')).mkdir(parents=True, exist_ok=True)

    x_RP_train = Dataset.x_RP_train
    x_RP_test = Dataset.x_RP_test
    x_features_series_train = Dataset.x_features_series_train
    x_features_series_test = Dataset.x_features_series_test
    x_centroids_train = Dataset.x_centroids_train
    x_centroids_test = Dataset.x_centroids_test
    y_train = Dataset.y_train
    y_test = Dataset.y_test

    EMB_DIM = x_centroids_train.shape[2]

    epochs = 30
    batch_size = 600
    """ note: each autoencoder has same embedding,
     embedding will be concated to match EMB_DIM, 
    i.e. centroids has dim EMB_DIM"""
    n_ae = 2  # num of ae
    each_embedding_dim = int(EMB_DIM / n_ae)
    patience = 35

    if no_joint_train:
        loss_weights = [0, 1, 0]
    else:
        loss_weights = [alpha, beta, gamma]
    RP_conv2d_ae = RP_Conv2D_AE()
    ts_conv2d_ae = ts_Conv2d_AE()
    dual_sae = dual_Softmax_AE()
    tb = TensorBoard(log_dir=os.path.join(results_path, 'tensorflow_logs'),  # log 目录
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     #                  batch_size=32,     # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=True,  # 是否可视化梯度直方图
                     write_images=True,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    # means only train classifier using default loss_weight
    if no_pretrain or no_joint_train:
        train_classifier(pretrained=False, epochs=3000, batch_size=batch_size)
    else:
        pass
        # pretrain_RP(100, batch_size)
        # pretrain_ts(300, batch_size)
        # train_classifier(pretrained=True, epochs=3000, batch_size=batch_size)

    show_confusion_matrix()
