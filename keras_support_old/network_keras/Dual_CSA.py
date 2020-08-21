import argparse
import math
import os
import pathlib
import time
from os.path import exists

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.keras import Input, Model, regularizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback
from tensorflow.python.keras.layers import concatenate, Lambda
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model

import dataset_factory
from keras_support.network_keras import CONV2D_AE
from keras_support.network_keras.TS_CONV2D_AE import CONV1D_AE
from params import *
from utils import visualizeData

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# make run on low memory machine
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def log(info):
    with open(os.path.join(results_path, 'log.txt'), 'a') as f:
        print('★ ', end='')
        print(info)
        print('★ ', end='', file=f)
        print(info, file=f)


def student_t(z, u, alpha=1.):
    """student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
    :param z: shape=(n_samples, n_embedding)
    :param u: ground_truth_centroids
    :param alpha:
    :return:student's t-distribution, or soft labels for each sample. shape=(n_samples, n_class)
    """
    tmp1= K.expand_dims(z, axis=1)
    tmp2=K.square(tmp1 - u)
    tmp3=K.sum(tmp2, axis=2)
    q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(z, axis=1) - u), axis=2) / alpha))
    q **= (alpha + 1.0) / 2.0
    tmp4 = K.transpose(q)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    return q


# input: embedding, ground_truth_centroids
# output: predicted q distribution, shape=(n_samples, n_class)
classification_layer = Lambda(lambda x: student_t(x[0], x[1]), output_shape=lambda x: (x[0][0], N_CLASS),
                              name='cls_pedcc'
                              )


def RP_Conv2D_AE():
    """ -----RP_conv_ae------"""
    RP_mat_size = data_set.multi_channel_RP_mat_test.shape[1]  # 40
    n_features = data_set.multi_channel_RP_mat_test.shape[3]
    # our network_keras will maxpooling 3 times with size 2, i.e., 8 times smaller.
    # here we check if the size fit to the network_keras, or we apply zero padding
    zero_padding = 0
    if RP_mat_size % 8 != 0:
        zero_padding = int((int(RP_mat_size / 8) * 8 + 8 - RP_mat_size) / 2)  # divided by 2, because pad to width and height at same time

    with strategy.scope():
        RP_conv_ae = CONV2D_AE((RP_mat_size, RP_mat_size, n_features), RP_emb_dim, n_features, 'RP', results_path,
                               zero_padding)
        RP_conv_ae.compile(optimizer='adam', loss='mse', metrics={'RP_reconstruction': 'mse'})
    return RP_conv_ae


def ts_Conv1d_AE():
    n_features = data_set.multi_feature_segment_test.shape[3]
    seg_size = data_set.multi_feature_segment_test.shape[2]
    # our network_keras will maxpooling 3 times with size 2, i.e., 8 times smaller.
    # here we check if the size fit to the network_keras, or we apply zero padding
    zero_padding = 0
    if seg_size % 8 != 0:
        zero_padding = int((int(
            seg_size / 8) * 8 + 8 - seg_size) / 2)  # divided by 2, because pad to width and height at same time

    with strategy.scope():
        ts_conv_ae = CONV1D_AE((1, seg_size, n_features), ts_emb_dim, n_features, 'ts', results_path,
                               zero_padding)
        ts_conv_ae.compile(optimizer='adam', loss='mse', metrics={'ts_reconstruction': 'mse'})
    return ts_conv_ae


def dual_CSA():
    """--------SAE----------"""
    concat_embedding = concatenate([RP_conv2d_ae.get_layer('RP_embedding').output,
                                    ts_conv1d_ae.get_layer('ts_embedding').output])
    centroids_input = Input((N_CLASS, TOTAL_EMB_DIM))
    classification = classification_layer([concat_embedding, centroids_input])
    # dense_out = Dense(input_dim=N_CLASS, units=N_CLASS, name='dense_out')(classification)
    with strategy.scope():
        dual_csa = Model(
            inputs=[RP_conv2d_ae.get_layer(index=0).input, centroids_input, ts_conv1d_ae.get_layer(index=0).input],
            outputs=[RP_conv2d_ae.get_layer('RP_reconstruction').output, classification,
                     ts_conv1d_ae.get_layer('ts_reconstruction').output])
        dual_csa.summary()
        plot_model(dual_csa, to_file=os.path.join(results_path, 'dual_csa.png'), show_shapes=True)

        dual_encoder = Model(
            inputs=[RP_conv2d_ae.get_layer(index=0).input, centroids_input, ts_conv1d_ae.get_layer(index=0).input],
            outputs=[concat_embedding]
        )
        regularizer = regularizers.l2(0.001)

        for layer in dual_csa.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    log(f'apply reg for {layer.name}')
                    setattr(layer, attr, regularizer)
        # dual_encoder.summary()
        plot_model(dual_encoder, to_file=os.path.join(results_path, 'dual_encoder.png'), show_shapes=True)
        dual_csa.compile(loss=['mse', 'kld', 'mse'], loss_weights=[var_alpha, var_beta, var_gamma], optimizer='adam',
                         metrics={'cls_pedcc': 'accuracy', 'RP_reconstruction': 'mse', 'ts_reconstruction': 'mse'}
                         )

    # optimizer = AdamW(learning_rate=lr_schedule(0), weight_decay=wd_schedule(0))

    return dual_csa, dual_encoder


"""-----------train---------------"""


def pretrain_RP(epochs=1000, batch_size=200, patience=10):
    """
    pretrain is unsupervised, all data could use to train
    """
    log('pretrain_RP')
    cp_path = os.path.join(results_path, 'RP_conv_ae_check_point.model')
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    csv_logger = CSVLogger(os.path.join(results_path, 'RP_log.csv'), append=True, separator=';')
    early_stopping = EarlyStopping(monitor='loss', patience=patience, verbose=2)

    RP_conv_ae_ = RP_conv2d_ae
    if exists(cp_path):
        with strategy.scope():
            RP_conv_ae_ = load_model(cp_path)

    for i in range(n_trainset_split_parts):
        log(f'train RP on trainset part:{i}')
        RP_train_part = data_set.get_RP_train_part(i)
        RP_conv_ae_.fit(x=RP_train_part, y=RP_train_part, batch_size=batch_size, epochs=epochs,
                        use_multiprocessing=False,
                        shuffle=True, callbacks=[cp, csv_logger, early_stopping])


def pretrain_ts(epochs=1000, batch_size=200, patience=10):
    """
    pretrain is unsupervised, all data could use to train
    """
    log('pretrain_ts')
    cp_path = os.path.join(results_path, 'ts_conv_ae_check_point.model')
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    csv_logger = CSVLogger(os.path.join(results_path, 'ts_log.csv'), append=True, separator=';')
    early_stopping = EarlyStopping(monitor='loss', patience=patience, verbose=2)

    ts_conv1d_ae_ = ts_conv1d_ae
    if exists(cp_path):
        with strategy.scope():
            ts_conv1d_ae_ = load_model(cp_path)
    for i in range(n_trainset_split_parts):
        log(f'train ts on trainset part:{i}')
        ts_train_part = data_set.get_multi_feature_seg_train_part(n_trainset_split_part_size, i)
        ts_conv1d_ae_.fit(x=ts_train_part, y=ts_train_part, batch_size=batch_size, epochs=epochs,
                          use_multiprocessing=False,
                          shuffle=True, callbacks=[cp, csv_logger, early_stopping])


def train_classifier(pretrained=True, epochs=100, batch_size=200, patience=30):
    log('train_classifier...')
    cp_path = os.path.join(results_path, 'sae_check_point.model')
    # val_cls_pedcc_accuracy
    cp = ModelCheckpoint(cp_path, monitor='val_cls_pedcc_accuracy',
                         verbose=1,
                         save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_cls_pedcc_accuracy', patience=patience, verbose=2)
    # reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
    #                               factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    visulazation_callback = SAE_embedding_visualization_callback(os.path.join(results_path, 'sae_cp_{epoch}.h5'))
    # if pretrained:
    #     with strategy.scope():
    #         log('loading trained dual ae...')
    #         load_model(os.path.join(results_path, 'RP_conv_ae_check_point.model'))
    #         load_model(os.path.join(results_path, 'ts_conv_ae_check_point.model'))
    if exists(cp_path):
        with strategy.scope():
            log('load sae_check_point.model')
            load_model(cp_path)
    csv_logger = CSVLogger(os.path.join(results_path, 'csa_log.csv'), append=True, separator=';')

    for i in range(n_trainset_split_parts):
        log(f'train classifier on train set part:{i}')
        RP_train_part = data_set.get_RP_train_part(i)
        centroid_train_part = data_set.get_centroid_train_part(n_trainset_split_part_size, i)
        ts_train_part = data_set.get_multi_feature_seg_train_part(n_trainset_split_part_size, i)
        label_train_part = data_set.get_label_train_part(n_trainset_split_part_size, i)
        hist = dual_csa.fit(
            x=[RP_train_part, centroid_train_part, ts_train_part],
            y=[RP_train_part, label_train_part, ts_train_part],
            epochs=epochs,
            batch_size=batch_size, shuffle=True,
            validation_data=(
            [data_set.multi_channel_RP_mat_test, data_set.centroid_test, data_set.multi_feature_segment_test],
            [data_set.multi_channel_RP_mat_test, data_set.label_test, data_set.multi_feature_segment_test]),
            callbacks=[early_stopping, csv_logger,
                       # Dynamic_loss_weights_callback(var_alpha, var_beta, var_gamma),
                       # visulazation_callback,
                       cp
                       ],
            use_multiprocessing=False
        )
    #
    score = np.argmax(hist.history['val_cls_pedcc_accuracy'])
    log('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[score],
                                                                           np.max(
                                                                               hist.history['val_cls_pedcc_accuracy'])))


# https://github.com/keras-team/keras/issues/2595
class Dynamic_loss_weights_callback(Callback):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hist_RP_loss = []
        self.hist_cls_loss = []
        self.hist_ts_loss = []

        self.best_acc = 0
        self.acc_not_improving_step = 0

    def on_epoch_end(self, epoch, logs):
        acc = logs['val_cls_pedcc_accuracy']
        if acc > self.best_acc:
            self.acc_not_improving_step = 0
            self.best_acc = acc
            log(f'best_acc:{self.best_acc}, not_improving_step:{self.acc_not_improving_step}')
        else:
            self.acc_not_improving_step += 1
            log(f'best_acc:{self.best_acc}, not_improving_step:{self.acc_not_improving_step}')

        # loss_weights = [self.alpha, self.beta, self.gamma]
        # self.hist_RP_loss.append(logs['RP_reconstruction_loss'])
        # self.hist_cls_loss.append(logs['cls_pedcc_loss'])
        # self.hist_ts_loss.append(logs['ts_reconstruction_loss'])
        # if  self.acc_not_improving_step > 3 and self.acc_not_improving_step % 3 ==0:
        #     rp_rate = np.abs(np.mean(np.diff(self.hist_RP_loss[-3:])))
        #     cls_rate = np.abs(np.mean(np.diff(self.hist_cls_loss[-3:])))
        #     ts_rate = np.abs(np.mean(np.diff(self.hist_cls_loss[-3:])))
        #     log(f'avg loss chage rate:{rp_rate} {cls_rate} {ts_rate}')
        #
        #     max_idx = np.argmax([rp_rate, cls_rate, ts_rate])
        #     min_idx = np.argmin([rp_rate, cls_rate, ts_rate])
        #     K.set_value(loss_weights[max_idx], K.get_value(loss_weights[max_idx])-.05)
        #     K.set_value(loss_weights[min_idx], K.get_value(loss_weights[min_idx])+.1)

        log(
            f'loss weights:{K.get_value(self.alpha)}, {K.get_value(self.beta)}, {K.get_value(self.gamma)}')


class SAE_embedding_visualization_callback(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(SAE_embedding_visualization_callback, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 4 == 0:
            embedding = dual_encoder.predict([data_set.multi_channel_RP_mat_test, data_set.centroid_test, data_set.multi_feature_segment_test])
            y_true = np.argmax(data_set.label_test, axis=1)
            visualizeData(embedding, y_true, N_CLASS,
                          os.path.join(results_path, 'visualization', 'sae_embedding_epoch{}.png'.format(epoch)))


def show_confusion_matrix():
    sae = load_model(os.path.join(results_path, 'sae_check_point.model'),
                     custom_objects={'student_t': student_t, 'N_CLASS': N_CLASS})
    pred = sae.predict([data_set.multi_channel_RP_mat_test, data_set.centroid_test, data_set.multi_feature_segment_test])
    y_pred = np.argmax(pred[1], axis=1)
    y_true = np.argmax(data_set.label_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=modes_to_use)
    log(cm)
    with open(os.path.join(results_path, 'confusion_matrix.txt'), 'w') as f:
        print(cm, file=f)
    # f, ax = plt.subplots()
    # sns.heatmap(cm, annot=True, ax=ax)
    # ax.set_title('confusion matrix')
    # ax.set_xlabel('predict')
    # ax.set_ylabel('true')
    # plt.show()

    re = classification_report(y_true, y_pred, target_names=['walk', 'bike', 'bus', 'driving', 'train/subway'],
                               digits=5)
    log(re)
    with open(os.path.join(results_path, 'classification_report.txt'), 'w') as f:
        print(re, file=f)


def visualize_sae_embedding():
    sae = load_model(os.path.join(results_path, 'sae_check_point.model'),
                     custom_objects={'student_t': student_t, 'N_CLASS': N_CLASS})
    embedding = dual_encoder.predict([data_set.multi_channel_RP_mat_test, data_set.centroid_test, data_set.multi_feature_segment_test])
    y_true = np.argmax(data_set.label_test, axis=1)
    visualizeData(embedding, y_true, N_CLASS, os.path.join(results_path, 'visualization', 'best.png'))


def visualize_dual_ae_embedding():
    load_model(os.path.join(results_path, 'RP_conv_ae_check_point.model'))
    load_model(os.path.join(results_path, 'ts_conv_ae_check_point.model'))
    embedding = dual_encoder.predict([data_set.multi_channel_RP_mat_test, data_set.centroid_test, data_set.multi_feature_segment_test])
    y_true = np.argmax(data_set.label_test, axis=1)
    visualizeData(embedding, y_true, N_CLASS, os.path.join(results_path, 'visualization', 'dual_ae_embedding.png'))


def visualize_centroids():
    visualizeData(data_set.centroid_test[0], modes_to_use, N_CLASS,
                  os.path.join(results_path, 'visualization', 'centroids_visualization.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSAE')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--results_path', default='./results/default', type=str)
    parser.add_argument('--alpha', default=ALPHA, type=float)
    parser.add_argument('--beta', default=BETA, type=float)
    parser.add_argument('--gamma', default=GAMMA, type=float)
    parser.add_argument('--no_pre', default=False, type=bool)
    parser.add_argument('--no_joint', default=False, type=bool)
    parser.add_argument('--epoch1', default=3000, type=int)
    parser.add_argument('--epoch2', default=3000, type=int)
    parser.add_argument('--RP_emb_dim', type=int)
    parser.add_argument('--ts_emb_dim', type=int)
    parser.add_argument('--n_trainset_split_parts', type=int, default=1)

    args = parser.parse_args()
    results_path = args.results_path
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    no_pretrain = args.no_pre
    no_joint_train = args.no_joint
    epoch1 = args.epoch1
    epoch2 = args.epoch2
    RP_emb_dim = args.RP_emb_dim
    ts_emb_dim = args.ts_emb_dim
    TOTAL_EMB_DIM = RP_emb_dim + ts_emb_dim
    dataset = args.dataset
    n_trainset_split_parts = args.n_trainset_split_parts

    pathlib.Path(os.path.join(results_path, 'visualization')).mkdir(parents=True, exist_ok=True)

    log(f'dataset_name:{dataset}, results_path:{results_path} , loss weight:{alpha},{beta},{gamma},'
        f'RP_emb_dim:{RP_emb_dim}, ts_emb_dim:{ts_emb_dim}, no_pretrain:{no_pretrain}, no_joint_train:{no_pretrain}, '
        f'n_trainset_split_parts:{n_trainset_split_parts}')

    strategy = tf.distribute.MirroredStrategy()

    batch_size_per_replica = 1
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    log('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    log(f'batch size:{batch_size}')

    data_set = dataset_factory.Dataset(dataset, n_trainset_split_parts)
    n_train_samples = data_set.n_samples_train
    n_test_samples = data_set.n_samples_test
    n_trainset_split_part_size = math.ceil(n_train_samples / n_trainset_split_parts)

    # """ note: each autoencoder has same embedding,
    #  embedding will be concated to match EMB_DIM,
    # i.e. centroids has dim EMB_DIM"""
    # EMB_DIM = x_centroids_train.shape[2]
    # log(f'EMB_DIM:{EMB_DIM}')
    # n_ae = 2  # num of ae
    # each_embedding_dim = int(EMB_DIM / n_ae)

    if no_joint_train:
        # var_alpha, var_beta, var_gamma = K.variable(0), K.variable(1), K.variable(0)
        var_alpha, var_beta, var_gamma = 0, 1, 0
    else:
        # var_alpha, var_beta, var_gamma = K.variable(alpha), K.variable(beta), K.variable(gamma)
        var_alpha, var_beta, var_gamma = alpha, beta, gamma
    # tb = TensorBoard(log_dir=os.path.join(results_path, 'tensorflow_logs'),
    #                  histogram_freq=0,
    #
    #                  write_graph=True,
    #                  write_grads=True,
    #                  write_images=True,
    #                  embeddings_freq=0,
    #                  embeddings_layer_names=None,
    #                  embeddings_metadata=None)

    # means only train classifier using default loss_weight
    # batch_size = 300
    if no_pretrain or no_joint_train:
        dual_csa, dual_encoder = dual_CSA()
        train_classifier(pretrained=False, epochs=3000, batch_size=batch_size, patience=30)
    else:
        t0 = time.time()
        RP_conv2d_ae = RP_Conv2D_AE()
        # pretrain_RP(epoch1, batch_size, patience=5)
        t1 = time.time()
        log('pretrain_RP Running time: %s Seconds' % (t1 - t0))
        ts_conv1d_ae = ts_Conv1d_AE()
        # pretrain_ts(epoch2, batch_size, patience=5)
        t2 = time.time()
        log('pretrain_ts Running time: %s Seconds' % (t2 - t1))
        dual_csa, dual_encoder = dual_CSA()
        # visualize_dual_ae_embedding()
        train_classifier(pretrained=True, epochs=3000, batch_size=batch_size, patience=30)
        t3 = time.time()
        log('train_classifier Running time: %s Seconds' % (t3 - t1))

    visualize_centroids()
    show_confusion_matrix()
    visualize_sae_embedding()
