import os
import pathlib
import time
from os.path import exists

import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from keras.engine.saving import load_model
from keras.layers import Input
from keras.layers import Lambda, \
    concatenate
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.utils import plot_model, multi_gpu_model
from sklearn.metrics import confusion_matrix, classification_report

import dataset
from CONV2D_AE import CONV2D_AE
from TS_CONV2D_AE import TS_CONV2D_AE
from dataset_generation import *
from utils import visualizeData
from params import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

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


# input: embedding, ground_truth_centroids
# output: predicted q distribution, shape=(n_samples, n_class)
classification_layer = Lambda(lambda x: student_t(x[0], x[1]), output_shape=lambda x: (x[0][0], N_CLASS))


def RP_Conv2D_AE():
    """ -----RP_conv_ae------"""
    RP_mat_size = x_RP_train.shape[1]  # 40
    n_features = x_RP_train.shape[3]
    RP_conv_ae = CONV2D_AE((RP_mat_size, RP_mat_size, n_features), each_embedding_dim, n_features, 'RP', results_path)
    if MULTI_GPU:
        RP_conv_ae = multi_gpu_model(RP_conv_ae, gpus=2)
    RP_conv_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return RP_conv_ae


def ts_Conv2d_AE():
    n_features = x_features_series_train.shape[3]
    ts_conv_ae = TS_CONV2D_AE((1, MAX_SEGMENT_SIZE, n_features), each_embedding_dim, n_features, 'ts', results_path)
    if MULTI_GPU:
        ts_conv_ae = multi_gpu_model(ts_conv_ae, gpus=2)
    ts_conv_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return ts_conv_ae


def dual_SAE():
    """--------SAE----------"""
    concat_embedding = concatenate([RP_conv2d_ae.get_layer('RP_embedding').output,
                                    ts_conv2d_ae.get_layer('ts_embedding').output])
    centroids_input = Input((N_CLASS, EMB_DIM))
    classification = classification_layer([concat_embedding, centroids_input])

    dual_sae = Model(
        inputs=[RP_conv2d_ae.get_layer(index=0).input, centroids_input, ts_conv2d_ae.get_layer(index=0).input],
        outputs=[RP_conv2d_ae.get_layer('RP_reconstruction').output, classification,
                 ts_conv2d_ae.get_layer('ts_reconstruction').output])
    # dual_sae.summary()
    plot_model(dual_sae, to_file=os.path.join(results_path, 'dual_sae.png'), show_shapes=True)

    dual_encoder = Model(
        inputs=[RP_conv2d_ae.get_layer(index=0).input, centroids_input, ts_conv2d_ae.get_layer(index=0).input],
        outputs=[concat_embedding]
    )
    # dual_encoder.summary()
    plot_model(dual_encoder, to_file=os.path.join(results_path, 'dual_encoder.png'), show_shapes=True)

    if MULTI_GPU:
        dual_sae = multi_gpu_model(dual_sae, gpus=2)
    dual_sae.compile(loss=['mse', 'kld', 'mse'], loss_weights=loss_weights, optimizer='adam',
                     metrics=['accuracy', categorical_accuracy, 'accuracy'])
    return dual_sae, dual_encoder


"""-----------train---------------"""


def pretrain_RP(epochs=1000, batch_size=200):
    """
    pretrain is unsupervised, all data could use to train
    """
    log('pretrain_RP')
    cp_path = os.path.join(results_path, 'RP_conv_ae_check_point.model')
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    RP_conv_ae_ = RP_conv2d_ae
    if exists(cp_path):
        RP_conv_ae_ = load_model(cp_path)
    RP_conv_ae_.fit(x_RP_train, x_RP_train, batch_size=batch_size, epochs=epochs, callbacks=[cp])


def pretrain_ts(epochs=1000, batch_size=200):
    """
    pretrain is unsupervised, all data could use to train
    """
    log('pretrain_ts')
    cp_path = os.path.join(results_path, 'ts_conv_ae_check_point.model')
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    ts_conv1d_ae_ = ts_conv2d_ae
    if exists(cp_path):
        ts_conv1d_ae_ = load_model(cp_path)
    ts_conv1d_ae_.fit(dataset.x_features_series_train, x_features_series_train, batch_size=batch_size,
                      epochs=epochs,
                      callbacks=[cp])


def train_classifier(pretrained=True, epochs=100, batch_size=200):
    log('train_classifier...')
    factor = 1. / np.cbrt(2)
    cp_path = os.path.join(results_path, 'sae_check_point.model')
    cp = ModelCheckpoint(cp_path, monitor='val_lambda_1_accuracy',
                         verbose=1,
                         save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_lambda_1_loss', patience=patience, verbose=2)
    # reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
    #                               factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    visulazation_callback = SAE_embedding_visualization_callback(os.path.join(results_path, 'sae_cp_{epoch}.h5'))
    if pretrained:
        log('loading trained dual ae...')
        load_model(os.path.join(results_path, 'RP_conv_ae_check_point.model'))
        load_model(os.path.join(results_path, 'ts_conv_ae_check_point.model'))
    if exists(cp_path):
        load_model(cp_path)
    hist = dual_sae.fit([x_RP_train, x_centroids_train, x_features_series_train],
                        [x_RP_train, y_train, x_features_series_train],
                        epochs=epochs,
                        batch_size=batch_size, shuffle=True,
                        validation_data=(
                            [x_RP_test, x_centroids_test, x_features_series_test],
                            [x_RP_test, y_test, x_features_series_test]),
                        callbacks=[early_stopping, cp, visulazation_callback, ],
                        )
    #
    score = np.argmax(hist.history['val_lambda_1_accuracy'])
    log('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[score],
                                                                           np.max(hist.history['val_lambda_1_accuracy'])))


class SAE_embedding_visualization_callback(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(SAE_embedding_visualization_callback, self).__init__(*args, **kwargs)

    # redefine the save so it only activates after 100 epochs
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 4 == 0:
            embedding = dual_encoder.predict([x_RP_test, x_centroids_test, x_features_series_test])
            y_true = np.argmax(y_test, axis=1)
            visualizeData(embedding, y_true, N_CLASS,
                          os.path.join(results_path, 'visualization', 'sae_embedding_epoch{}.png'.format(epoch)))


def show_confusion_matrix():
    sae = load_model(os.path.join(results_path, 'sae_check_point.model'),
                     custom_objects={'student_t': student_t, 'N_CLASS': N_CLASS})
    pred = sae.predict([x_RP_test, x_centroids_test, x_features_series_test])
    y_pred = np.argmax(pred[1], axis=1)
    y_true = np.argmax(y_test, axis=1)
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
    embedding = dual_encoder.predict([x_RP_test, x_centroids_test, x_features_series_test])
    y_true = np.argmax(y_test, axis=1)
    visualizeData(embedding, y_true, N_CLASS, os.path.join(results_path, 'visualization', 'best.png'))


def visualize_dual_ae_embedding():
    load_model(os.path.join(results_path, 'RP_conv_ae_check_point.model'))
    load_model(os.path.join(results_path, 'ts_conv_ae_check_point.model'))
    embedding = dual_encoder.predict([x_RP_test, x_centroids_test, x_features_series_test])
    y_true = np.argmax(y_test, axis=1)
    visualizeData(embedding, y_true, N_CLASS, os.path.join(results_path, 'visualization', 'dual_ae_embedding.png'))


def visualize_centroids():
    visualizeData(x_centroids_test[0], modes_to_use, N_CLASS,
                  os.path.join(results_path, 'visualization', 'centroids_visualization.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSAE')
    parser.add_argument('--results_path', default='results/default', type=str)
    parser.add_argument('--alpha', default=ALPHA, type=float)
    parser.add_argument('--beta', default=BETA, type=float)
    parser.add_argument('--gamma', default=GAMMA, type=float)
    parser.add_argument('--no_pre', default=False, type=bool)
    parser.add_argument('--no_joint', default=False, type=bool)
    parser.add_argument('--epoch1', default=100, type=int)
    parser.add_argument('--epoch2', default=350, type=int)

    args = parser.parse_args()
    results_path = args.results_path
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    no_pretrain = args.no_pre
    no_joint_train = args.no_joint
    epoch1 = args.epoch1
    epoch2 = args.epoch2
    pathlib.Path(os.path.join(results_path, 'visualization')).mkdir(parents=True, exist_ok=True)
    log('results_path:{} , loss weight:{},{},{}, no_pretrain:{}, no_joint_train:{}'.format(results_path, alpha, beta,
                                                                                           gamma,
                                                                                           no_pretrain, no_joint_train))

    x_RP_train = dataset.x_RP_train
    x_RP_test = dataset.x_RP_test
    x_features_series_train = dataset.x_features_series_train
    x_features_series_test = dataset.x_features_series_test
    x_centroids_train = dataset.x_centroids_train
    x_centroids_test = dataset.x_centroids_test
    y_train = dataset.y_train
    y_test = dataset.y_test

    EMB_DIM = x_centroids_train.shape[2]

    epochs = 30
    batch_size = 480
    """ note: each autoencoder has same embedding,
     embedding will be concated to match EMB_DIM, 
    i.e. centroids has dim EMB_DIM"""
    n_ae = 2  # num of ae
    each_embedding_dim = int(EMB_DIM / n_ae)
    patience = 100

    if no_joint_train:
        loss_weights = [0, 1, 0]
    else:
        loss_weights = [alpha, beta, gamma]
    RP_conv2d_ae = RP_Conv2D_AE()
    ts_conv2d_ae = ts_Conv2d_AE()
    dual_sae, dual_encoder = dual_SAE()
    tb = TensorBoard(log_dir=os.path.join(results_path, 'tensorflow_logs'),
                     histogram_freq=0,

                     write_graph=True,
                     write_grads=True,
                     write_images=True,
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    # means only train classifier using default loss_weight
    if no_pretrain or no_joint_train:
        train_classifier(pretrained=False, epochs=3000, batch_size=batch_size)
    else:
        t0 = time.time()
        # pretrain_RP(epoch1, batch_size)
        # t1 = time.time()
        # log('pretrain_RP Running time: %s Seconds' % (t1 - t0))
        # pretrain_ts(epoch2, batch_size)
        # t2 = time.time()
        # log('pretrain_ts Running time: %s Seconds' % (t2 - t1))
        train_classifier(pretrained=True, epochs=3000, batch_size=batch_size)
        t3 = time.time()
        log('train_classifier Running time: %s Seconds' % (t3 - t0))

    visualize_centroids()
    visualize_dual_ae_embedding()
    show_confusion_matrix()
    visualize_sae_embedding()
