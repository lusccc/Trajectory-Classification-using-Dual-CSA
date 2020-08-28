import os
from os.path import exists

import matplotlib.pyplot as plt
import seaborn as sns
from backup.trajectory_features_and_segmentation import MAX_SEGMENT_SIZE
from dataset_generation import *
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Input
from keras.layers import Lambda, \
    concatenate
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.utils import plot_model, multi_gpu_model
from sklearn.metrics import confusion_matrix, classification_report

from trajectory_extraction_geolife import modes_to_use
from keras_support.network_keras import CONV2D_AE
from keras_support.network_keras.TS_CONV2D_AE import CONV1D_AE
from params import TOTAL_EMBEDDING_DIM, MULTI_GPU
from utils import visualize_data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    RP_conv_ae = CONV2D_AE((RP_mat_size, RP_mat_size, n_features), each_embedding_dim, n_features, 'RP')
    if MULTI_GPU:
        RP_conv_ae = multi_gpu_model(RP_conv_ae, gpus=2)
    RP_conv_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return RP_conv_ae


def ts_Conv2d_AE():
    n_features = x_features_series_train.shape[3]
    ts_conv1d_ae = CONV1D_AE((1, MAX_SEGMENT_SIZE, n_features), each_embedding_dim, n_features, 'ts')
    if MULTI_GPU:
        ts_conv1d_ae = multi_gpu_model(ts_conv1d_ae, gpus=2)
    ts_conv1d_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return ts_conv1d_ae


def dual_SAE():
    """--------SAE----------"""
    concat_embedding = concatenate([RP_conv2d_ae.get_layer('RP_embedding').output,
                                    ts_conv2d_ae.get_layer('ts_embedding').output])
    centroids_input = Input((N_CLASS, TOTAL_EMBEDDING_DIM))
    classification = classification_layer([concat_embedding, centroids_input])

    dual_sae = Model(
        inputs=[RP_conv2d_ae.get_layer(index=0).input, centroids_input, ts_conv2d_ae.get_layer(index=0).input],
        outputs=[RP_conv2d_ae.get_layer('RP_reconstruction').output, classification,
                 ts_conv2d_ae.get_layer('ts_reconstruction').output])
    dual_sae.summary()
    plot_model(dual_sae, to_file='../results/dual_sae.png', show_shapes=True)

    dual_encoder = Model(
        inputs=[RP_conv2d_ae.get_layer(index=0).input, centroids_input, ts_conv2d_ae.get_layer(index=0).input],
        outputs=[concat_embedding]
    )
    dual_encoder.summary()
    plot_model(dual_encoder, to_file='../results/dual_encoder.png', show_shapes=True)

    if MULTI_GPU:
        dual_sae = multi_gpu_model(dual_sae, gpus=2)
    dual_sae.compile(loss=['mse', 'kld', 'mse'], loss_weights=loss_weights, optimizer='adam',
                     metrics=['accuracy', categorical_accuracy, 'accuracy'])
    return dual_sae, dual_encoder


"""-----------train---------------"""
tb = TensorBoard(log_dir='./logs',  # lines 目录
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
    cp_path = '../results/RP_conv_ae_check_point.model'
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
    cp_path = '../results/ts_conv_ae_check_point.model'
    cp = ModelCheckpoint(cp_path, monitor='loss', verbose=1,
                         save_best_only=True, mode='min')
    ts_conv1d_ae_ = ts_conv2d_ae
    if exists(cp_path):
        ts_conv1d_ae_ = load_model(cp_path)
    ts_conv1d_ae_.fit(x_features_series_train, x_features_series_train, batch_size=batch_size, epochs=epochs,
                      callbacks=[tb, cp])


def train_classifier(epochs=100, batch_size=200):
    print('train_classifier...')
    cp = ModelCheckpoint('../results/sae_check_point.model', monitor='val_lambda_1_acc',
                         verbose=1,
                         save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_lambda_1_loss', patience=patience, verbose=2)
    visulazation_callback = SAE_embedding_visualization_callback('./results/sae_cp_{epoch}.h5')
    hist = dual_sae.fit([x_RP_train, x_centroids_train, x_features_series_train],
                        [x_RP_train, y_train, x_features_series_train],
                        epochs=epochs,
                        batch_size=batch_size, shuffle=True,
                        validation_data=(
                            [x_RP_test, x_centroids_test, x_features_series_test],
                            [x_RP_test, y_test, x_features_series_test]),
                        callbacks=[early_stopping, tb, cp, visulazation_callback]
                        )
    #
    score = np.argmax(hist.history['val_lambda_1_acc'])
    print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[score],
                                                                             np.max(hist.history['val_lambda_1_acc'])))


class SAE_embedding_visualization_callback(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(SAE_embedding_visualization_callback, self).__init__(*args, **kwargs)

    # redefine the save so it only activates after 100 epochs
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 4 == 0:
            embedding = dual_encoder.predict([x_RP_test, x_centroids_test, x_features_series_test])
            y_true = np.argmax(y_test, axis=1)
            visualize_data(embedding, y_true, N_CLASS, './results/visualization_and_analysis/sae_embedding_epoch{}.png'.format(epoch))


def show_confusion_matrix():
    sae = load_model('../results/sae_check_point.model', custom_objects={'student_t': student_t, 'N_CLASS': N_CLASS})
    pred = sae.predict([x_RP_test, x_centroids_test, x_features_series_test])
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


def visualize_sae_embedding():
    sae = load_model('../results/sae_check_point.model', custom_objects={'student_t': student_t, 'N_CLASS': N_CLASS})
    embedding = dual_encoder.predict([x_RP_test, x_centroids_test, x_features_series_test])
    y_true = np.argmax(y_test, axis=1)
    visualize_data(embedding, y_true, N_CLASS, './results/visualization_and_analysis/best.png')


def visualize_dual_ae_embedding():
    load_model('../results/RP_conv_ae_check_point.model')
    load_model('../results/ts_conv_ae_check_point.model')
    embedding = dual_encoder.predict([x_RP_test, x_centroids_test, x_features_series_test])
    y_true = np.argmax(y_test, axis=1)
    visualize_data(embedding, y_true, N_CLASS, './results/visualization_and_analysis/dual_ae_embedding.png')


def visualize_centroids():
    visualize_data(x_centroids_test[0], modes_to_use, N_CLASS, './results/visualization_and_analysis/centroids_visualization.png')


if __name__ == '__main__':
    epochs = 30
    batch_size = 300
    """ note: each autoencoder has same embedding,
     embedding will be concated to match TOTAL_EMBEDDING_DIM, 
    aka. centroid has dim TOTAL_EMBEDDING_DIM"""
    n_ae = 2  # num of ae
    each_embedding_dim = int(TOTAL_EMBEDDING_DIM / n_ae)
    """no joint train here!"""
    loss_weights = [0, 1, 0]
    patience = 35

    RP_conv2d_ae = RP_Conv2D_AE()
    ts_conv2d_ae = ts_Conv2d_AE()
    dual_sae, dual_encoder = dual_SAE()
    visualize_centroids()
    train_classifier(3000, batch_size)
    show_confusion_matrix()
    visualize_sae_embedding()
