from math import cos, pi

import numpy as np
import tables
from numpy.linalg import norm
from pyts.image.recurrence import _trajectories

from params import *

from utils import scale_any_shape_data

#  Settings for the embedding

# Distance metric in phase space ->
# Possible choices ("manhattan","euclidean","supremum")
METRIC = "euclidean"

EPS = 0.05  # Fixed recurrence threshold


def gen_RP_mats_for_multiple_series(series, scale=False):
    """
    generate a RP mat for each series
    """

    n_series = len(series)

    # each series generates n_vectors phase space phase_vectors
    phase_vectorss = _trajectories(series, dimension=DIM, time_delay=TAU)
    n_vectors = phase_vectorss.shape[1]
    # calc distances mats, 1 mat for each series.
    # shape:(n_samples, n_vectors, n_vectors)
    # from source code in pyts.image.recurrence.RecurrencePlot.transform
    distance_mats = np.sqrt(
        np.sum((phase_vectorss[:, None, :, :] - phase_vectorss[:, :, None, :]) ** 2,
               axis=3, dtype='float32')
    )

    mycode = False
    if mycode:
        ## my code to calc distances
        RP_mats = []  # !!skipped Heavside function
        count = 0
        for distance_mat, phase_vectors in zip(distance_mats, phase_vectorss):
            print('{}/{}'.format(count, n_series))
            # now no need to use this method since we want to calc sign value
            # https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
            # distances = pdist(phase_vectors, )
            sign_value_mat = np.empty((n_vectors, n_vectors))
            for i in range(n_vectors):
                for j in range(n_vectors):
                    a = phase_vectors[i]
                    b = phase_vectors[j]
                    if i == j:
                        sign_value = 1
                    else:
                        # equation (5) in "Robust Single Accelerometer-Based Activity Recognition
                        # Using Modified Recurrence Plot"
                        sign_value = sign(a, b)
                    sign_value_mat[i][j] = sign_value
            sign_value_mat = np.array(sign_value_mat)
            RP_mat = sign_value_mat * distance_mat
            RP_mats.append(RP_mat)
            count += 1

        RP_mats = np.array(RP_mats)
    else:
        RP_mats = distance_mats  # !!skipped Heavside function
    if scale:
        RP_mats = scale_any_shape_data(RP_mats)
    return RP_mats


def sign(m, n):
    # equation (5) in "Robust Single Accelerometer-Based Activity Recognition
    # Using Modified Recurrence Plot"
    m = np.array(m)
    n = np.array(n)
    dim = m.shape[0]  # vector dim
    v = np.repeat(1, dim)  # base vec
    cos_angle = np.dot(m - n, v) / (norm(m - n) * norm(v))
    if cos_angle < cos(3. / 4. * pi):
        return -1
    else:
        return 1


def gen_mf_RP_mats(series, mats_earray, batch_size=100, scale=False):
    n = len(series)
    for i in range(0, n, batch_size):
        # (batch_size, n_vectors, embedding_dim)
        mats = gen_RP_mats_for_multiple_series(series[i:i + batch_size])
        mats = np.expand_dims(mats, axis=3)
        mats_earray.append(mats)

        print()


if __name__ == '__main__':

    labels = np.load('./geolife_features/trjs_segs_features_labels.npy')

    data_type = ['clean', 'noise']

    for type in data_type:
        print('++++++++data type:', type)
        # only use movement features to generate RP mat
        trjs_segs_features = np.load('./geolife_features/trjs_segs_{}_features.npy'.format(type))[:]
        features_RP_mats = []
        n_features = trjs_segs_features.shape[3]
        # reshape to (n_samples, MAX_SEGMENT_SIZE, n_features)
        trjs_segs_features = np.reshape(trjs_segs_features, [trjs_segs_features.shape[0], trjs_segs_features.shape[2],
                                                             trjs_segs_features.shape[3]])
        for i in range(n_features):
            print(i)
            mats_file = tables.open_file('./geolife_features/RP_mats_{}_feature_{}_mf.hdf5'.format(type, i), mode='w')
            mats_earray = mats_file.create_earray(
                mats_file.root,
                'data',  # 数据名称，之后需要通过它来访问数据
                tables.Float32Atom(),  # 设定数据格式（和data1格式相同）
                shape=(0, N_VECTORS, N_VECTORS, 1),  # 第一维的 0 表示数据可沿行扩展
                filters=tables.Filters(complevel=5, complib='blosc')
            )
            single_feature_segs = trjs_segs_features[:, :, i]  # (n, 48)
            # generate RP mat for each seg
            gen_mf_RP_mats(single_feature_segs, mats_earray)
            print(mats_file.root.data.shape)
            # mats_file.close()

            features_RP_mats.append(mats_earray)
        features_RP_mats = np.concatenate(features_RP_mats, axis=3)
        print()
        all_mats_file = tables.open_file('./geolife_features/RP_mats_{}_mf.hdf5'.format(type), mode='w')
        all_mats_earray = mats_file.create_earray(
            mats_file.root,
            'data',  # 数据名称，之后需要通过它来访问数据
            tables.Float32Atom(),  # 设定数据格式（和data1格式相同）
            shape=(0, N_VECTORS, N_VECTORS, n_features),  # 第一维的 0 表示数据可沿行扩展
            filters=tables.Filters(complevel=5, complib='blosc')
        )
        all_mats_earray.append(features_RP_mats)



            # feature_RP_mats = gen_RP_mats_for_multiple_series(single_feature_segs[:], scale=False)

        #     features_RP_mats.append(feature_RP_mats)
        #     max = np.max(feature_RP_mats)
        #     min = np.min(feature_RP_mats)
        #     print('RP mat value range:', min, max)
        # features_RP_mats = np.concatenate(features_RP_mats, axis=3)
        #
        # print(' ####\nfeatures_RP_mats.shape:{}'.format(features_RP_mats.shape))
        # np.save('./geolife_features/RP_mats_{}_mf.npy'.format(type), features_RP_mats)
