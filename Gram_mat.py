# coding=utf-8
from math import sqrt, sin, isclose

from scipy import interpolate, signal, spatial, ndimage
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def dot(a, b):
    return np.dot(a, b)


def dist(a, b):
    dist = np.linalg.norm(a - b)
    return dist


def cos_dist(a, b):
    cos_sim = spatial.distance.cosine(a, b)
    return cos_sim


def sin_dist(a, b):
    # 自己与自己的距离是0
    if a[0] == b[0] and a[1] == b[1]:
        # print 'a=b:{}={}  return 0'.format(a, b)
        return 0
    len_a = np.sqrt(a.dot(a))
    len_b = np.sqrt(b.dot(b))
    cos_angle = a.dot(b) / (len_a * len_b)
    norm_a = np.linalg.norm(a)
    # 在一条直线上,
    # isclose:https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python/33024979
    if isclose(cos_angle, 1.) or isclose(cos_angle, -1.):
        # print '\n{}  {}, angle = {}'.format(a, b, cos_angle)
        # print cos_angle == -1.
        dist = norm_a
    else:
        angle = np.arccos(cos_angle)
        dist = np.linalg.norm(a) * sin(angle)
    if np.isnan(dist):
        # print '\ncos_angle:{} abs:{}'.format(cos_angle, np.abs(cos_angle))
        # print 'sin_dist({}, {}) is nan'.format(a, b)
        # print 'cos_angle=%.24f' % cos_angle
        assert cos_angle == -1.
    return dist


def gen_gram_mat(series, f):
    '''
    :param series:
    :param f: 做运算的函数 f(a,b)
    :return:
    '''
    scaler = MinMaxScaler((-1, 1))
    series = scaler.fit_transform(series)

    mat = []
    seq_len = len(series)
    # 生成gram矩阵
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            element = f(series[i], series[j])
            row.append(element)
        mat.append(row)
    mat = np.array(mat)
    _mat = scaler.fit_transform(mat)
    return _mat

def segs_2_gram_mats(segs, f, out):
    mats = []
    # size = len(max(trjs, key=len)) if size is None else size
    n = len(segs)
    for i, seg in enumerate(segs):
        print('seg: {} / {}'.format(i, n))
        coords = np.array(seg)
        g = gen_gram_mat(coords, f)
        mats.append(g)
    mats = np.array(mats)  # 用于绘图
    mats = np.abs(mats)  # 使得只注重形状信息而不注重方向
    mats_res = np.expand_dims(mats, axis=3)  # 用于输入网络
    print('mats.shape:{}'.format(mats.shape))
    np.save(out, mats_res)
    return mats, mats_res

def interp_2d_segs(segs, size=24):
    '''
    https://stackoverflow.com/questions/31464345/fitting-a-closed-curve-to-a-set-of-points
    answer from Joe Kington
    '''
    size = len(max(segs, key=len)) if size is None else size  # 插值后的长度
    # max_len = 160.
    interped = []
    n = len(segs)
    for i, seg in enumerate(segs):
        print ('\n seg :{} / {}'.format(i, n))
        seg = np.array(seg)
        trj_len = float(len(seg))
        scale_ratio = size / trj_len  # 缩放比例
        seg *= scale_ratio

        points = seg[:, [1, 2]]
        ori = np.copy(points)

        x, y = points.T
        i = np.arange(len(points))

        interp_i = np.linspace(0, i.max(), size)

        xi = interp1d(i, x, kind='cubic')(interp_i)
        yi = interp1d(i, y, kind='cubic')(interp_i)


        points = [[x, y] for x, y in zip(xi, yi)]
        interped.append(np.array(points))
    interped = np.array(interped)
    print( 'interped.shape:{}'.format(interped.shape))
    return interped


if __name__ == '__main__':
    trjs_segs = np.load('./geolife/trjs_segments.npy', allow_pickle=True)
    trjs_segs = interp_2d_segs(trjs_segs, size=88)
    mats, _ = segs_2_gram_mats(trjs_segs, sin_dist, out='./geolife/trjs_Gram_mats.npy')
