import numpy as np

from params import TOTAL_EMBEDDING_DIM, N_CLASS
from utils import scale_1d_data

'''
https://github.com/anlongstory/CSAE
'''



mu = np.zeros(TOTAL_EMBEDDING_DIM)
sigma = np.eye(TOTAL_EMBEDDING_DIM)

u_init = np.random.multivariate_normal(mu, sigma, N_CLASS)
v = np.zeros(u_init.shape)
u = []
for i in u_init:
    i = i / np.linalg.norm(i)
    u.append(i)
u = np.array(u)
G = 1e-2


def countnext(u, v, G):
    num = u.shape[0]
    dd = np.zeros((TOTAL_EMBEDDING_DIM, num, num))
    for m in range(num):
        for n in range(num):
            dd[:, m, n] = (u[m, :] - u[n, :]).T
            dd[:, n, m] = -dd[:, m, n]
    L = np.sum(dd ** 2, 0) ** 0.5
    L = L.reshape(1, L.shape[0], L.shape[1])
    L[L < 1e-2] = 1e-2
    a = np.repeat(L ** 3, TOTAL_EMBEDDING_DIM, 0)
    F = np.sum(dd / a, 2).T
    tmp_F = []
    for i in range(F.shape[0]):
        tmp_F.append(np.dot(F[i], u[i]))
    d = np.array(tmp_F).T.reshape(len(tmp_F), 1)
    Fr = u * np.repeat(d, TOTAL_EMBEDDING_DIM, 1)
    Ft = F - Fr
    un = u + v
    ll = np.sum(un ** 2, 1) ** 0.5
    un = un / np.repeat(ll.reshape(ll.shape[0], 1), TOTAL_EMBEDDING_DIM, 1)
    vn = v + G * Ft
    return un, vn


def generate_center(u, v, G):
    for i in range(200):
        un, vn = countnext(u, v, G)
        u = un
        v = vn
    return u * (TOTAL_EMBEDDING_DIM) ** 0.5
    # return r



def repeat(c, n, scale=True):
    """
    centroids repeat n time to form a list
    """
    cs = np.array([c for i in range(n)])  # !!!!full of same element
    if scale:
        cs = scale_1d_data(cs)
    return cs


if __name__ == '__main__':
    x_train_RP = np.load('./geolife/train_mf_RP_mats.npy')
    x_test_RP = np.load('./geolife/test_mf_RP_mats.npy')

    n_train = x_train_RP.shape[0]
    n_test = x_test_RP.shape[0]

    c = generate_center(u, v, G)

    # !!those data are generated, no real trajectory data involved!!
    scale = True
    c_train = repeat(c, n_train, scale)
    c_test = repeat(c, n_test, scale)
    np.save('./geolife/train_centroids.npy', c_train)
    np.save('./geolife/test_centroids.npy', c_test)






