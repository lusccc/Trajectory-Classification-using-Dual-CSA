import argparse
import os

import numpy as np

from params import *
from utils import scale_any_shape_data

'''
https://github.com/anlongstory/CSAE
'''


class PEDDC(object):

    def __init__(self, EMBEDDING_DIM):
        super().__init__()
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def countnext(self, u, v, G):
        num = u.shape[0]
        dd = np.zeros((self.EMBEDDING_DIM, num, num))
        for m in range(num):
            for n in range(num):
                dd[:, m, n] = (u[m, :] - u[n, :]).T
                dd[:, n, m] = -dd[:, m, n]
        L = np.sum(dd ** 2, 0) ** 0.5
        L = L.reshape(1, L.shape[0], L.shape[1])
        L[L < 1e-2] = 1e-2
        a = np.repeat(L ** 3, self.EMBEDDING_DIM, 0)
        F = np.sum(dd / a, 2).T
        tmp_F = []
        for i in range(F.shape[0]):
            tmp_F.append(np.dot(F[i], u[i]))
        d = np.array(tmp_F).T.reshape(len(tmp_F), 1)
        Fr = u * np.repeat(d, self.EMBEDDING_DIM, 1)
        Ft = F - Fr
        un = u + v
        ll = np.sum(un ** 2, 1) ** 0.5
        un = un / np.repeat(ll.reshape(ll.shape[0], 1), self.EMBEDDING_DIM, 1)
        vn = v + G * Ft
        return un, vn

    def generate_center(self):
        print('EMBEDDING_DIM:{}'.format(self.EMBEDDING_DIM))
        mu = np.zeros(self.EMBEDDING_DIM)
        sigma = np.eye(self.EMBEDDING_DIM)

        u_init = np.random.multivariate_normal(mu, sigma, N_CLASS)
        v = np.zeros(u_init.shape)
        u = []
        for i in u_init:
            i = i / np.linalg.norm(i)
            u.append(i)
        u = np.array(u)
        G = 1e-2

        for i in range(200):
            un, vn = self.countnext(u, v, G)
            u = un
            v = vn
        return u * (self.EMBEDDING_DIM) ** 0.5
        # return r

    def repeat(self, c, n, scale=True):
        """
        centroid repeat n time to form a list
        """
        cs = np.array([c for i in range(n)])  # !!!!full of same element
        if scale:
            cs = scale_any_shape_data(cs)
        return cs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PEDCC')
    # parser.add_argument('--features_path', default='None', type=str)
    parser.add_argument('--save_dir', type=str)
    # parser.add_argument('--data_type', type=str)  # train or test
    parser.add_argument('--dim', default=TOTAL_EMBEDDING_DIM, type=int)
    args = parser.parse_args()
    # features_path = args.features_path
    save_dir = args.save_dir
    dim = args.dim
    # data_type = args.data_type
    scale = True

    pedcc = PEDDC(dim)
    # load single pedcc for generate train or test set, because each generating of pedcc is random, we should keep
    # train and test have same pedcc
    if not os.path.exists(f'{save_dir}/pedcc.npy'):
        c = pedcc.generate_center()
        print('saving single pedcc')
        np.save(f'{save_dir}/pedcc.npy', c)
    else:
        c = np.load(f'{save_dir}/pedcc.npy')
        if dim != c.shape[1]:
            c = pedcc.generate_center()  # generate for new dim
            print('saving single pedcc')
            np.save(f'{save_dir}/pedcc.npy', c)

    print()