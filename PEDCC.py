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
        centroids repeat n time to form a list
        """
        cs = np.array([c for i in range(n)])  # !!!!full of same element
        if scale:
            cs = scale_any_shape_data(cs)
        return cs




if __name__ == '__main__':
    pedcc = PEDDC(TOTAL_EMBEDDING_DIM)
    c = pedcc.generate_center()
    n_samples = np.load('./data/geolife_features/trjs_segs_features_labels.npy').shape[0]
    # !!those data are generated, no real trajectory data involved!!
    scale = True
    centroids = pedcc.repeat(c, n_samples, scale)
    np.save('./data/geolife_features/centroids.npy', centroids)






