import numpy as np

'''
https://github.com/anlongstory/CSAE
'''

LATENT_VARIABLE_DIM = 48
N_CLASS = 10

mu = np.zeros(LATENT_VARIABLE_DIM)
sigma = np.eye(LATENT_VARIABLE_DIM)

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
    dd = np.zeros((LATENT_VARIABLE_DIM, num, num))
    for m in range(num):
        for n in range(num):
            dd[:, m, n] = (u[m, :] - u[n, :]).T
            dd[:, n, m] = -dd[:, m, n]
    L = np.sum(dd ** 2, 0) ** 0.5
    L = L.reshape(1, L.shape[0], L.shape[1])
    L[L < 1e-2] = 1e-2
    a = np.repeat(L ** 3, LATENT_VARIABLE_DIM, 0)
    F = np.sum(dd / a, 2).T
    tmp_F = []
    for i in range(F.shape[0]):
        tmp_F.append(np.dot(F[i], u[i]))
    d = np.array(tmp_F).T.reshape(len(tmp_F), 1)
    Fr = u * np.repeat(d, LATENT_VARIABLE_DIM, 1)
    Ft = F - Fr
    un = u + v
    ll = np.sum(un ** 2, 1) ** 0.5
    un = un / np.repeat(ll.reshape(ll.shape[0], 1), LATENT_VARIABLE_DIM, 1)
    vn = v + G * Ft
    return un, vn


def generate_center(u, v, G):
    for i in range(200):
        un, vn = countnext(u, v, G)
        u = un
        v = vn
    return u * (LATENT_VARIABLE_DIM) ** 0.5
    # return r


def get_centroids(n_samples):
    # shape(5,48)
    c = generate_center(u, v, G)
    cs = np.array([c for i in range(n_samples)]) # full of same element

    return cs

get_centroids(1000)
