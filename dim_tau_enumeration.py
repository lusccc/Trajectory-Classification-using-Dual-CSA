ns = []
dims = []
taus = []
n_vecs = []


def find_dim_tau():
    # for n in range(150, 400):
    for n in range(400, 600):
        # n=600
        for dim in range(1, 10):
            for tau in range(1, 10):
                n_vectors = n - tau * (dim - 1)
                if n > 0 and n % 8 == 0 and n_vectors % 8 == 0 and 300 > n_vectors > 32:
                    ns.append(n)
                    dims.append(dim)
                    taus.append(tau)
                    n_vecs.append(n_vectors)
                    print(n, dim, tau, n_vectors)


find_dim_tau()
print(*ns, sep=' ')
print(*dims, sep=' ')
print(*taus, sep=' ')
print(*n_vecs, sep=' ')
