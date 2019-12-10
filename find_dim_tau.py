def find_dim_tau():
    for n in range(0, 200):
        for dim in range(0, 10):
            for tau in range(0, 10):
                n_vectors = n - tau * (dim - 1)
                if n % 8 == 0 and n_vectors % 8 == 0:
                    print(n, dim, tau, n_vectors)


find_dim_tau()
