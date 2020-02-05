import tables
import numpy as np


def gen_mf_RP_mats(series, batch_size=10, scale=False):
    n = len(series)
    for i in range(0, n, batch_size):
        print(i, i + batch_size)
        print(series[i:i+batch_size])

a = np.array(range(125))
gen_mf_RP_mats(a)
