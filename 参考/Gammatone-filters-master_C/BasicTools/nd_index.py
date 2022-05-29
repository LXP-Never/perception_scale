import numpy as np


def nd_index(i, size):
    size = np.asarray(size)
    n_dim = len(size)
    index_all = np.zeros(n_dim, dtype=np.int)
    left = i
    for dim_i in range(0, n_dim-1):
        capacity = np.prod(size[dim_i+1:])
        index_all[dim_i] = np.int(np.floor(left/capacity))
        left = left - index_all[dim_i]*capacity
    index_all[-1] = np.int(left)
    return index_all


if __name__ == '__main__':
    i, j = nd_index(10, [4, 4])
    print(i, j)
