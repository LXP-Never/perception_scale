import matplotlib.pyplot as plt
import numpy as np
import time
from GTF import GTF


def efficiency_check():
    fs = np.int32(16e3)
    gt_filter = GTF(fs, freq_low=80, freq_high=5e3, n_band=16)

    ir_duration = 1
    n_sample = np.int(fs*ir_duration)
    x = np.random.rand(fs)

    t_start = time.time()
    ir_c = gt_filter.filter(x);
    t_comsum_c = time.time()-t_start

    t_start = time.time()
    ir_py = gt_filter.filter_py(x)
    t_comsum_py = time.time()-t_start

    print('time consumed(s) for filtering signal with length of 16e3 samples\n\
           {:<10}:{:.2f} \n\
           {:<10}:{:.2f}'.format('c', t_comsum_c,
                                 'python', t_comsum_py))


if __name__ == "__main__":
    efficiency_check()
