import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
from GTF import GTF

fig_dir = './images/gain_normalization/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


def gain_norm_test():
    fs = 16e3
    gt_filter = GTF(fs, freq_low=80, freq_high=5e3, n_band=16)

    # ir: 滤波器脉冲信号
    irs = gt_filter.get_ir(is_gain_norm=False)
    fig = gt_filter.plot_ir_spec(irs)
    plt.plot(fig)
    plt.savefig(fig, 'irs.png')

    # gain normalization
    irs_norm = gt_filter.get_ir()
    fig = gt_filter.plot_ir_spec(irs_norm)
    plt.plot(fig)
    plt.savefig(fig, 'irs_norm.png')

    # delays and gains
    fig = gt_filter.plot_delay_gain_cfs()
    plt.plot(fig)
    plt.savefig(fig, 'delay_gain.png')

    plt.show()


if __name__ == '__main__':
    gain_norm_test()
