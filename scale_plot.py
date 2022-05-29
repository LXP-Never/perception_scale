# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/25
"""

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def hz2erb_1983(f):
    """ 中心频率f(Hz) f to ERB(Hz) """
    f = f / 1000.0
    return (6.23 * (f ** 2)) + (93.39 * f) + 28.52


def hz2erb_1990(f):
    """ 中心频率f(Hz) f to ERB(Hz) """
    return 24.7 * (4.37 * f / 1000 + 1.0)


def hz2erb_1998(f):
    """ 中心频率f(Hz) f to ERB(Hz)
        hz2erb_1990 和 hz2erb_1990_2 的曲线几乎一模一样
        M. Slaney, Auditory Toolbox, Version 2, Technical Report No: 1998-010, Internal Research Corporation, 1998
        http://cobweb.ecn.purdue.edu/~malcolm/interval/1998-010/
    """
    return 24.7 + (f / 9.26449)


def Hz2erb_2002(f):
    """ [Hohmann2002] Equation 16 """
    EarQ = 9.265  # _ERB_Q
    minBW = 24.7  # minBW
    return EarQ * np.log(1 + f / (minBW * EarQ))


def Hz2erb_matlab(f):
    """ Convert Hz to ERB number """
    n_erb = 11.17268 * np.log(1 + (46.06538 * f) / (f + 14678.49))
    return n_erb


def Hz2erb_other(f):
    """ Convert Hz to ERB number """
    n_erb = 21.4 * np.log10(1 + 0.00437 * f)
    return n_erb


# freq2erb_matlab 和 Hz2erb_2002相差无几
# erb_1990和erb_1998相差无几

if __name__ == "__main__":
    fs = 16000
    hz = np.linspace(0, fs // 2, fs // 2)
    erb_1983 = hz2erb_1983(hz)
    erb_1990 = hz2erb_1990(hz)
    erb_1998 = hz2erb_1998(hz)
    erb_2002 = Hz2erb_2002(hz)
    Hz2erb_matlab = Hz2erb_matlab(hz)
    Hz2erb_other = Hz2erb_other(hz)

    plt.plot(hz, erb_1983, label="erb_1983")
    plt.plot(hz, erb_1990, label="erb_1990")
    plt.plot(hz, erb_1998, label="erb_1998")
    plt.plot(hz, erb_2002, label="erb_2002")
    plt.plot(hz, Hz2erb_matlab, label="Hz2erb_matlab")
    plt.plot(hz, Hz2erb_other, label="Hz2erb_other")

    plt.legend()  # 显示图例
    plt.xlabel("Center frequency (Hz)", fontsize=12)  # x轴的名字
    plt.ylabel("Equivalent Rectangular Bandwidth (Hz)", fontsize=12)

    plt.xticks(fontsize=10)  # x轴的刻度
    plt.yticks(fontsize=10)

    plt.xlim(0, fs // 2)  # 坐标轴的范围
    plt.ylim(0)


    def formatnum(x, pos):
        return '$%.1f$' % (x / 1000)


    formatter = FuncFormatter(formatnum)
    # plt.gca().xaxis.set_major_formatter(formatter)
    # plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.show()
