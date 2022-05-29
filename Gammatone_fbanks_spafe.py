# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/24
"""
Gammatone-filter-banks implementation
based on https://github.com/mcusi/gammatonegram/
"""
import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

# Slaney's ERB Filter constants
EarQ = 9.26449
minBW = 24.7


def generate_center_frequencies(min_freq, max_freq, filter_nums):
    """ 计算中心频率(ERB scale)
    :param min_freq: 中心频率域的最小频率。
    :param max_freq: 中心频率域的最大频率。
    :param filter_nums: 滤波器的个数，即等于计算中心频率的个数。
    :return: 一组中心频率
    """
    # init vars
    n = np.linspace(1, filter_nums, filter_nums)
    c = EarQ * minBW

    # 计算中心频率
    cfreqs = (max_freq + c) * np.exp((n / filter_nums) * np.log(
        (min_freq + c) / (max_freq + c))) - c

    return cfreqs


def compute_gain(fcs, B, wT, T):
    """ 为了 阶数 计算增益和矩阵化计算
    :param fcs: 中心频率
    :param B: 滤波器的带宽
    :param wT: 对应于用于频域计算的 w * T = 2 * pi * freq * T
    :param T: 周期(单位秒s)，1/fs
    :return:
        Gain: 表示filter gains 的2d numpy数组
        A: 用于最终计算的二维数组
    """
    # 为了简化 预先计算
    K = np.exp(B * T)
    Cos = np.cos(2 * fcs * np.pi * T)
    Sin = np.sin(2 * fcs * np.pi * T)
    Smax = np.sqrt(3 + 2 ** (3 / 2))
    Smin = np.sqrt(3 - 2 ** (3 / 2))

    # 定义A矩阵的行
    A11 = (Cos + Smax * Sin) / K
    A12 = (Cos - Smax * Sin) / K
    A13 = (Cos + Smin * Sin) / K
    A14 = (Cos - Smin * Sin) / K

    # 计算增益 (vectorized)
    A = np.array([A11, A12, A13, A14])
    Kj = np.exp(1j * wT)
    Kjmat = np.array([Kj, Kj, Kj, Kj]).T
    G = 2 * T * Kjmat * (A.T - Kjmat)
    Coe = -2 / K ** 2 - 2 * Kj ** 2 + 2 * (1 + Kj ** 2) / K
    Gain = np.abs(G[:, 0] * G[:, 1] * G[:, 2] * G[:, 3] * Coe ** -4)
    return A, Gain


def gammatone_filter_banks(nfilts=22, nfft=512, fs=16000, low_freq=None, high_freq=None, scale="contsant", order=4):
    """ 计算Gammatone-filterbanks, (G,F)
    :param nfilts: filterbank中滤波器的数量 (Default 22)
    :param nfft: FFT size (Default is 512)
    :param fs: 采样率 (Default 16000 Hz)
    :param low_freq: 最低频带 (Default 0 Hz)
    :param high_freq: 最高频带 (Default samplerate/2)
    :param scale: 选择Max bins 幅度 "ascend"(上升)，"descend"(下降)或 "constant"(恒定)(=1)。默认是"constant"
    :param order: 滤波器阶数
    :return: 一个大小为(nfilts, nfft/2 + 1)的numpy数组，包含滤波器组。
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # define custom difference func
    def Dif(u, a):
        return u - a.reshape(nfilts, 1)

    # init vars
    fbank = np.zeros([nfilts, nfft])
    width = 1.0
    maxlen = nfft // 2 + 1
    T = 1 / fs
    n = 4
    u = np.exp(1j * 2 * np.pi * np.array(range(nfft // 2 + 1)) / nfft)
    idx = range(nfft // 2 + 1)

    fcs = generate_center_frequencies(low_freq, high_freq, nfilts)  # 计算中心频率，转换到ERB scale
    ERB = width * ((fcs / EarQ) ** order + minBW ** order) ** (1 / order)  # 计算带宽
    B = 1.019 * 2 * np.pi * ERB

    # compute input vars
    wT = 2 * fcs * np.pi * T
    pole = np.exp(1j * wT) / np.exp(B * T)

    # compute gain and A matrix
    A, Gain = compute_gain(fcs, B, wT, T)

    # compute fbank
    fbank[:, idx] = (
            (T ** 4 / Gain.reshape(nfilts, 1)) *
            np.abs(Dif(u, A[0]) * Dif(u, A[1]) * Dif(u, A[2]) * Dif(u, A[3])) *
            np.abs(Dif(u, pole) * Dif(u, pole.conj())) ** (-n))

    # 确保所有filters的最大值为1.0
    try:
        fbs = np.array([f / np.max(f) for f in fbank[:, range(maxlen)]])
    except BaseException:
        fbs = fbank[:, idx]

    # compute scaler
    if scale == "ascendant":
        c = [
            0,
        ]
        for i in range(1, nfilts):
            x = c[i - 1] + 1 / nfilts
            c.append(x * (x < 1) + 1 * (x > 1))
    elif scale == "descendant":
        c = [
            1,
        ]
        for i in range(1, nfilts):
            x = c[i - 1] - 1 / nfilts
            c.append(x * (x > 0) + 0 * (x < 0))
    else:
        c = [1 for i in range(nfilts)]

    # apply scaler
    c = np.array(c).reshape(nfilts, 1)
    fbs = c * np.abs(fbs)
    return fbs


if __name__ == "__main__":
    nfilts = 22
    NFFT = 512
    fs = 16000

    FFT_len = NFFT // 2 + 1
    fs_bin = fs // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, FFT_len, FFT_len)

    # ================ 画三角滤波器 ===========================
    filterbanks = gammatone_filter_banks(nfilts=22, nfft=512, fs=16000,
                                         low_freq=None, high_freq=None,
                                         scale="contsant", order=4)
    print(filterbanks.shape)    # (22, 257)
    plt.plot(x * fs_bin, filterbanks.T)

    # plt.xlim(0)  # 坐标轴的范围
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()

    # ================ 绘制语谱图 ==========================
    wav = librosa.load("p225_001.wav", sr=fs)[0]
    S = librosa.stft(wav, n_fft=NFFT, hop_length=NFFT // 2, win_length=NFFT, window="hann", center=False)
    mag = np.abs(S)  # 幅度谱 (257, 127) librosa.magphase()

    filter_banks = np.dot(filterbanks, mag)  # (M,F)*(F,T)=(M,T)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    plt.figure()
    librosa.display.specshow(filter_banks, sr=fs, x_axis='time', y_axis='linear', cmap="jet")
    plt.xlabel('时间/s', fontsize=14)
    plt.ylabel('频率/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    def formatnum(x, pos):
        return '$%d$' % (x / 1000)


    formatter = FuncFormatter(formatnum)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()
