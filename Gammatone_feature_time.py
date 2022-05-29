# -*- coding:utf-8 -*-
# Author:凌逆战 | Never.Ling
# Date: 2022/5/24
"""
时域滤波器组 FFT 转频域滤波器组 与语音频谱相乘
参考：https://github.com/TAriasVergara/Acoustic_features
"""
import librosa
import librosa.display
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号


def erb_space(low_freq=50, high_freq=8000, filter_nums=64):
    """ 计算中心频率(ERB scale)
    :param min_freq: 中心频率域的最小频率。
    :param max_freq: 中心频率域的最大频率。
    :param filter_nums: 滤波器的个数，即等于计算中心频率的个数。
    :return: 一组中心频率
    """
    EarQ = 9.26449
    minBW = 24.7
    c = EarQ * minBW
    n = np.linspace(1, filter_nums, filter_nums)

    cf_array = (high_freq + c) * np.exp((n / filter_nums) * np.log(
        (low_freq + c) / (high_freq + c))) - c
    return cf_array


def gammatone_impulse_response(samplerate_hz, length_in_samples, center_freq_hz, p):
    """ gammatone滤波器的时域公式
    :param samplerate_hz: 采样率
    :param length_in_samples: 信号长度
    :param center_freq_hz: 中心频率
    :param p: 滤波器阶数
    :return: gammatone 脉冲响应
    """
    # 生成一个gammatone filter (1990 Glasberg&Moore parametrized)
    erb = 24.7 + (center_freq_hz / 9.26449)  # equivalent rectangular bandwidth.
    # 中心频率
    an = (np.pi * np.math.factorial(2 * p - 2) * np.power(2, float(-(2 * p - 2)))) / np.square(np.math.factorial(p - 1))
    b = erb / an  # 带宽

    a = 1  # 幅度(amplitude). 这在后面的归一化过程中有所不同。
    t = np.linspace(1. / samplerate_hz, length_in_samples / samplerate_hz, length_in_samples)
    gammatone_ir = a * np.power(t, p - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * center_freq_hz * t)
    return gammatone_ir


def generate_filterbank(fs, fmax, L, N, p=4):
    """
    L: 在样本中测量的信号的大小
    N: 滤波器数量
    p: Gammatone脉冲响应的阶数
    """
    # 中心频率
    if fs == 8000:
        fmax = 4000
    center_freqs = erb_space(50, fmax, N)  # 中心频率列表
    center_freqs = np.flip(center_freqs)  # 反转数组
    n_center_freqs = len(center_freqs)  # 中心频率的数量

    filterbank = np.zeros((N, L))

    # 为每个中心频率生成 滤波器
    for i in range(n_center_freqs):
        # aa = gammatone_impulse_response(fs, L, center_freqs[i], p)
        filterbank[i, :] = gammatone_impulse_response(fs, L, center_freqs[i], p)
    return filterbank


def gfcc(cochleagram, numcep=13):
    feat = dct(cochleagram, type=2, axis=1, norm='ortho')[:, :numcep]
    #    feat-= (np.mean(feat, axis=0) + 1e-8)#Cepstral mean substration
    return feat


def cochleagram(sig_spec, filterbank, nfft):
    """
    :param sig_spec: 语音频谱
    :param filterbank: 时域滤波器组
    :param nfft: fft_size
    :return:
    """
    filterbank = powerspec(filterbank, nfft)  # 时域滤波器组经过FFT变换
    filterbank /= np.max(filterbank, axis=-1)[:, None]  # Normalize filters
    cochlea_spec = np.dot(sig_spec, filterbank.T)  # 矩阵相乘
    cochlea_spec = np.where(cochlea_spec == 0.0, np.finfo(float).eps, cochlea_spec)  # 把0变成一个很小的数
    # cochlea_spec= np.log(cochlea_spec)-np.mean(np.log(cochlea_spec),axis=0)
    cochlea_spec = np.log(cochlea_spec)
    return cochlea_spec, filterbank


def powerspec(X, nfft):
    # Fourier transform
    # Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=nfft)
    Y = np.absolute(Y)

    # non-redundant part
    m = int(nfft / 2) + 1
    Y = Y[:, :m]

    return np.abs(Y) ** 2


if __name__ == "__main__":
    nfilts = 22
    NFFT = 512
    fs = 16000
    Order = 4

    FFT_len = NFFT // 2 + 1
    fs_bin = fs // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, FFT_len, FFT_len)
    # ================ 画三角滤波器 ===========================
    # gammatone_impulse_response = gammatone_impulse_response(fs/2, 512, 200, Order)    #  gammatone冲击响应
    generate_filterbank = generate_filterbank(fs, fs // 2, FFT_len, nfilts, Order)
    filterbanks = powerspec(generate_filterbank, NFFT)  # 时域滤波器组经过FFT变换
    filterbanks /= np.max(filterbanks, axis=-1)[:, None]  # Normalize filters
    print(generate_filterbank.shape)    # (22, 257)
    # plt.plot(filterbanks.T)
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
