# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from scipy.fftpack import dct
from scipy.signal import lfilter
import matplotlib.pyplot as plt


def erb_space(low_freq=50, high_freq=8000, n=64):
    ear_q = 9.26449
    min_bw = 24.7
    cf_array = -(ear_q * min_bw) + np.exp(
        np.linspace(1, n, n) * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw)) / n) \
               * (high_freq + ear_q * min_bw)
    return cf_array


# gammatone 脉冲响应
def gammatone_impulse_response(samplerate_hz, length_in_samples, center_freq_hz, p):
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
    center_freqs = erb_space(50, fmax, N)
    center_freqs = np.flip(center_freqs)
    n_center_freqs = len(center_freqs)

    filterbank = np.zeros((N, L))

    # 为每个中心频率生成 滤波器
    for i in range(n_center_freqs):
        filterbank[i, :] = gammatone_impulse_response(fs, L, center_freqs[i], p)
    return filterbank


def gfcc(cochleagram, numcep=13):
    feat = dct(cochleagram, type=2, axis=1, norm='ortho')[:, :numcep]
    #    feat-= (np.mean(feat, axis=0) + 1e-8)#Cepstral mean substration
    return feat


def powerspec(X, n_padded):
    # Fourier transform
    #    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
    Y = np.absolute(Y)

    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]

    return np.abs(Y) ** 2, n_padded


def cochleagram(sig_spec, filterbank, nfft):
    """
    sig_spec: It's the STFT of the speech signal
    """
    filterbank, _ = powerspec(filterbank, nfft)  # |FFT|
    filterbank /= np.max(filterbank, axis=-1)[:, None]  # Normalize filters
    cochlea_spec = np.dot(sig_spec, filterbank.T)
    cochlea_spec = np.where(cochlea_spec == 0.0, np.finfo(float).eps, cochlea_spec)
    #    cochlea_spec= np.log(cochlea_spec)-np.mean(np.log(cochlea_spec),axis=0)
    cochlea_spec = np.log(cochlea_spec)
    return cochlea_spec, filterbank


if __name__ == "__main__":
    sample_rate = 16000
    length_in_samples = 16000
    center_freq_hz = 4000
    p =4
    gammatone_ir = gammatone_impulse_response(sample_rate, length_in_samples, center_freq_hz, p)
    filterbank = generate_filterbank(sample_rate, fmax=8000, L=length_in_samples, N=3, p=4)
    plt.plot(gammatone_ir)
    # plt.plot(filterbank.T)
    plt.show()
