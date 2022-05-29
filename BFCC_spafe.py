# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/25
"""

"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

def hz2bark(f):
    """ Hz to bark频率 (Wang, Sekey & Gersho, 1992.) """
    return 6. * np.arcsinh(f / 600.)


def bark2hz(fb):
    """ Bark频率 to Hz """
    return 600. * np.sinh(fb / 6.)


def fft2hz(fft, fs=16000, nfft=512):
    """ FFT频点 to Hz """
    return (fft * fs) / (nfft + 1)


def hz2fft(fb, fs=16000, nfft=512):
    """ Bark频率 to FFT频点 """
    return (nfft + 1) * fb / fs


def fft2bark(fft, fs=16000, nfft=512):
    """ FFT频点 to Bark频率 """
    return hz2bark((fft * fs) / (nfft + 1))


def bark2fft(fb, fs=16000, nfft=512):
    """ Bark频率 to FFT频点 """
    # bin = sample_rate/2 / nfft/2=sample_rate/nfft    # 每个频点的频率数
    # bins = hz_points/bin=hz_points*nfft/ sample_rate    # hz_points对应第几个fft频点
    return (nfft + 1) * bark2hz(fb) / fs


def Fm(fb, fc):
    """ 计算一个特定的中心频率的Bark filter
    :param fb: frequency in Bark.
    :param fc: center frequency in Bark.
    :return: 相关的Bark filter 值/幅度
    """
    if fc - 2.5 <= fb <= fc - 0.5:
        return 10 ** (2.5 * (fb - fc + 0.5))
    elif fc - 0.5 < fb < fc + 0.5:
        return 1
    elif fc + 0.5 <= fb <= fc + 1.3:
        return 10 ** (-2.5 * (fb - fc - 0.5))
    else:
        return 0


def bark_filter_banks(nfilts=20, nfft=512, fs=16000, low_freq=0, high_freq=None, scale="constant"):
    """ 计算Bark-filterbanks,(B,F)
    :param nfilts: 滤波器组中滤波器的数量 (Default 20)
    :param nfft: FFT size.(Default is 512)
    :param fs: 采样率，(Default 16000 Hz)
    :param low_freq: MEL滤波器的最低带边。(Default 0 Hz)
    :param high_freq: MEL滤波器的最高带边。(Default samplerate/2)
    :param scale (str): 选择Max bins 幅度 "ascend"(上升)，"descend"(下降)或 "constant"(恒定)(=1)。默认是"constant"
    :return:一个大小为(nfilts, nfft/2 + 1)的numpy数组，包含滤波器组。
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # 按Bark scale 均匀间隔计算点数(点数以Bark为单位)
    low_bark = hz2bark(low_freq)
    high_bark = hz2bark(high_freq)
    bark_points = np.linspace(low_bark, high_bark, nfilts + 4)

    bins = np.floor(bark2fft(bark_points))  # Bark Scale等分布对应的 FFT bin number
    # [  0.   2.   5.   7.  10.  13.  16.  20.  24.  28.  33.  38.  44.  51.
    #   59.  67.  77.  88. 101. 115. 132. 151. 172. 197. 224. 256.]
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    # init scaler
    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c = 0

    for i in range(0, nfilts):      # --> B
        # compute scaler
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)
        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        for j in range(int(bins[i]), int(bins[i + 4])):     # --> F
            fc = bark_points[i+2]   # 中心频率
            fb = fft2bark(j)        # Bark 频率
            fbank[i, j] = c * Fm(fb, fc)
    return np.abs(fbank)


if __name__ == "__main__":
    nfilts = 22
    NFFT = 512
    fs = 16000
    wav = librosa.load("p225_001.wav",sr=fs)[0]
    S = librosa.stft(wav, n_fft=NFFT, hop_length=NFFT // 2, win_length=NFFT, window="hann", center=False)
    mag = np.abs(S)  # 幅度谱 (257, 127) librosa.magphase()
    filterbanks = bark_filter_banks(nfilts=nfilts, nfft=NFFT, fs=fs, low_freq=0, high_freq=None, scale="constant")
    # ================ 画三角滤波器 ===========================
    FFT_len = NFFT // 2 + 1
    fs_bin = fs // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, FFT_len, FFT_len)

    plt.plot(x * fs_bin, filterbanks.T)

    # plt.xlim(0)  # 坐标轴的范围
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()

    filter_banks = np.dot(filterbanks, mag)  # (M,F)*(F,T)=(M,T)
    filter_banks = 20 * np.log10(filter_banks)  # dB
    # ================ 绘制语谱图 ==========================
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