# -*- coding:utf-8 -*-
# Author:凌逆战 | Never.Ling
# Date: 2022/5/28
"""
基于Josh McDermott的Matlab滤波器组代码:
https://github.com/wil-j-wil/py_bank
https://github.com/flavioeverardo/erb_bands
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

class EquivalentRectangularBandwidth():
    def __init__(self, nfreqs, sample_rate, total_erb_bands, low_freq, max_freq):
        if low_freq == None:
            low_freq = 20
        if max_freq == None:
            max_freq = sample_rate // 2
        freqs = np.linspace(0, max_freq, nfreqs)  # 每个STFT频点对应多少Hz
        self.EarQ = 9.265  # _ERB_Q
        self.minBW = 24.7  # minBW
        # 在ERB刻度上建立均匀间隔
        erb_low = self.freq2erb(low_freq)  # 最低 截止频率
        erb_high = self.freq2erb(max_freq)  # 最高 截止频率
        # 在ERB频率上均分为（total_erb_bands + ）2个 频带
        erb_lims = np.linspace(erb_low, erb_high, total_erb_bands + 2)
        cutoffs = self.erb2freq(erb_lims)  # 将 ERB频率再转到 hz频率, 在线性频率Hz上找到ERB截止频率对应的频率
        # self.nfreqs  F
        # self.freqs # 每个STFT频点对应多少Hz
        self.filters = self.get_bands(total_erb_bands, nfreqs, freqs, cutoffs)

    def freq2erb(self, frequency):
        """ [Hohmann2002] Equation 16"""
        return self.EarQ * np.log(1 + frequency / (self.minBW * self.EarQ))

    def erb2freq(self, erb):
        """ [Hohmann2002] Equation 17"""
        return (np.exp(erb / self.EarQ) - 1) * self.minBW * self.EarQ

    def get_bands(self, total_erb_bands, nfreqs, freqs, cutoffs):
        """
        获取erb bands、索引、带宽和滤波器形状
        :param erb_bands_num: ERB 频带数
        :param nfreqs: 频点数 F
        :param freqs: 每个STFT频点对应多少Hz
        :param cutoffs: 中心频率 Hz
        :param erb_points: ERB频带界限 列表
        :return:
        """
        cos_filts = np.zeros([nfreqs, total_erb_bands])  # (F, ERB)
        for i in range(total_erb_bands):
            lower_cutoff = cutoffs[i]  # 上限截止频率 Hz
            higher_cutoff = cutoffs[i + 2]  # 下限截止频率 Hz, 相邻filters重叠50%

            lower_index = np.min(np.where(freqs > lower_cutoff))  # 下限截止频率对应的Hz索引 Hz。np.where 返回满足条件的索引
            higher_index = np.max(np.where(freqs < higher_cutoff))  # 上限截止频率对应的Hz索引
            avg = (self.freq2erb(lower_cutoff) + self.freq2erb(higher_cutoff)) / 2
            rnge = self.freq2erb(higher_cutoff) - self.freq2erb(lower_cutoff)
            cos_filts[lower_index:higher_index + 1, i] = np.cos(
                (self.freq2erb(freqs[lower_index:higher_index + 1]) - avg) / rnge * np.pi)  # 减均值，除方差

        # 加入低通和高通，得到完美的重构
        filters = np.zeros([nfreqs, total_erb_bands + 2])  # (F, ERB)
        filters[:, 1:total_erb_bands + 1] = cos_filts
        # 低通滤波器上升到第一个余cos filter的峰值
        higher_index = np.max(np.where(freqs < cutoffs[1]))  # 上限截止频率对应的Hz索引
        filters[:higher_index + 1, 0] = np.sqrt(1 - np.power(filters[:higher_index + 1, 1], 2))
        # 高通滤波器下降到最后一个cos filter的峰值
        lower_index = np.min(np.where(freqs > cutoffs[total_erb_bands]))
        filters[lower_index:nfreqs, total_erb_bands + 1] = np.sqrt(
            1 - np.power(filters[lower_index:nfreqs, total_erb_bands], 2))
        return cos_filts


if __name__ == "__main__":
    fs = 16000
    NFFT = 512  # 信号长度
    ERB_num = 20
    low_lim = 20  # 最低滤波器中心频率
    high_lim = fs / 2  # 最高滤波器中心频率

    freq_num = NFFT // 2 + 1
    fs_bin = fs // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, freq_num, freq_num)

    # ================ 画三角滤波器 ===========================
    ERB = EquivalentRectangularBandwidth(freq_num, fs, ERB_num, low_lim, high_lim)
    filterbanks = ERB.filters.T  # (257, 20)
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