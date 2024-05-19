# -*- coding:utf-8 -*-
# Author:凌逆战 | Never.Ling
# Date: 2022/9/12
"""
基于Josh McDermott的Matlab滤波器组代码:
https://github.com/wil-j-wil/py_bank
https://github.com/flavioeverardo/erb_bands
"""
import numpy as np
import librosa


class CosFilterBank():
    def __init__(self, nfilter, nfft, sr, low_freq=None, max_freq=None):
        self.nfilter = nfilter
        self.nfft = nfft
        if low_freq == None:
            low_freq = 20
        if max_freq == None:
            max_freq = sr // 2
        self.nfreqs = nfft // 2 + 1
        self.freqs_bin = np.linspace(0, max_freq, self.nfreqs)  # 每个STFT频点对应多少Hz
        self.EarQ = 9.265  # _ERB_Q
        self.minBW = 24.7  # minBW
        # 在ERB刻度上建立均匀间隔
        erb_low = self.hz2erb(low_freq)  # 最低 截止频率
        erb_high = self.hz2erb(max_freq)  # 最高 截止频率
        # 在ERB频率上均分为（total_erb_bands + 2个 频带
        self.erb_lims = np.linspace(erb_low, erb_high, nfilter + 2)
        self.cutoffs = self.erb2hz(self.erb_lims)  # 将 ERB频率再转到 hz频率, 在线性频率Hz上找到ERB截止频率对应的频率

    def hz2erb(self, frequency):
        """ [Hohmann2002] Equation 16"""
        return self.EarQ * np.log(1 + frequency / (self.minBW * self.EarQ))

    def erb2hz(self, erb):
        """ [Hohmann2002] Equation 17"""
        return (np.exp(erb / self.EarQ) - 1) * self.minBW * self.EarQ

    def get_fbanks(self):
        cos_filts = np.zeros([self.nfilter, self.nfreqs])  # (ERB, F)
        for i in range(self.nfilter):
            lower_freq = self.cutoffs[i]  # 上限截止频率 Hz
            higher_freq = self.cutoffs[i + 2]  # 下限截止频率 Hz, 相邻filters重叠50%

            lower_index = np.min(np.where(self.freqs_bin > lower_freq))  # 下限截止频率对应的Hz索引 Hz。np.where 返回满足条件的索引
            higher_index = np.max(np.where(self.freqs_bin < higher_freq))  # 上限截止频率对应的Hz索引
            avg = (self.erb_lims[i] + self.erb_lims[i + 2]) / 2
            rnge = self.erb_lims[i + 2] - self.erb_lims[i]
            cos_filts[i, lower_index:higher_index + 1] = np.cos(
                (self.hz2erb(self.freqs_bin[lower_index:higher_index + 1]) - avg) / rnge * np.pi)  # 减均值，除方差
        # 加入低通和高通，得到完美的重构
        filters = np.zeros([self.nfilter + 2, self.nfreqs])  # (ERB, F)
        filters[1: self.nfilter + 1, :] = cos_filts
        # 低通滤波器上升到第一个余cos filter的峰值
        higher_index = np.max(np.where(self.freqs_bin < self.cutoffs[1]))  # 上限截止频率对应的Hz索引
        filters[0, :higher_index + 1] = np.sqrt(1 - filters[1, :higher_index + 1] ** 2)
        # 高通滤波器下降到最后一个cos filter的峰值
        lower_index = np.min(np.where(self.freqs_bin > self.cutoffs[self.nfilter]))
        filters[self.nfilter + 1, lower_index:self.nfreqs] = np.sqrt(
            1 - np.power(filters[self.nfilter, lower_index:self.nfreqs], 2))
        return cos_filts


if __name__ == "__main__":
    import librosa.display
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

    nfilts = 22
    NFFT = 512
    fs = 16000
    wav = librosa.load("./wav_data/p225_001.wav", sr=fs)[0]
    S = librosa.stft(wav, n_fft=NFFT, hop_length=NFFT // 2, win_length=NFFT, window="hann", center=False)
    mag = np.abs(S)  # 幅度谱 (257, 127) librosa.magphase()
    filterbanks = CosFilterBank(nfilter=20, nfft=512, sr=16000)
    bark_fbanks = filterbanks.get_fbanks()
    print("bark_fbanks", bark_fbanks.shape)
    # ================ 画三角滤波器 ===========================
    FFT_len = NFFT // 2 + 1
    fs_bin = fs // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, FFT_len, FFT_len)

    plt.plot(x * fs_bin, bark_fbanks.T)

    # plt.xlim(0)  # 坐标轴的范围
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()

    filter_banks = np.dot(bark_fbanks, mag)  # (M,F)*(F,T)=(M,T)
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
