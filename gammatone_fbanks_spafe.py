# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/24
"""
Gammatone-filter-banks implementation
参考自：https://github.com/SuperKogito/spafe，
from spafe.features.gfcc import gfcc
而spage参考下文
based on https://github.com/mcusi/gammatonegram/
"""
import librosa
import numpy as np


class GammatoneFilterBank():
    def __init__(self, nfilter=22, nfft=512, sr=16000, low_freq=None, high_freq=None, scale="constant", order=4):
        """ 计算Gammatone-filterbanks, (G,F)
        :param nfilter: filterbank中滤波器的数量 (Default 22)
        :param nfft: FFT size (Default is 512)
        :param fs: 采样率 (Default 16000 Hz)
        :param low_freq: 最低频带 (Default 0 Hz)
        :param high_freq: 最高频带 (Default sr/2)
        :param scale: 选择Max bins 幅度 "ascend"(上升)，"descend"(下降)或 "constant"(恒定)(=1)。默认是"constant"
        :param order: 滤波器阶数
        :return: 一个大小为(nfilts, nfft/2 + 1)的numpy数组，包含滤波器组。
        """
        # Slaney's ERB Filter constants
        self.EarQ = 9.26449
        self.minBW = 24.7
        self.nfilter = nfilter
        self.nfft = nfft
        self.sr = sr
        self.low_freq = low_freq or 0
        self.high_freq = high_freq or sr / 2
        self.scale = scale
        self.order = order

    def generate_center_frequencies(self, min_freq, max_freq, filter_nums):
        """ 计算中心频率(ERB scale)
        :param min_freq: 中心频率域的最小频率。
        :param max_freq: 中心频率域的最大频率。
        :param filter_nums: 滤波器的个数，即等于计算中心频率的个数。
        :return: 一组中心频率
        """
        # init vars
        m = np.array(range(self.nfilter)) + 1  # spafe用的是这个
        c = self.EarQ * self.minBW

        # 计算中心频率
        center_freqs = (max_freq + c) * np.exp(
            (m / filter_nums) * (np.log(min_freq + c) - np.log(max_freq + c))
        ) - c

        return center_freqs[::-1]

    def compute_gain(self, fcs, B, wT, T):
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

    def Dif(self, u, a):
        # define custom difference func
        return u - a.reshape(self.nfilter, 1)

    def scale_fbank(self, scale, nfilter):
        """
        Generate scaling vector.

        Args:
            scale  (str) : type of scaling.
            nfilts (int) : number of filters.

        Returns:
            (numpy.ndarray) : scaling vector.

        Note:
            .. math::
                ascendant  : \\frac{1}{nfilts} \\times [ 1, ..., i, ..., nfilts]

                descendant : \\frac{1}{nfilts} \\times [ nfilts, ..., i, ..., 1]
        """
        return {
            "ascendant": np.array([i / nfilter for i in range(1, nfilter + 1)]).reshape(
                nfilter, 1
            ),
            "descendant": np.array([i / nfilter for i in range(nfilter, 0, -1)]).reshape(
                nfilter, 1
            ),
            "constant": np.ones(shape=(nfilter, 1)),
        }[scale]

    def get_fbank(self):
        # init vars
        fbank = np.zeros([self.nfilter, self.nfft // 2 + 1])  # (g,F)
        width = 1.0
        T = 1 / self.sr
        n = 4
        u = np.exp(1j * 2 * np.pi * np.array(range(self.nfft // 2 + 1)) / self.nfft)
        idx = range(self.nfft // 2 + 1)

        fcs = self.generate_center_frequencies(self.low_freq, self.high_freq, self.nfilter)  # 计算中心频率，转换到ERB scale
        ERB = width * ((fcs / self.EarQ) ** self.order + self.minBW ** self.order) ** (1 / self.order)  # 计算带宽
        B = 1.019 * 2 * np.pi * ERB

        # compute input vars
        wT = 2 * fcs * np.pi * T
        pole = np.exp(1j * wT) / np.exp(B * T)

        # compute alpha and A matrix
        A, Gain = self.compute_gain(fcs, B, wT, T)

        # compute fbank
        fbank[:, idx] = (
                (T ** 4 / Gain.reshape(self.nfilter, 1)) *
                np.abs(self.Dif(u, A[0]) * self.Dif(u, A[1]) * self.Dif(u, A[2]) * self.Dif(u, A[3])) *
                np.abs(self.Dif(u, pole) * self.Dif(u, pole.conj())) ** (-n))

        # 确保所有filters的最大值为1.0
        try:
            fbank = np.array([f / np.max(f) for f in fbank[:, idx]])
        except BaseException:
            fbank = fbank[:, idx]

        # compute scaler
        scaling = self.scale_fbank(scale=self.scale, nfilter=self.nfilter)

        fbank = fbank * scaling
        return fbank


if __name__ == "__main__":
    import librosa.display
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

    nfilts = 22
    NFFT = 512
    fs = 16000
    wav = librosa.load("../wav_data/p225_001.wav", sr=fs)[0]
    S = librosa.stft(wav, n_fft=NFFT, hop_length=NFFT // 2, win_length=NFFT, window="hann", center=False)
    mag = np.abs(S)  # 幅度谱 (257, 127) librosa.magphase()
    filterbanks = GammatoneFilterBank(nfilter=nfilts, nfft=NFFT, sr=fs,
                                      low_freq=0, high_freq=None)
    filterbank = filterbanks.get_fbank()  # (M,F) (20, 257)
    # ================ 画三角滤波器 ===========================
    FFT_len = NFFT // 2 + 1
    fs_bin = fs // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, FFT_len, FFT_len)

    plt.plot(x * fs_bin, filterbank.T)
    # plt.xlim(0)  # 坐标轴的范围
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()

    FBank_spec = np.dot(filterbank, mag)  # (M,F)*(F,T)=(M,T)
    recover_spec = np.matmul(filterbank.transpose((1, 0)), FBank_spec)  # (F,T)

    log_FBank_spec = 20 * np.log10(FBank_spec)  # dB
    log_recover_FBank_spec = 20 * np.log10(recover_spec)  # dB
    print("mag", np.mean(mag ** 2))  # 2.2741268
    print("recover_spec", np.mean(recover_spec ** 2))  # 20.895620527735336
    print("恢复和原始的均方差：", np.mean((recover_spec - mag) ** 2))  # 16.299886740990214

    # ================ 绘制语谱图 ==========================
    plt.figure()
    librosa.display.specshow(log_FBank_spec, sr=fs, x_axis='time', y_axis='linear', cmap="jet")
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
