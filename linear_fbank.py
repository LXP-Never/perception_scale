# -*- coding:utf-8 -*-
# Author:凌逆战 | Never.Ling
# Date: 2022/9/12
"""
import matplotlib.pyplot as plt
from spafe.fbanks import linear_fbanks

# compute fbanks
fbanks = linear_fbanks.linear_filter_banks(nfilts=24, nfft=512, fs=16000)
"""
import numpy as np


class LinearFilterBank():
    def __init__(self, nfilter=22, nfft=512, sr=16000, lowfreq=0, highfreq=None, scale="constant"):
        self.nfilter = nfilter
        self.nfft = nfft
        self.freq_size = int(nfft / 2 + 1)
        highfreq = highfreq or sr / 2
        self.scale = scale
        # compute points evenly spaced in frequency (points are in Hz)
        delta_hz = (highfreq - lowfreq) / (nfilter + 1)
        self.scale_freqs = lowfreq + delta_hz * np.arange(0, nfilter + 2)   # 每个滤波器在多少Hz
        # print("hz_points", hz_points)
        # bin = sr/2 / (NFFT+1)/2=sample_rate/(NFFT+1)    # 每个频点的频率数
        # bins = hz_points/bin=hz_points*NFFT/ sample_rate    # hz_points对应第几个fft频点
        self.bins_list = np.floor((nfft + 1) * self.scale_freqs / sr).astype(np.int)

    def scale_fbank(self, scale, nfilts):
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
            "ascendant": np.array([i / nfilts for i in range(1, nfilts + 1)]).reshape(
                nfilts, 1
            ),
            "descendant": np.array([i / nfilts for i in range(nfilts, 0, -1)]).reshape(
                nfilts, 1
            ),
            "constant": np.ones(shape=(nfilts, 1)),
        }[scale]

    def get_filterbanks(self):
        fbank = np.zeros((self.nfilter, self.nfft // 2 + 1))
        for i in range(self.nfilter):
            for j in range(self.bins_list[i], self.bins_list[i + 1]):
                fbank[i, j] = (j - self.bins_list[i]) / (self.bins_list[i + 1] - self.bins_list[i])
            for j in range(self.bins_list[i + 1], self.bins_list[i + 2]):
                fbank[i, j] = (self.bins_list[i + 2] - j) / (self.bins_list[i + 2] - self.bins_list[i + 1])

        # compute scaling
        scaling = self.scale_fbank(scale=self.scale, nfilts=self.nfilter)
        fbank = fbank * scaling

        return fbank


if __name__ == "__main__":
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

    sr = 16000
    nfft = 512
    window_len = 512
    num_filter = 22

    linear_filter = LinearFilterBank()
    melfilterbank = linear_filter.get_filterbanks()
    print(melfilterbank.shape)  # (M,F)

    plt.plot(melfilterbank.T)
    plt.show()

    wav = librosa.load("../wav_data/p225_001.wav", sr=sr)[0]
    spec = librosa.stft(wav, n_fft=nfft, hop_length=window_len // 2, win_length=window_len)
    mag = np.abs(spec)  # (F,T)

    melfilter_feature = np.dot(melfilterbank, mag)  # (M,T)
    filter_banks = 20 * np.log10(melfilter_feature)  # dB

    # 绘制 频谱图 方法2
    plt.figure()
    librosa.display.specshow(filter_banks, sr=sr, x_axis='time', y_axis='linear', cmap="jet")
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

    # ----------------------------------------------------------------

    # mag = torch.randn(1,22,4)   # (B,M,T)
    # print("mag", mag.shape)
    # restore = mel_filter.interp_band_gain(mag)  # (B,M,T)-->(B, F,T)
    # print(restore.shape)    # (F,T)(257, 4)
