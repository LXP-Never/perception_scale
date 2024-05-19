# -*- coding:utf-8 -*-
# Author:凌逆战 | Never.Ling
# Date: 2022/9/12
"""
# from spafe.fbanks.bark_fbanks.bark_filter_banks
"""
import numpy as np


class BarkFilterBank():
    def __init__(self, nfilter=20, nfft=512, sr=16000, lowfreq=0, highfreq=None, transpose=False, scale="constant"):
        """
        :param nfilter:  滤波器组中滤波器的数量 (Default 20)
        :param nfft: FFT size.(Default is 512)
        :param sr: 采样率，(Default 16000 Hz)
        :param lowfreq: MEL滤波器的最低带边。(Default 0 Hz)
        :param highfreq: MEL滤波器的最高带边。(Default sr/2)
        :param transpose:
        :param scale: 选择Max bins 幅度 "ascend"(上升)，"descend"(下降)或 "constant"(恒定)(=1)。默认是"constant"
        """
        self.nfilter = nfilter
        self.nfft = nfft
        self.sr = sr
        # init freqs
        high_freq = highfreq or sr / 2
        low_freq = lowfreq or 0
        self.scale = scale
        self.transpose = transpose

        # 按Bark scale 均匀间隔计算点数(点数以Bark为单位)
        low_bark = self.hz2bark(low_freq)
        high_bark = self.hz2bark(high_freq)
        self.bark_points = np.linspace(low_bark, high_bark, nfilter + 4)

        self.bins = np.floor(self.bark2fft(self.bark_points)).astype(np.int32)  # Bark Scale等分布对应的 FFT bin number
        # print("self.bins", self.bins)

        # init scaler
        if scale == "descendant" or scale == "constant":
            self.c = 1
        else:
            self.c = 0

    def hz2bark(self, f):
        """ Hz to bark频率 (Wang, Sekey & Gersho, 1992.) """
        return 6. * np.arcsinh(f / 600.)

    def bark2hz(self, fb):
        """ Bark频率 to Hz """
        return 600. * np.sinh(fb / 6.)

    def fft2hz(self, fft, fs=16000, nfft=512):
        """ FFT频点 to Hz """
        return (fft * fs) / (nfft + 1)

    def hz2fft(self, fb, fs=16000, nfft=512):
        """ 频率 to FFT频点 """
        return (nfft + 1) * fb / fs

    def fft2bark(self, fft, fs=16000, nfft=512):
        """ FFT频点 to Bark频率 """
        return self.hz2bark((fft * fs) / (nfft + 1))

    def bark2fft(self, fb, fs=16000, nfft=512):
        """ Bark频率 to FFT频点 """
        # bin = sample_rate/2 / nfft/2=sample_rate/nfft    # 每个频点的频率数
        # bins = hz_points/bin=hz_points*nfft/ sample_rate    # hz_points对应第几个fft频点
        return (nfft + 1) * self.bark2hz(fb) / fs

    def Fm(self, fb, fc):
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

    def get_fbank(self):
        fbank = np.zeros([self.nfilter, self.nfft // 2 + 1])  # (B,F)
        for i in range(0, self.nfilter):  # --> B
            # compute scaler
            if self.scale == "descendant":
                self.c -= 1 / self.nfilter
                self.c = self.c * (self.c > 0) + 0 * (self.c < 0)
            elif self.scale == "ascendant":
                self.c += 1 / self.nfilter
                self.c = self.c * (self.c < 1) + 1 * (self.c > 1)

            for j in range(self.bins[i], self.bins[i + 4]):  # --> F
                fc = self.bark_points[i + 2]  # 中心频率
                fb = self.fft2bark(j)  # Bark 频率
                fbank[i, j] = self.c * self.Fm(fb, fc)
        return np.abs(fbank)


if __name__ == "__main__":
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter


    def formatnum(x, pos):
        return '$%d$' % (x / 1000)


    formatter = FuncFormatter(formatnum)

    nfilts = 32
    NFFT = 512
    sr = 16000
    wav = librosa.load("../wav_data/TIMIT.wav", sr=sr)[0]
    S = librosa.stft(wav, n_fft=NFFT, hop_length=NFFT // 2, win_length=NFFT, window="hann", center=False)
    mag = np.abs(S)  # 幅度谱 (F,T)

    filterbanks = BarkFilterBank(nfilter=nfilts, nfft=NFFT, sr=sr, lowfreq=0, highfreq=None)
    bark_fbanks = filterbanks.get_fbank()  # (M,F)
    Fbank_feature = np.dot(bark_fbanks, mag)  # (M,F)*(F,T)=(M,T)
    inv_feature = np.dot(Fbank_feature.T, bark_fbanks)  # (T,M)*(M,F)=(T,F)

    print("bark_fbanks", bark_fbanks.shape)
    # ================ 画三角滤波器 ===========================
    fig = plt.figure(figsize=(8, 12))
    fig.suptitle('bark_fbank_nfft={}_num_filter={}'.format(NFFT, nfilts))

    plt.subplot(3, 1, 1)
    freqs_num = NFFT // 2 + 1
    fs_bin = sr // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, freqs_num, freqs_num)
    plt.plot(x * fs_bin, bark_fbanks.T)
    plt.grid(linestyle='--')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(20 * np.log10(Fbank_feature), sr=sr, x_axis='time', y_axis='linear', cmap="jet")
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('Freqs/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.subplot(3, 1, 3)
    librosa.display.specshow(20 * np.log10(inv_feature.T), sr=sr, x_axis='time', y_axis='linear', cmap="jet")
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('Freqs/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.show()
