# -*- coding:utf-8 -*-
# Author:凌逆战 | Never.Ling
# Date: 2022/7/15
"""
不是RNNoise原本的刻度，因为RNNoise刻度最高到400，而有481个频点
本文档主要使用mel刻度，但是使用的是RNNoise的打叉法三角滤波器组

RNNoise特征提取和直接和三角滤波器组相乘结果一样
"""
import numpy as np


class MelFilterBank():
    def __init__(self, nfilter=32, nfft=512, sr=16000,
                 lowfreq=0, highfreq=None, transpose=False):
        """
        :param nfilter: filterbank中的滤波器数量
        :param nfft: FFT size
        :param sr: 采样率
        :param lowfreq: Mel-filter的最低频带边缘
        :param highfreq: Mel-filter的最高频带边缘，默认samplerate/2
        """
        self.nfilter = nfilter
        self.freq_bins = int(nfft / 2 + 1)
        highfreq = highfreq or sr / 2
        self.transpose = transpose

        # 按梅尔均匀间隔计算 点
        lowmel, highmel = self.hz2mel(lowfreq), self.hz2mel(highfreq)
        melpoints = np.linspace(lowmel, highmel, nfilter)
        hz_points = self.mel2hz(melpoints)  # 将mel频率再转到hz频率
        # print("hz_points", hz_points)
        # bin = (sr/2) / (NFFT+/2) =sample_rate/NFFT    # 每个频点的频率数
        # bins_list = hz_points/bin=hz_points/(sample_rate/NFFT)     # hz_points对应第几个fft频点
        self.bins_list = np.floor(hz_points * nfft / sr).astype(np.int32)
        print("self.bins_list", len(self.bins_list), self.bins_list)  # 32
        # [  0   1   3   5   8  10  13  15  18  22  25  29  33  38  42  48  53  59
        #   66  73  80  88  97 107 117 128 140 153 167 182 198 216 235 256]

    def hz2mel(self, hz, approach="Oshaghnessy"):
        """ Hz to Mels """
        return {
            "Oshaghnessy": 2595 * np.log10(1 + hz / 700.0),
            "Lindsay": 2410 * np.log10(1 + hz / 625),
        }[approach]

    def mel2hz(self, mel, approach="Oshaghnessy"):
        """ Mels to HZ """
        return {
            "Oshaghnessy": 700 * (10 ** (mel / 2595.0) - 1),
            "Lindsay": 625 * (10 ** (mel / 2410) - 1),
        }[approach]

    def get_filter_bank(self):
        # RNNoise的三角滤波器，打叉画法
        fbank = np.zeros((self.nfilter, self.freq_bins))  # (M,F)
        for i in range(self.nfilter - 1):
            band_size = (self.bins_list[i + 1] - self.bins_list[i])
            for j in range(band_size):
                frac = j / band_size
                fbank[i, self.bins_list[i] + j] = 1 - frac  # 降
                fbank[i + 1, self.bins_list[i] + j] = frac  # 升
        # 第一个band和最后一个band的窗只有一半因而能量乘以2
        fbank[0] *= 2
        fbank[-1] *= 2
        return fbank

    def interp_band_gain(self, gain):
        # gain (M,T)
        gain_interp = np.zeros((self.freq_bins, gain.shape[-1]))  # (F,T)
        for i in range(self.nfilter - 1):
            band_size = (self.bins_list[i + 1] - self.bins_list[i])
            for j in range(band_size):
                frac = j / band_size
                # gain_interp[eband5ms[i] * 4 + j] = frac * gain[i] + \
                #                                    (1 - frac) * gain[i + 1]
                gain_interp[self.bins_list[i] + j] = (1 - frac) * gain[i] + \
                                                     frac * gain[i + 1]

        return gain_interp


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import librosa.display
    from matplotlib.ticker import FuncFormatter


    def formatnum(x, pos):
        return '$%d$' % (x / 1000)


    formatter = FuncFormatter(formatnum)

    sr = 16000
    band_nums = 32
    NFFT = win_length = 512
    freq_bins = NFFT // 2 + 1
    eband5ms = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100)

    x = librosa.load("../wav_data/clean/TIMIT/TRAIN_DR1_FCJF0_SA1.wav", sr=sr)[0]
    X = librosa.stft(x, n_fft=NFFT, hop_length=win_length // 2,
                     win_length=win_length,
                     window="hann", center=False)  # (F,T)
    mag = np.abs(X)  # (F,T)
    FBank_class = MelFilterBank(nfilter=band_nums, nfft=NFFT,
                                sr=sr, lowfreq=0, highfreq=None, transpose=False)

    Fbank = FBank_class.get_filter_bank()  # (M,F)
    filter_banks = np.dot(Fbank, mag)  # (M,F) * (F,T) = (M,T)

    # ================ 绘制三角滤波器组 ==========================
    fig = plt.figure(figsize=(8, 12))
    fig.suptitle('RNNoise_fbank_nfft={}_filter_nums={}'.format(NFFT, band_nums))

    plt.subplot(3, 1, 1)
    fs_bin = sr // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, freq_bins, freq_bins)
    plt.plot(x, Fbank.T)

    plt.subplot(3, 1, 2)
    librosa.display.specshow(20 * np.log10(filter_banks), sr=sr, x_axis='time', y_axis='linear', cmap="jet")
    plt.xlabel('time/s', fontsize=14)
    plt.ylabel('freqs/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.subplot(3, 1, 3)
    interp_band_gain = FBank_class.interp_band_gain(filter_banks)  # (F,T)
    print("插值结果：", interp_band_gain[:10, 1])
    transpose_band_gain = np.dot(Fbank.T, filter_banks)  # (F,M)*(M*T)=(F,T)
    print("转置相乘：", transpose_band_gain[:10, 1])
    librosa.display.specshow(20 * np.log10(interp_band_gain), sr=sr, x_axis='time', y_axis='linear', cmap="jet")
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('Freqs/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.show()
