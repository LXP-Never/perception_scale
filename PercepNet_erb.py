# -*- coding:utf-8 -*-
# Author:凌逆战 | Never.Ling
# Date: 2022/9/18
"""

"""
import numpy as np
import torch


class ErbFilterBank():
    def __init__(self, nfilter=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None,transpose=False):
        """
        :param nfilter: filterbank中的滤波器数量
        :param nfft: FFT size
        :param samplerate: 采样率
        :param lowfreq: Erb-filter的最低频带边缘
        :param highfreq: Erb-filter的最高频带边缘，默认samplerate/2
        """
        self.nfilter = nfilter
        self.freq_size = int(nfft / 2 + 1)
        highfreq = highfreq or samplerate / 2
        self.transpose = transpose

        # 按梅尔均匀间隔计算 点
        lowerb = self.hz2erb(lowfreq)
        higherb = self.hz2erb(highfreq)
        erbpoints = np.linspace(lowerb, higherb, nfilter + 2)
        self.hz_points = self.erb2hz(erbpoints)  # 将erb频率再转到hz频率
        # print("hz_points", len(self.hz_points), self.hz_points)
        # bin = sr/2 / (NFFT+1)/2=sample_rate/(NFFT+1)    # 每个频点的频率数
        # bins = hz_points/bin=hz_points*NFFT/ sample_rate    # hz_points对应第几个fft频点
        self.bins_list = np.floor((nfft + 1) * self.hz_points / samplerate).astype(np.int)

    def hz2erb(self, hz):
        """ Hz to Erbs """
        return 9.265 * np.log(1+hz/(24.7*9.265))

    def erb2hz(self, n_erb):
        """ Erbs to HZ """
        return 24.7 * 9.265 * (np.exp(n_erb/9.265) -1)

    def get_fbank(self):
        """ 计算一个 Erb-filterbank (M,F) """
        fbank = torch.zeros([self.nfilter, self.freq_size])  # (m,f)
        for i in range(0, self.nfilter):
            for j in range(self.bins_list[i], self.bins_list[i + 1]):
                fbank[i, j] = (j - self.bins_list[i]) / (self.bins_list[i + 1] - self.bins_list[i])
            for j in range(self.bins_list[i + 1], self.bins_list[i + 2]):
                fbank[i, j] = (self.bins_list[i + 2] - j) / (self.bins_list[i + 2] - self.bins_list[i + 1])
        #    fbank -= (np.mean(fbank, axis=0) + 1e-8)
        if self.transpose:
            fbank= fbank.transpose(-1,-2)
        return fbank

    def interp_band_gain_numpy(self, erb_spec):
        """ 将22个点的erb特征，插值成241个点 (M,T)-->(F,T) """
        restore_features = np.zeros((self.freq_size, erb_spec.shape[-1]))  # (F,T)
        for i in range(0, self.nfilter):
            for j in range(self.bins_list[i], self.bins_list[i + 1]):
                side = (j - self.bins_list[i]) / (self.bins_list[i + 1] - self.bins_list[i])  # 升
                restore_features[j] += erb_spec[i] * side
            for k in range(self.bins_list[i + 1], self.bins_list[i + 2]):
                side = (self.bins_list[i + 2] - k) / (self.bins_list[i + 2] - self.bins_list[i + 1])  # 降
                restore_features[k] += erb_spec[i] * side

        return restore_features

    def interp_band_gain(self, erb_spec):
        """ 将22个点的erb特征，插值成241个点 (M,T)-->(F,T) """
        restore_features = torch.zeros((erb_spec.shape[0], self.freq_size, erb_spec.shape[-1]), device=erb_spec.device)  # (F,T)
        for i in range(0, self.nfilter):
            for j in range(self.bins_list[i], self.bins_list[i + 1]):
                side = (j - self.bins_list[i]) / (self.bins_list[i + 1] - self.bins_list[i])  # 升
                restore_features[:,j] += erb_spec[:,i] * side
            for k in range(self.bins_list[i + 1], self.bins_list[i + 2]):
                side = (self.bins_list[i + 2] - k) / (self.bins_list[i + 2] - self.bins_list[i + 1])  # 降
                restore_features[:,k] += erb_spec[:,i] * side

        return restore_features


if __name__ == "__main__":
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from torch import nn
    import torch

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

    sr = 16000
    window_len=512
    nfft=512
    num_filter=22

    erb_filter = ErbFilterBank(nfilter=num_filter, nfft=nfft, samplerate=sr, lowfreq=100, highfreq=None)
    erbfilterbank = erb_filter.get_fbank()
    print(erbfilterbank.shape)    # (M,F)

    plt.plot(erbfilterbank.T)
    plt.show()

    wav = librosa.load("../wav_data/p225_001.wav",sr=sr)[0]
    spec = librosa.stft(wav,n_fft=nfft,hop_length=window_len//2,win_length=window_len)
    mag = np.abs(spec)  # (F,T)

    # 矩阵相乘获得erb频谱 --------------------------------------
    erbfilter_feature = np.dot(erbfilterbank, mag)  # (M,T)
    filter_banks = 20 * np.log10(erbfilter_feature)  # dB

    # 滤波器组作为Linear权重获得erb频谱 --------------------------------------
    # model = nn.Linear(in_features=257,out_features=22,delta=False)   # W(out,in) [22, 257]
    # model.weight = nn.Parameter(erbfilterbank, requires_grad=False)
    # output = model(torch.from_numpy(mag.T))  # (T,F)*(F,M)
    # output = output.T.detach().numpy()   # (M,T)
    # filter_banks = 20 * np.log10(output)  # dB

    # 绘制 频谱图 方法2 --------------------------------------
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

    # 插值频谱恢复 ----------------------------------------------------------------
    # restore = erb_filter.interp_band_gain_numpy(filter_banks)  # (M,T)-->(F,T)
    restore = np.matmul(erbfilterbank.T.numpy(),filter_banks)


    print("restore", restore.shape)    # (F,T)(257, 4)
    librosa.display.specshow(restore, sr=sr, x_axis='time', y_axis='linear', cmap="jet")
    plt.show()
