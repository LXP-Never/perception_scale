# -*- coding:utf-8 -*-
# Author:凌逆战 | Never.Ling
# Date: 2022/9/11
"""
f, t, zxx = signal.stft(wav_preprocessed, fs, window, nperseg, noverlap=nperseg / 2)
"""
import numpy as np
import scipy
import torch


class MelFilterBank():
    def __init__(self, nfilter=20, nfft=512, sr=16000, fmin=0, fmax=None, transpose=False):
        """
        :param nfilter: filterbank中的滤波器数量
        :param nfft: FFT size
        :param sr: 采样率
        :param fmin: Mel-filter的最低频带边缘
        :param highfreq: Mel-filter的最高频带边缘，默认samplerate/2
        """
        self.nfilter = nfilter
        self.freq_bins = nfft // 2 + 1  # 频点数
        fmax = fmax or sr / 2
        self.transpose = transpose

        # 按梅尔均匀间隔计算 点
        lowmel, highmel = self.hz2mel(fmin), self.hz2mel(fmax)
        melpoints = np.linspace(lowmel, highmel, nfilter + 2)
        self.hz_points = self.mel2hz(melpoints)  # 将mel频率再转到hz频率
        # print("hz_points", len(self.hz_points), self.hz_points)
        # bin = sr/2 / (NFFT+1)/2=sample_rate/(NFFT+1)    # 每个频点的频率数
        # bins = hz_points/bin=hz_points*NFFT/ sample_rate    # hz_points对应第几个fft频点
        self.bins_list = np.floor((nfft + 1) * self.hz_points / sr).astype(np.int32)
        # print("self.bins_list", len(self.bins_list), self.bins_list)

    def hz2mel(self, hz):
        """ Hz to Mels """
        return 2595 * np.log10(1 + hz / 700.0)

    def mel2hz(self, mel):
        """ Mels to HZ """
        return 700 * (10 ** (mel / 2595.0) - 1)

    def get_fbank(self):
        """ 计算一个Mel-filterbank (M,F) """
        fbank = np.zeros([self.nfilter, self.freq_bins], dtype="float32")  # (m,f)
        for i in range(0, self.nfilter):
            for j in range(self.bins_list[i], self.bins_list[i + 1]):
                fbank[i, j] = (j - self.bins_list[i]) / (self.bins_list[i + 1] - self.bins_list[i])
            for j in range(self.bins_list[i + 1], self.bins_list[i + 2]):
                fbank[i, j] = (self.bins_list[i + 2] - j) / (self.bins_list[i + 2] - self.bins_list[i + 1])
        if self.transpose:
            fbank = fbank.transpose(-1, -2)
        return fbank

    def interp_band_gain_numpy(self, mask):
        """ 将22个点的mel特征，插值成241个点 (M,T)-->(F,T) """
        restore_features = np.zeros((self.freq_bins, mask.shape[-1]))  # (F,T)
        for i in range(0, self.nfilter):
            for j in range(self.bins_list[i], self.bins_list[i + 1]):
                side = (j - self.bins_list[i]) / (self.bins_list[i + 1] - self.bins_list[i])  # 升
                restore_features[j] += mask[i] * side
            for k in range(self.bins_list[i + 1], self.bins_list[i + 2]):
                side = (self.bins_list[i + 2] - k) / (self.bins_list[i + 2] - self.bins_list[i + 1])  # 降
                restore_features[k] += mask[i] * side

        return restore_features

    def interp_band_gain(self, mask):
        """ 将22个点的mel特征，插值成241个点 (B,M,T)-->(B,F,T) """
        restore_features = torch.zeros((mask.shape[0], self.freq_bins, mask.shape[-1]), device=mask.device)  # (F,T)
        for i in range(0, self.nfilter):
            for j in range(self.bins_list[i], self.bins_list[i + 1]):
                side = (j - self.bins_list[i]) / (self.bins_list[i + 1] - self.bins_list[i])  # 升
                restore_features[:, j] += mask[:, i] * side
            for k in range(self.bins_list[i + 1], self.bins_list[i + 2]):
                side = (self.bins_list[i + 2] - k) / (self.bins_list[i + 2] - self.bins_list[i + 1])  # 降
                restore_features[:, k] += mask[:, i] * side

        return restore_features


def get_mfcc(FBank_spec, num_cep=13, dct_type=2, norm='ortho', lifter=22):
    """
    :param FBank_spec: (M,T)
    :param num_cep: the number of cepstrum to return, default 13
    :param dct_type:
    :param norm:
    :param lifter: apply a lifter to cepstral. 0 is no lifter. Default is 22.
    由于系数低位的值过小，需要向前“抬升”，增加了高频DCT系数的幅度。源于线性预测分析。
    原文：
                he principal advantage of cepstral coefficients is that they are generally decorrelated and this allows
            diagonal covariances to be used in the HMMs. However, one minor problem with them is that the higher order
            cepstra are numerically quite small and this results in a very wide range of variances when going from the
            low to high cepstral coefficients . HTK does not have a problem with this but for pragmatic reasons such as
            displaying model parameters, flooring variances, etc, it is convenient to re-scale the cepstral coefficients
            to have similar magnitudes. This is done by setting the configuration parameter CEPLIFTER  to some value L
            to lifter the cepstra according to the following formula.

            来源:
                http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node53.html#eceplifter
    :return:
    """
    log_MelSpec = 10 * np.log10(FBank_spec ** 2 + 1e-10)
    mfcc = scipy.fftpack.dct(log_MelSpec, axis=-2, type=dct_type, norm=norm)[:num_cep, :]

    if lifter > 0:
        ncoeff, nframes = mfcc.shape
        # shape lifter for broadcasting
        n = np.arange(ncoeff, dtype=mfcc.dtype)
        LI = np.sin(np.pi * n / lifter)
        LI = librosa.util.expand_to(LI, ndim=FBank_spec.ndim, axes=-2)
        mfcc *= 1 + (lifter / 2) * LI
        return mfcc
    elif lifter == 0:
        return mfcc
    else:
        raise f"MFCC lifter={lifter} must be a non-negative number"


def cal_delta(feat, N=2):
    """
    :param feat: MFCC或一阶差分系数 (T,M)
    :param N: 间隔帧数，一般为1或2; 对于每个帧，根据前面和N帧计算delta特征
    :returns: 差分系数矩阵
    https://github.com/Byshx/Poccala/blob/07d59a595eda27bebd04fd215f9de4f5583a777a/StatisticalModel/AudioProcessing.py
    https://github.com/dannis999/trained_SpeechRecognition/blob/master/speech_features/base.py
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    num_frames = len(feat)
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')  # padded version of feat
    for t in range(num_frames):
        delta_feat[t] = np.dot(np.arange(-N, N + 1),
                               padded[t: t + 2 * N + 1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


def get_delta(mfcc, d1=True, d2=None):
    assert d1 or d2, "d1 and d2 cannot be None at the same time"
    # 计算一阶差分系数
    if d1 is True:
        delta = cal_delta(mfcc)
        coefficient = np.concatenate((mfcc, delta), 1)
        # 计算二阶差分系数
        if d2 is True:
            delta_ = cal_delta(delta)
            coefficient = np.concatenate((coefficient, delta_), 1)
        return coefficient


def delta(feat, N):
    """ 计算差分系数
    :param feat: 标准梅尔倒谱系数或一阶差分系数 (M, T)
    :param N: 间隔帧数，一般为1或2; 对于每个帧，根据前面和N帧计算delta特征
    :returns: 差分系数矩阵
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    feat = feat.T
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')  # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N + 1),
                               padded[t: t + 2 * N + 1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat.T


def Test_Fbank():
    sr = 16000
    num_filter = 40
    NFFT = 512

    mel_filter = MelFilterBank(nfilter=num_filter, nfft=NFFT, sr=sr, fmin=0, fmax=None)
    FBank = mel_filter.get_fbank()  # (M,F)

    plt.title(f'mel_fbank_nfft={NFFT}_num_filter={num_filter}')
    freqs_num = NFFT // 2 + 1
    fs_bin = sr // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, freqs_num, freqs_num)
    plt.plot(x * fs_bin, FBank.T)
    plt.grid(linestyle='--')
    plt.show()


def Test_interp_band():
    sr = 16000
    num_filter = 40
    window_len = NFFT = 512
    EPS = 1e-10

    mel_filter = MelFilterBank(nfilter=num_filter, nfft=NFFT, samplerate=sr, lowfreq=0, highfreq=None)
    FBank = mel_filter.get_fbank()  # (M,F)
    # filterbank = np.sum(melfilterbank.numpy(), axis=0,keepdims=True)  # (M,F)
    # filterbank = filterbank/np.sum(melfilterbank, axis=0, keepdims=True)
    print("FBank", FBank.shape)

    wav = librosa.load("../wav_data/TIMIT.wav", sr=sr)[0]
    # spec = librosa.stft(wav, n_fft=NFFT, hop_length=window_len // 2, win_length=window_len)
    # mag = np.abs(spec)  # (F,T) (257, 183)
    freq_axis, time_axis, spec = signal.stft(wav, sr, window="hann", nperseg=window_len, noverlap=window_len // 2)
    mag = np.abs(spec)  # (257, 184)

    # 矩阵相乘获得mel频谱 --------------------------------------
    FBank_spec = np.matmul(FBank, mag)  # (M,T)
    FBank_spec_inverse = np.matmul(FBank.T, FBank_spec)  # (F,T)(257, 4)
    FBank_spec_inter = mel_filter.interp_band_gain_numpy(FBank_spec)  # (M,T)-->(F,T)
    print("转置和滤波器插值是否相等", np.allclose(FBank_spec_inverse, FBank_spec_inter))

    fig = plt.figure(figsize=(8, 15))
    fig.suptitle('mel_fbank_nfft={}_num_filter={}'.format(NFFT, num_filter))

    plt.subplot(3, 1, 1)
    plt.title("mag")
    plt.imshow(10 * np.log10(mag ** 2 + EPS), cmap='jet', aspect='auto', origin='lower')  # norm=norm,restore
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('FreqBins/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(3, 1, 2)
    plt.title("FBank_spec")
    plt.imshow(10 * np.log10(FBank_spec ** 2 + EPS), cmap='jet', aspect='auto', origin='lower')  # norm=norm
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('FreqBands/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(3, 1, 3)
    plt.title("FBank_spec_inverse")
    plt.imshow(10 * np.log10(FBank_spec_inverse ** 2 + EPS), cmap='jet', aspect='auto', origin='lower')  # norm=norm
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('FreqBins/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()


def Test_mfcc():
    sr = 16000
    num_filter = 40
    num_cep = 40
    window_len = NFFT = 512

    mel_filter = MelFilterBank(nfilter=num_filter, nfft=NFFT, sr=sr)
    FBank = mel_filter.get_fbank()  # (M,F) (40,257)
    # filterbank = np.sum(melfilterbank.numpy(), axis=0,keepdims=True)  # (M,F)
    # filterbank = filterbank/np.sum(melfilterbank, axis=0, keepdims=True)
    print("FBank", FBank.shape)

    wav = librosa.load("../wav_data/TIMIT.wav", sr=sr)[0]
    spec = librosa.stft(wav, n_fft=NFFT, hop_length=window_len // 2, win_length=window_len)
    mag = np.abs(spec)  # (F,T) (257, 183)

    # 矩阵相乘获得mel频谱 --------------------------------------
    FBank_spec = np.matmul(FBank, mag)  # (M,F)*(F,T)=(M,T)
    mfcc = get_mfcc(FBank_spec, num_cep=num_cep, dct_type=2, norm='ortho', lifter=22)  # (13, 184)
    print("mfcc", mfcc.shape)  # (13, 183)
    # 替换第一个道倒谱系数为对数能量
    mfcc[0] = np.log(np.sum(mag, axis=0) + 1e-10)

    mfcc_d = delta(mfcc, 2)  # 一阶差分
    mfcc_dd = delta(mfcc_d, 2)  # 二阶差分

    # 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵
    wav_feature = np.column_stack((mfcc, mfcc_d, mfcc_dd))

    # mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40, n_mels=40, n_fft=NFFT, hop_length=window_len // 2,
    #                             win_length=window_len)  # (M,T) (13, 184)
    # print("mfcc", mfcc.shape)

    plt.imshow(wav_feature, cmap='jet', aspect='auto', origin='lower')
    plt.show()


if __name__ == "__main__":
    from scipy import signal
    import librosa
    import librosa.feature
    import librosa.display
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import torch

    # Test_Fbank()
    # Test_interp_band()
    Test_mfcc()
