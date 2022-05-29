# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/19
"""
1、提取Mel filterBank
2、提取mel spectrum
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.ticker import FuncFormatter
from ops import hz2mel, mel2hz

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号



def mel_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """计算一个Mel-filterbank (M,F)
    :param nfilt: filterbank中的滤波器数量
    :param nfft: FFT size
    :param samplerate: 采样率
    :param lowfreq: Mel-filter的最低频带边缘
    :param highfreq: Mel-filter的最高频带边缘，默认samplerate/2
    """
    highfreq = highfreq or samplerate / 2

    # 按梅尔均匀间隔计算 点
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    hz_points = mel2hz(melpoints)  # 将mel频率再转到hz频率
    # bin = samplerate/2 / NFFT/2=sample_rate/NFFT    # 每个频点的频率数
    # bins = hz_points/bin=hz_points*NFFT/ sample_rate    # hz_points对应第几个fft频点
    bin = np.floor((nfft + 1) * hz_points / samplerate)

    fbank = np.zeros([nfilt, int(nfft / 2 + 1)])  # (m,f)
    for i in range(0, nfilt):
        for j in range(int(bin[i]), int(bin[i + 1])):
            fbank[i, j] = (j - bin[i]) / (bin[i + 1] - bin[i])  # 升
        for j in range(int(bin[i + 1]), int(bin[i + 2])):
            fbank[i, j] = (bin[i + 2] - j) / (bin[i + 2] - bin[i + 1])  # 降

    #    fbank -= (np.mean(fbank, axis=0) + 1e-8)
    return fbank


wav_path = "./p225_001.wav"
fs = 16000
NFFT = 512
win_length = 512
num_filter = 22
low_freq_mel = 0
high_freq_mel = hz2mel(fs // 2)  # 求最高hz频率对应的mel频率
mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filter + 2)  # 在mel频率上均分成42个点
hz_points = mel2hz(mel_points)  # 将mel频率再转到hz频率
print(hz_points)

# bin = sample_rate/2 / NFFT/2=sample_rate/NFFT    # 每个频点的频率数
# bins = hz_points/bin=hz_points*NFFT/ sample_rate    # hz_points对应第几个fft频点
bins = np.floor((NFFT + 1) * hz_points / fs)
print(bins)
# [  0.   2.   5.   8.  12.  16.  20.  25.  31.  37.  44.  52.  61.  70.
#   81.  93. 107. 122. 138. 157. 178. 201. 227. 256.]

wav = librosa.load(wav_path, sr=fs)[0]
S = librosa.stft(wav, n_fft=NFFT, hop_length=NFFT // 2, win_length=win_length, window="hann", center=False)
mag = np.abs(S)  # 幅度谱 (257, 127) librosa.magphase()

filterbanks = mel_filterbanks(nfilt=num_filter, nfft=NFFT, samplerate=fs, lowfreq=0, highfreq=fs // 2)

# ================ 画三角滤波器 ===========================
FFT_len = NFFT // 2 + 1
fs_bin = fs // 2 / (NFFT // 2)  # 一个频点多少Hz
x = np.linspace(0, FFT_len, FFT_len)

plt.plot(x * fs_bin, filterbanks.T)

plt.xlim(0)  # 坐标轴的范围
plt.ylim(0, 1)
plt.tight_layout()
plt.grid(linestyle='--')
plt.show()

filter_banks = np.dot(filterbanks, mag)  # (M,F)*(F,T)=(M,T)
filter_banks = 20 * np.log10(filter_banks)  # dB

# ================ 绘制语谱图 ==========================
# 绘制 频谱图 方法1
plt.imshow(filter_banks, cmap="jet", aspect='auto')
ax = plt.gca()  # 获取其中某个坐标系
ax.invert_yaxis()  # 将y轴反转
plt.tight_layout()
plt.show()

# 绘制 频谱图 方法2
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

