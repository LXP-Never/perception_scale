# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/20
"""
RNNoise的方法 通过mel滤波器组特征提取 频谱
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

wav_path = "./wav_data/p225_001.wav"
sr = 16000
NFFT = 960
win_length = 960
num_filter = 22
low_freq_mel = 0

wav = librosa.load(wav_path, sr=sr)[0]
S = librosa.stft(wav, n_fft=NFFT, hop_length=NFFT // 2, win_length=win_length, window="hann", center=False)
mag = np.abs(S)  # 幅度谱 (F,T)(481, 67) librosa.magphase()

# 0  200 400 600 800  1k 1.2 1.4 1.6  2k  2.4 2.8 3.2 4k  4.8 5.6 6.8  8k  9.6  12k 15.6 20k
band = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100]  # 22个点

fbank = np.zeros((num_filter, int(np.floor(NFFT / 2 + 1))))  # (22,257)

# 方法1：设计三角滤波器
# 打叉画法(MFCC中的滤波器组是打叉画法)
for i in range(0, num_filter - 1):
    band_size = (band[i + 1] - band[i]) * 4
    print("band_size", band_size)
    for j in range(band_size):
        frac = j / band_size
        fbank[i, band[i] * 4 + j] = 1 - frac  # 降
        fbank[i + 1, band[i] * 4 + j] = frac  # 升
# 第一个band和最后一个band的窗只有一半因而能量乘以2
fbank[0] *= 2
fbank[-1] *= 2

plt.plot(fbank.T)
plt.show()

filter_banks = np.dot(fbank, mag)  # (m,T)
# filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # 数值稳定性
filter_banks = 20 * np.log10(filter_banks)  # dB

plt.figure()
librosa.display.specshow(filter_banks, sr=sr, x_axis='time', y_axis='linear', cmap="jet")
plt.xlabel('Time/s', fontsize=14)
plt.ylabel('Freq/kHz', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# ======================= 特征还原 =================================
restore_features = np.zeros(mag.shape)  # (F,T)
print(restore_features.shape)
for i in range(0, num_filter - 1):
    band_size = (band[i + 1] - band[i]) * 4

    for j in range(band_size):
        frac = j / band_size
        restore_features[band[i] * 4 + j] = (1 - frac) * filter_banks[i] + frac * filter_banks[i + 1]

plt.figure()
librosa.display.specshow(restore_features, sr=sr, x_axis='time', y_axis='linear', cmap="jet")
plt.xlabel('Time/s', fontsize=14)
plt.ylabel('Freq/kHz', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()
