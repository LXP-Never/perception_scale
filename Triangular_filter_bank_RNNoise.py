# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/20
"""
使用RNNoise的方法提取BFCC特征
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

fbank = np.zeros((num_filter, S.shape[1]))  # (22,257)

for i in range(0, num_filter - 1):
    band_size = (band[i + 1] - band[i]) * 4
    for j in range(band_size):
        frac = j / band_size
        fbank[i] += (1 - frac) * mag[band[i] * 4 + j]  # 降
        fbank[i + 1] += frac * mag[band[i] * 4 + j]  # 升
# 第一个band和最后一个band的窗只有一半因而能量乘以2
fbank[0] *= 2
fbank[-1] *= 2


# 绘制 频谱图 方法1
fbank = 20 * np.log10(fbank)  # dB
plt.imshow(fbank, cmap="jet", aspect='auto', origin='lower')
plt.show()
