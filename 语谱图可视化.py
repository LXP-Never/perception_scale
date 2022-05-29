from scipy import interpolate
from scipy.signal import decimate
import librosa
from scipy import signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

wb_file_name = "./VCTK_p225/GMM/r=2/label/p225_355_WB.wav"
nb_file_name = "./VCTK_p225/GMM/r=2/logits/p225_355_pre.wav"

y_wb, _ = librosa.load(wb_file_name, sr=16000, mono=True)

nb_wav = librosa.core.resample(y_wb, 16000, 8000)  # 下采样率 16000-->8000
nb_wav = librosa.core.resample(nb_wav, 8000, 16000)  # 上采样率 8000-->16000，并不恢复高频部分

y_pre, _ = librosa.load(nb_file_name, sr=16000, mono=True)

nb_wav = nb_wav[23500:]  # 23500
y_wb = y_wb[23500:]
y_pre = y_pre[23500:]

# ###################
fig = plt.figure(figsize=(10, 5))  # (15,7)
gca = plt.gca()
# gca.set_position([0.1, 0.1, 0.9, 0.9])
norm = matplotlib.colors.Normalize(vmin=-200, vmax=-40)
plt.subplot(1, 3, 1)
plt.title("窄带音频", fontsize=15)
plt.specgram(nb_wav, Fs=16000, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
plt.xlabel('时间/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.subplots_adjust(top=0.948, bottom=0.110, left=0.081, right=0.915)
plt.subplots_adjust(hspace=0.474, wspace=0.440)  # 调整子图间距

plt.subplot(1, 3, 2)
plt.title("宽带音频", fontsize=15)
plt.specgram(y_wb, Fs=16000, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
plt.xlabel('时间/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.subplots_adjust(top=0.948, bottom=0.110, left=0.081, right=0.915)
plt.subplots_adjust(hspace=0.474, wspace=0.440)  # 调整子图间距

plt.subplot(1, 3, 3)
plt.title("重构宽带音频", fontsize=15)
plt.specgram(y_pre, Fs=16000, scale_by_freq=True, sides='default', cmap="jet", norm=norm)
plt.subplots_adjust(top=0.948, bottom=0.110, left=0.081, right=0.915)
plt.subplots_adjust(hspace=0.474, wspace=0.440)  # 调整子图间距
plt.xlabel('时间/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.subplots_adjust(top=0.948, bottom=0.110, left=0.081, right=0.915)
plt.subplots_adjust(hspace=0.474, wspace=0.440)  # 调整子图间距

l = 0.93  # 左边
b = 0.115  # 底部
w = 0.009  # 右
h = 0.82  # 高
# 对应 l,b,w,h；设置colorbar位置；
rect = [l, b, w, h]
cbar_ax = fig.add_axes(rect)
plt.colorbar(norm=norm, cax=cbar_ax, format="%+2.f dB")  # -200 -50
# plt.tight_layout()

plt.show()
