"""
参考：https://github.com/flavioeverardo/erb_bands
"""

import argparse
import os
import numpy as np
import librosa
import erb as erb
import matplotlib.pyplot as plt


# 音频(wav)文件的ERB频带表示
# python main.py --samples=32768 --erb=40  --file=snare.wav
def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=32768, help="FFT大小或样本数量 (1000-32768). Default: 32768.")
    parser.add_argument("--erb", type=int, default=10, help="ERB频带数目(10-100). Default: 40.")
    parser.add_argument("--file", type=str, default="./wav/snare.wav", help="WAV 文件名")
    parser.add_argument('--show-plot', action='store_true', help="保存后显示plot文件")
    return parser.parse_args()


# 检查命令行参数
def check_input(arguments):
    # Check for errors
    if arguments.samples < 1000 or arguments.samples > 32768:
        raise ValueError("""Number of samples requested is out of bounds""")
    if arguments.erb < 10 or arguments.erb > 100:
        raise ValueError("""Number of erb bands requested is out of bounds""")
    if arguments.file == "":
        raise ValueError("""File cannot be emtpy""")

# 获取ERB波段
def main():
    args = parse_params()
    check_input(args)  # 检查命令行参数

    # STFT 参数
    sr = 44100.0  # 采样率
    N = args.samples  # FFT大小或样本数量
    M = N  # 窗长
    H = int(M / 64)  # 帧移
    W = np.hanning(M)  # Window Type
    B = args.erb  # ERB Bands
    low_lim = 20  # 最低滤波器中心频率
    high_lim = sr / 2  # 最高滤波器中心频率

    wav, sr = librosa.load(args.file, sr=sr, mono=True)  # 加载wav文件
    S = librosa.stft(y=wav, n_fft=N, win_length=M, hop_length=H, window='hann')
    mag = np.abs(S) # (F,T)
    mag = mag / np.sum(W)  # 归一化 STFT 输出
    # 频谱 平均 Spectrum Average
    spec_avg = np.mean(mag, axis=1)     # (16385,)
    spec_avg = spec_avg / np.max(spec_avg)  # 归一化
    len_signal = spec_avg.shape[0]  # 滤波器组的长度

    # Equivalent Rectangular Bandwidth
    erb_bank = erb.EquivalentRectangularBandwidth(len_signal, sr, B, low_lim, high_lim)

    freqs_index = erb_bank.freq_index  # 获取频率索引
    freqs = erb_bank.freqs.tolist()  # 获取频率范围
    bandwidths = erb_bank.bandwidths  # 获取频率带宽

    erb_bands = erb_bank.erb_bands  # 获得ERB波段，并将其转换为整数
    erb_bands = list(map(int, erb_bands))

    # 得到ERB/中心频率的振幅
    erb_amp = [0]
    for i in range(len(freqs_index)):
        erb_amp.append(spec_avg[freqs_index[i]])

    # 归一化ERB振幅
    max_erb_amp = max(erb_amp)
    erb_amp = erb_amp / max_erb_amp

    # 获取滤波器
    filters = erb_bank.filters

    # Plot
    plt.figure(figsize=(12, 7))
    plt.subplot(311)
    plt.grid(True)
    plt.plot(freqs, filters[:, 1:-1])
    plt.title("%s Auditory filters" % B)
    plt.xlabel('Frequencies (Hz)')
    plt.ylabel('Linear Amp [0-1]')

    plt.subplot(312)
    plt.grid(True)
    plt.plot(freqs, spec_avg)
    plt.title(" Spectrum")
    plt.xlabel('Frequency')
    plt.xlim(xmin=20)
    plt.ylabel('Linear Amp [0-1]')
    plt.xscale('log')

    plt.subplot(313)
    plt.grid(True)
    plt.plot(erb_amp)
    plt.title(" ERB Scale")
    plt.xlabel('ERB Numbers (1-%s)' % B)
    plt.ylabel('Linear Amp [0-1]')

    plt.tight_layout()

    plt.savefig('%s.png' % os.path.splitext(args.file)[0])
    plt.show()


if __name__ == '__main__':
    main()
