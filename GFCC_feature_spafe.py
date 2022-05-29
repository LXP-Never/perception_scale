# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/25
"""
参考：spafe
https://github.com/SuperKogito/spafe/blob/master/spafe/fbanks/gammatone_fbanks.py
"""
#############################################################################################
#                           Gammatone-filter-banks implementation
#############################################################################################
import numpy as np
import librosa
from matplotlib import pyplot as plt
from spafe.features.gfcc import gfcc
import librosa.display

if __name__ == "__main__":
    sr = 16000
    y = librosa.load("./p225_001.wav", sr)[0]
    gfccs = gfcc(y, fs=16000, num_ceps=13, pre_emph=0, pre_emph_coeff=0.97, win_len=0.025, win_hop=0.01,
                 win_type='hamming', nfilts=26, nfft=512, low_freq=None, high_freq=None,
                 scale='constant', dct_type=2, use_energy=False, lifter=22, normalize=1)

    # Displaying  the MFCCs:
    # librosa.display.specshow(gfccs, sr=sr, x_axis='time')

    plt.imshow(gfccs, cmap="jet", aspect='auto')
    ax = plt.gca()  # 获取其中某个坐标系
    ax.invert_yaxis()  # 将y轴反转
    plt.show()
