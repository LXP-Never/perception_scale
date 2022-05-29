# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/21
"""

"""
import numpy as np
import librosa
import erb as erb
import matplotlib.pyplot as plt

sr = 16000
len_signal=8000   # 信号长度
ERB_num = 20
low_lim = 20  # 最低滤波器中心频率
high_lim = sr / 2  # 最高滤波器中心频率
erb_bank = erb.EquivalentRectangularBandwidth(len_signal, sr, ERB_num, low_lim, high_lim)

plt.plot(erb_bank.filters)
plt.show()





