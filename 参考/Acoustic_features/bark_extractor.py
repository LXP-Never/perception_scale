# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt



def barke(y, Fs, nB=17, nfft=512):
    """
    y: log-STFT of the signal [T,F]
    Fs: 采样率
    nfft: FFT size
    nB: 频带数
    """
    f = np.linspace(0, Fs / 2, nfft / 2 + 1)
    barkScale = bark_bands(f)

    barkIndices = []
    for i in range(0, len(barkScale)):
        barkIndices.append(int(barkScale[i]))
    barkIndices = np.asarray(barkIndices)

    if len(y.shape) > 1:  # In case the signal is framed
        BarkE = []
        for idx_y in y:
            BarkE.append(bark_idx(barkIndices, nB, idx_y))
    else:
        BarkE = bark_idx(barkIndices, nB, y)
    return np.vstack(BarkE)


def bark_bands(f):
    b = []
    for i in range(0, len(f)):
        b.append(13 * math.atan(i * 0.00076) + 3.5 * math.atan((i / 7500) ** 2))  # Bark scale values
    return b


def bark_idx(barkIndices, nB, sig):
    barkEnergy = []
    #    eps = 1e-30
    for i in range(nB):
        brk = np.nonzero(barkIndices == i)
        brk = np.asarray(brk)[0]
        sizeb = len(brk)
        if (sizeb > 0):
            barkEnergy.append(sum(np.abs(sig[brk])) / sizeb)
        else:
            barkEnergy.append(0)
    e = np.asarray(barkEnergy)  # +eps
    return e

x = np.linspace(0,8000,512)

plt.plot(x,bark_bands(x))
plt.show()




