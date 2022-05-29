# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/26
"""

"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

# init vars
F0 = 0
FSP = 200 / 3
BARK_FREQ = 1000
BARK_PT = (BARK_FREQ - F0) / FSP
LOGSTEP = np.exp(np.log(6.4) / 27.0)


def hz2mel(hz, htk=1):
    """
    Convert a value in Hertz to Mels

    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
         htk: Optional variable, if htk = 1 uses the mel axis defined in the HTKBook otherwise use Slaney's formula.

    Returns:
        a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    if htk == 1:
        return 2595 * np.log10(1 + hz / 700.)
    else:
        # format variable
        hz = np.array(hz, ndmin=1)

        # definee lambda functions to simplify code
        def e(i):
            return (hz[i] - F0) / FSP

        def g(i):
            return BARK_PT + (np.log(hz[i] / BARK_FREQ) / np.log(LOGSTEP))

        mel = [e(i) if hz[i] < BARK_PT else g(i) for i in range(hz.shape[0])]
        return np.array(mel)


def mel2hz(mel, htk=1):
    """
    Convert a value in Mels to Hertz

    Args:
        mel: a value in Mels. This can also be a numpy array, conversion
             proceeds element-wise.
        htk: Optional variable, if htk = 1 uses the mel axis defined in the
             HTKBook otherwise use Slaney's formula.

    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    if htk == 1:
        return 700 * (10 ** (mel / 2595.0) - 1)
    else:
        # format variable
        mel = np.array(mel, ndmin=1)

        # definee lambda functions to simplify code
        def e(i):
            return F0 + FSP * mel[i]

        def g(i):
            return BARK_FREQ * np.exp(np.log(LOGSTEP) * (mel[i] - BARK_PT))

        f = [e(i) if mel[i] < BARK_PT else g(i) for i in range(mel.shape[0])]
        return np.array(f)


def mel_filter_banks(nfilts=20, nfft=512, fs=16000, low_freq=0, high_freq=None, scale="constant"):
    """
    Compute Mel-filterbanks.The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilts    (int) : the number of filters in the filterbank.
                          (Default 20)
        nfft      (int) : the FFT size.
                          (Default is 512)
        fs        (int) : sample rate/ sampling frequency of the signal.
                          (Default 16000 Hz)
        low_freq  (int) : lowest band edge of mel filters.
                          (Default 0 Hz)
        high_freq (int) : highest band edge of mel filters.
                          (Default samplerate/2)
        scale    (str)  : choose if mx bins amplitudes sum up to one or are constants.
                          Default is "constant"

    Returns:
        a numpy array of size nfilts * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # compute points evenly spaced in mels (ponts are in Hz)
    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, nfilts + 2)

    # we use fft bins, so we have to convert from Hz to fft bin number
    bins = np.floor((nfft + 1) * mel2hz(mel_points) / fs)
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    # init scaler
    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c = 0

    # 计算fbanks的幅度(amps)
    for j in range(0, nfilts):
        b0, b1, b2 = bins[j], bins[j + 1], bins[j + 2]

        # compute scaler
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)
        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        # compute fbank bins
        fbank[j, int(b0):int(b1)] = c * (np.arange(int(b0), int(b1)) - int(b0)) / (b1 - b0)
        fbank[j, int(b1):int(b2)] = c * (int(b2) - np.arange(int(b1), int(b2))) / (b2 - b1)

    return fbank


def inverse_mel_filter_banks(nfilts=20,
                             nfft=512,
                             fs=16000,
                             low_freq=0,
                             high_freq=None,
                             scale="constant"):
    """
    Compute inverse Mel-filterbanks. The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilt     (int) : the number of filters in the filterbank.
                          (Default 20)
        nfft      (int) : the FFT size.
                          (Default is 512)
        fs        (int) : sample rate/ sampling frequency of the signal.
                          (Default 16000 Hz)
        low_freq  (int) : lowest band edge of mel filters.
                          (Default 0 Hz)
        high_freq (int) : highest band edge of mel filters.
                          (Default samplerate/2)
        scale    (str)  : choose if mx bins amplitudes sum up to one or are constants.
                          Default is "const"

    Returns:
        a numpy array of size nfilt * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # inverse scaler value
    scales = {
        "ascendant": "descendant",
        "descendant": "ascendant",
        "constant": "constant"
    }
    iscale = scales[scale]
    # generate inverse mel fbanks by inversing regular mel fbanks
    imel_fbanks = mel_filter_banks(nfilts=nfilts,
                                   nfft=nfft,
                                   fs=fs,
                                   low_freq=low_freq,
                                   high_freq=high_freq,
                                   scale=iscale)
    # inverse regular filter banks
    for i, pts in enumerate(imel_fbanks):
        imel_fbanks[i] = pts[::-1]

    return np.abs(imel_fbanks)


fbark = mel_filter_banks(nfilts=20,
                         nfft=512,
                         fs=16000,
                         low_freq=0,
                         high_freq=None,
                         scale="constant")

ifbark = inverse_mel_filter_banks(nfilts=20,
                                  nfft=512,
                                  fs=16000,
                                  low_freq=0,
                                  high_freq=None,
                                  scale="constant")
plt.plot(fbark.T)

plt.show()

plt.plot(ifbark.T)
plt.show()