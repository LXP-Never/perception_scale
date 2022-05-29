# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/25
"""
https://github.com/kingback2019/Speech_MFCC_GFCC_Python/blob/main/mfcc_extractor.py
和GFCC_feature_1很像，无法看到FilterBank, 但是可以提取GFCC特征
"""
import librosa
from scipy.signal import lfilter, lfilter_zi, lfiltic

import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt


def get_window(win_len, win_type):
    if win_type == 'hanning':
        win_len += 2
        window = np.hanning(win_len)
        window = window[1: -1]
    elif win_type == 'hamming':
        win_len += 2
        window = np.hamming(win_len)
        window = window[1: -1]
    elif win_type == 'triangle':
        window = 1. - (np.abs(win_len + 1. - 2. * np.arange(0., win_len + 2., 1.)) / (win_len + 1.))
        window = window[1: -1]
    else:
        window = np.ones(win_len)
    return window


def erb_space(low_freq=50, high_freq=8000, n=64):
    ear_q = 9.26449
    min_bw = 24.7

    cf_array = -(ear_q * min_bw) + np.exp(
        np.linspace(1, n, n) * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw)) / n) \
               * (high_freq + ear_q * min_bw)
    return cf_array


def make_erb_filters(sr, num_channels, low_freq):
    t = 1. / sr
    cf = erb_space(low_freq, sr // 2, num_channels)

    ear_q = 9.26449
    min_bw = 24.7
    order = 4

    erb = np.power(np.power(cf / ear_q, order) + (min_bw ** order), 1. / order)
    b = 1.019 * 2 * np.pi * erb

    a0 = t
    a2 = 0
    b0 = 1
    b1 = -2 * np.cos(2 * cf * np.pi * t) / np.exp(b * t)
    b2 = np.exp(-2 * b * t)

    a11 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) + 2 * np.sqrt(3 + 2 ** 1.5) * t * np.sin(
        2 * cf * np.pi * t) / np.exp(b * t)) / 2
    a12 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) - 2 * np.sqrt(3 + 2 ** 1.5) * t * np.sin(
        2 * cf * np.pi * t) / np.exp(b * t)) / 2
    a13 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) + 2 * np.sqrt(3 - 2 ** 1.5) * t * np.sin(
        2 * cf * np.pi * t) / np.exp(b * t)) / 2
    a14 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) - 2 * np.sqrt(3 - 2 ** 1.5) * t * np.sin(
        2 * cf * np.pi * t) / np.exp(b * t)) / 2

    p1 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
          (np.cos(2 * cf * np.pi * t) - np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
    p2 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
          (np.cos(2 * cf * np.pi * t) + np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
    p3 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
          (np.cos(2 * cf * np.pi * t) - np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
    p4 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
          (np.cos(2 * cf * np.pi * t) + np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
    p5 = np.power(
        -2 / np.exp(2 * b * t) - 2 * np.exp(4j * cf * np.pi * t) + 2 * (1 + np.exp(4j * cf * np.pi * t)) / np.exp(
            b * t), 4)
    gain = np.abs(p1 * p2 * p3 * p4 / p5)

    allfilts = np.ones((np.size(cf, 0), 1), dtype=np.float32)
    fcoefs = np.column_stack((a0 * allfilts, a11, a12, a13, a14, a2 * allfilts, b0 * allfilts, b1, b2, gain))
    return fcoefs, cf


def erb_frilter_bank(x, fcoefs):
    a0 = fcoefs[:, 0]
    a11 = fcoefs[:, 1]
    a12 = fcoefs[:, 2]
    a13 = fcoefs[:, 3]
    a14 = fcoefs[:, 4]
    a2 = fcoefs[:, 5]
    b0 = fcoefs[:, 6]
    b1 = fcoefs[:, 7]
    b2 = fcoefs[:, 8]
    gain = fcoefs[:, 9]

    output = np.zeros((np.size(gain, 0), np.size(x, 0)))

    for chan in range(np.size(gain, 0)):
        y1 = lfilter(np.array([a0[chan] / gain[chan], a11[chan] / gain[chan], a2[chan] / gain[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), x)
        y2 = lfilter(np.array([a0[chan], a12[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y1)
        y3 = lfilter(np.array([a0[chan], a13[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y2)
        y4 = lfilter(np.array([a0[chan], a14[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y3)

        output[chan, :] = y4
    return output


def cochleagram_extractor(xx, sr, win_len, shift_len, channel_number, win_type):
    fcoefs, f = make_erb_filters(sr, channel_number, 50)
    fcoefs = np.flipud(fcoefs)
    xf = erb_frilter_bank(xx, fcoefs)

    window = get_window(win_len, win_type)
    window = window.reshape((1, win_len))

    xe = np.power(xf, 2.0)
    frames = 1 + ((np.size(xe, 1) - win_len) // shift_len)
    cochleagram = np.zeros((channel_number, frames))
    for i in range(frames):
        one_frame = np.multiply(xe[:, i * shift_len:i * shift_len + win_len], np.repeat(window, channel_number, 0))
        cochleagram[:, i] = np.sqrt(np.mean(one_frame, 1))

    cochleagram = np.where(cochleagram == 0.0, np.finfo(float).eps, cochleagram)
    cochleagram = np.power(cochleagram, 1. / 3)
    return cochleagram


def gfcc_extractor(cochleagram, gf_channel, cc_channels):
    dctcoef = np.zeros((cc_channels, gf_channel))
    for i in range(cc_channels):
        n = np.linspace(0, gf_channel - 1, gf_channel)
        dctcoef[i, :] = np.cos((2 * n + 1) * i * np.pi / (2 * gf_channel))
    plt.figure()
    plt.imshow(dctcoef)
    plt.show()
    return np.matmul(dctcoef, cochleagram)


if __name__ == '__main__':
    # wav_data, wav_header = read_sphere_wav(u"clean.wav")
    # sr, wav_data = wavfile.read(u"clean.wav")
    wav_data = librosa.load("../p225_001.wav", sr=16000)[0]
    sr = 16000
    cochlea = cochleagram_extractor(wav_data, sr, 320, 160, 64, 'hanning')
    gfcc = gfcc_extractor(cochlea, 64, 31)
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.imshow(np.flipud(cochlea))
    plt.subplot(212)
    plt.imshow(np.flipud(gfcc))
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(gfcc[0, :])
    plt.show()
