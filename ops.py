# -*- coding:utf-8 -*-
# Author:凌逆战 | Never.Ling
# Date: 2022/5/29
"""

"""
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def hz2mel(hz):
    """ Hz to Mels """
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    """ Mels to HZ """
    return 700 * (10 ** (mel / 2595.0) - 1)