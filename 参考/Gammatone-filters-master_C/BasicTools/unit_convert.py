import numpy as np
# TODO:

EPS = 1e-20


# erb
def hz2erbscale(freq):
    """convert hz to erb scale"""
    if np.min(4.37*freq/1e3+1) <= 0:
        print(freq)
    return 21.4*np.log10(4.37*freq/1e3+1)


def erbscale2hz(erb_num):
    """convert erb scale to hz"""
    return (10**(erb_num/21.4)-1)/4.37*1e3


# mel
def hz2mel(freq):
    """according hz2mel function of matlab
    """
    mel = 2595*np.log10(1+freq/700)
    return mel


def mel2hz(mel):
    freq = (10**(mel/2595)-1)*700
    return freq
