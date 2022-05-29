import sigproc
import numpy as np
from scipy.fftpack import dct


# ======================  MFCC  ============================
def mfcc(signal, samplerate=16000, numcep=13, nfilt=32, nfft=512, lowfreq=0,
         highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.    
    :param numcep: the number of cepstrum to return, default 13    
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97. 
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22. 
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, nfilt, nfft, lowfreq, highfreq, preemph)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy: feat[:, 0] = np.log(energy)  # replace first cepstral coefficient with log of frame energy
    #    feat -= (np.mean(feat, axis=0) + 1e-8)
    return feat


def mel_spectrum(sig_spec, melfb):
    """
    sig_spec: STFT of the speech signal
    melfb: Mel filterbank - get_filterbanks()
    """
    spec_mel = np.dot(sig_spec, melfb.T)
    spec_mel = np.where(spec_mel == 0.0, np.finfo(float).eps, spec_mel)
    #    return np.log(spec_mel)-np.mean(np.log(spec_mel),axis=0)
    return np.log(spec_mel)


def mfcc_opt(mel_spectrum, numcep=13):
    feat = dct(mel_spectrum, type=2, axis=1, norm='ortho')[:, :numcep]
    #    feat = lifter(feat,ceplifter)
    #    if appendEnergy: feat[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
    #    feat -= (np.mean(feat, axis=0) + 1e-8)#Cepstral mean subtraction
    return feat


def fbank(signal, samplerate=16000, nfilt=26, nfft=512, preemph=0.97):
    """从音频信号计算梅尔滤波器组的能量特征。

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with. 
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97. 
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """

    if (len(signal.shape) > 1):  # In case of framed signal
        tempsig = []
        for idx_sig in signal:
            tempsig.append(sigproc.preemphasis(idx_sig, preemph))   # 预加重
        signal = np.vstack(tempsig)
    else:
        signal = np.vstack(sigproc.preemphasis(signal, preemph)).T
    #    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate)
    pspec = sigproc.powspec(signal, nfft)
    energy = np.sum(pspec, 1)  # this stores the total energy in each frame
    energy = np.where(energy == 0, np.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    fb = mel_filterbanks(nfilt, nfft, samplerate)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    feat = np.where(feat == 0, np.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    return feat, energy


def ssc(signal, samplerate=16000, winlen=0.025, winstep=0.01,
        nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97):
    """从音频信号计算谱子带质心特征(pectral Subband Centroid features)

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)    
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)    
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97. 
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. 
    """
    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate)
    pspec = sigproc.powspec(frames, nfft)
    pspec = np.where(pspec == 0, np.finfo(float).eps, pspec)  # if things are all zeros we get problems

    fb = mel_filterbanks(nfilt, nfft, samplerate)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    R = np.tile(np.linspace(1, samplerate / 2, np.size(pspec, 1)), (np.size(pspec, 0), 1))

    return np.dot(pspec * R, fb.T) / feat


def hz2mel(hz):
    """ Hz to Mels """
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    """ Mels to HZ """
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """计算一个Mel-filterbank (M,F)
    :param nfilt: filterbank中的滤波器数量
    :param nfft: FFT size
    :param samplerate: 采样率
    :param lowfreq: Mel-filter的最低频带边缘
    :param highfreq: Mel-filter的最高频带边缘，默认samplerate/2
    """
    highfreq = highfreq or samplerate / 2

    # 按梅尔均匀间隔计算 点
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    hz_points = mel2hz(melpoints)  # 将mel频率再转到hz频率
    # bin = samplerate/2 / NFFT/2=sample_rate/NFFT    # 每个频点的频率数
    # bins = hz_points/bin=hz_points*NFFT/ sample_rate    # hz_points对应第几个fft频点
    bin = np.floor((nfft + 1) * hz_points / samplerate)

    fbank = np.zeros([nfilt, int(nfft / 2 + 1)])  # (m,f)
    for i in range(0, nfilt):
        for j in range(int(bin[i]), int(bin[i + 1])):
            fbank[i, j] = (j - bin[i]) / (bin[i + 1] - bin[i])
        for j in range(int(bin[i + 1]), int(bin[i + 2])):
            fbank[i, j] = (bin[i + 2] - j) / (bin[i + 2] - bin[i + 1])

    #    fbank -= (np.mean(fbank, axis=0) + 1e-8)
    return fbank


# ------------------------------------------------------------------------------------------------
def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2) * np.sin(np.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra
