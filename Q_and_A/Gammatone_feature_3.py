# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/24
"""
pyfilterbank-master: 写的很好的一个API, 但是我看不到细节
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import factorial
from scipy.signal import lfilter

# ERB means "Equivalent retangular band(-width)" Constants:
EarQ = 9.265  # _ERB_Q
minBW = 24.7  # minBW


def erb_count(f_c):
    """ 根据中心频率f_c(Hz)计算 等效矩形带宽的数量 """
    return 21.4 * np.log10(4.37 * 0.001 * f_c + 1)


def erb_aud(f_c):
    """ 根据中心频率f_c(Hz)计算滤波器 的等效矩阵带宽
        Implements Equation 13 in [Hohmann2002]
    """
    return minBW + f_c / EarQ


def hertz_to_erbscale(frequency):
    """ [Hohmann2002] Equation 16
    :param frequency: Hz
    :return: ERB 频率
    """
    return EarQ * np.log(1 + frequency / (minBW * EarQ))


def erbscale_to_hertz(erb):
    """ [Hohmann2002] Equation 17
    :param erb:  ERB 频率
    :return: Hz
    """
    return (np.exp(erb / EarQ) - 1) * minBW * EarQ


def frequencies_gammatone_bank(start_band, end_band, norm_freq, density):
    """返回一系列gamatone滤波器的中心频率
    :param start_band: Erb counts below norm_freq.
    :param end_band: Erb counts  over norm_freq.
    :param norm_freq: The reference frequency where all filters are around
    :param density: ERB density 1would be `erb_aud`.
    :return: 中心频率 array
    """
    norm_erb = hertz_to_erbscale(norm_freq)
    f_c = erbscale_to_hertz(np.arange(start_band, end_band, density) + norm_erb)  # 返回中心频率数组
    return f_c


def design_filter(sample_rate=44100, order=4, centerfrequency=1000.0, band_width=None, band_width_factor=1.0,
                  attenuation_half_bandwidth_db=-3):
    """  返回 gammatone filter 系数 [Hohmann2002]_.
    :param sample_rate: 采样率
    :param order: 阶数
    :param centerfrequency: 中心频率
    :param band_width: 带宽
    :param band_width_factor:
    :param attenuation_half_bandwidth_db: 衰减半带宽db
    :return: b, a : ndarray, ndarray
    """
    if band_width:
        phi = np.pi * band_width / sample_rate
        # alpha = 10**(0.1 * attenuation_half_bandwidth_db / order)
        # p = (-2 + 2 * alpha * cos(phi)) / (1 - alpha)
        # lambda_ = -p/2 - sqrt(p*p/4 - 1)
    elif band_width_factor:
        erb_audiological = band_width_factor * erb_aud(centerfrequency)
        phi = np.pi * erb_audiological / sample_rate
        # a_gamma = ((factorial(pi * (2*order - 2)) *
        #             2**(-(2*order - 2))) / (factorial(order - 1)**2))
        # b = erb_audiological / a_gamma
        # lambda_ = exp(-2 * pi * b / sample_rate)
    else:
        raise ValueError(
            'You need to specify either `band_width` or `band_width_factor!`')

    alpha = 10 ** (0.1 * attenuation_half_bandwidth_db / order)
    p = (-2 + 2 * alpha * np.cos(phi)) / (1 - alpha)
    lambda_ = -p / 2 - np.sqrt(p * p / 4 - 1)
    beta = 2 * np.pi * centerfrequency / sample_rate
    coef = lambda_ * np.exp(1j * beta)
    factor = 2 * (1 - abs(coef)) ** order
    b, a = np.array([factor]), np.array([1., -coef])
    return b, a


def fosfilter(b, a, order, signal, states=None):
    """返回用`b` and `a`(一阶部分)滤波的信号
    这个函数是为了通过一阶节级联复伽马滤波器(section cascaded complex gammatone filters.)来滤波信号而创建的。

    :param b, a:一阶滤波器的滤波系数。可以是复数。
    :param order: 滤波器阶数
    :param signal: 要滤波的信号
    :param states: filter states 为长度' order '。初始时可以设置为None。
    :return:
        signal : Output signal, that is filtered and complex valued (analytical signal).
        states : 数组，filter states为长度' order '。在块处理时，需要将它循环回此函数。
    """

    if not states:
        states = np.zeros(order, dtype=np.complex128)

    for i in range(order):
        state = [states[i]]
        signal, state = lfilter(b, a, signal, zi=state)
        states[i] = state[0]
        b = np.ones_like(b)
    return signal, states


def freqz_fos(b, a, order, nfft, plotfun=None):
    impulse = _create_impulse(nfft)
    response, states = fosfilter(b, a, order, impulse)
    freqresponse = np.fft.rfft(np.real(response))
    frequencies = np.fft.rfftfreq(nfft)
    if plotfun:
        plotfun(frequencies, freqresponse)
    return freqresponse, frequencies, response


def design_filtbank_coeffs(samplerate, order, f_c, bandwidths=None, bandwidth_factor=None,
                           attenuation_half_bandwidth_db=-3):
    for i, cf in enumerate(f_c):
        if bandwidths:
            bw = bandwidths[i]
            bwf = None
        else:
            bw = None
            bwf = bandwidth_factor

        yield design_filter(
            samplerate, order, cf, band_width=bw,
            band_width_factor=bwf,
            attenuation_half_bandwidth_db=attenuation_half_bandwidth_db)


class GammatoneFilterbank:
    def __init__(self, samplerate=44100, order=4, startband=-12, endband=12,
                 normfreq=1000.0, density=1.0, bandwidth_factor=1.0, desired_delay_sec=0.02):

        self.samplerate = samplerate
        self.order = order
        self.centerfrequencies = frequencies_gammatone_bank(startband, endband, normfreq, density)
        self._coeffs = tuple(design_filtbank_coeffs(samplerate, order, self.centerfrequencies,
                                                    bandwidth_factor=bandwidth_factor))
        self.init_delay(desired_delay_sec)
        self.init_gains()

    def init_delay(self, desired_delay_sec):
        self.desired_delay_sec = desired_delay_sec
        self.desired_delay_samples = int(self.samplerate * desired_delay_sec)
        self.max_indices, self.slopes = self.estimate_max_indices_and_slopes(
            delay_samples=self.desired_delay_samples)
        self.delay_samples = self.desired_delay_samples - self.max_indices
        self.delay_memory = np.zeros((len(self.centerfrequencies), np.max(self.delay_samples)))

    def init_gains(self):
        self.gains = np.ones(len(self.centerfrequencies))
        # not correct until now:
        # x, s = list(zip(*self.analyze(_create_impulse(self.samplerate/10))))
        # rss = [np.sqrt(np.sum(np.real(b)**2)) for b in x]
        # self.gains = 1/np.array(rss)

    def analyze(self, signal, states=None):
        for i, (b, a) in enumerate(self._coeffs):
            st = None if not states else states[i]
            yield fosfilter(b, a, self.order, signal, states=st)

    def reanalyze(self, bands, states=None):
        for i, ((b, a), band) in enumerate(zip(self._coeffs, bands)):
            st = None if not states else states[i]
            yield fosfilter(b, a, self.order, band, states=st)

    def synthesize(self, bands):
        return np.array(list(self.delay(
            [b * g for b, g in zip(bands, self.gains)]))).sum(axis=0)

    def delay(self, bands):
        self.phase_factors = np.abs(self.slopes) * 1j / self.slopes
        for i, band in enumerate(bands):
            phase_factor = self.phase_factors[i]
            delay_samples = self.delay_samples[i]
            if delay_samples == 0:
                yield np.real(band) * phase_factor
            else:
                yield np.concatenate(
                    (self.delay_memory[i, :delay_samples],
                     np.real(band[:-delay_samples])),
                    axis=0)
                self.delay_memory[i, :delay_samples] = np.real(
                    band[-delay_samples:])

    def estimate_max_indices_and_slopes(self, delay_samples=None):
        if not delay_samples:
            delay_samples = int(self.samplerate / 10)
        sig = _create_impulse(delay_samples)
        bands = list(zip(*self.analyze(sig)))[0]
        ibandmax = [np.argmax(np.abs(b[:delay_samples])) for b in bands]
        slopes = [b[i + 1] - b[i - 1] for (b, i) in zip(bands, ibandmax)]
        return np.array(ibandmax), np.array(slopes)

    def freqz(self, nfft=4096, plotfun=None):
        def gen_freqz():
            for b, a in self._coeffs:
                yield freqz_fos(b, a, self.order, nfft, plotfun)

        return list(gen_freqz())


def _create_impulse(num_samples):
    sig = np.zeros(num_samples) + 0j
    sig[0] = 1.0
    return sig


def example_filterbank():
    from pylab import plt
    import numpy as np

    x = _create_impulse(2000)
    gfb = GammatoneFilterbank(density=1)

    analyse = gfb.analyze(x)
    imax, slopes = gfb.estimate_max_indices_and_slopes()
    fig, axs = plt.subplots(len(gfb.centerfrequencies), 1)
    for (band, state), imx, ax in zip(analyse, imax, axs):
        ax.plot(np.real(band))
        ax.plot(np.imag(band))
        ax.plot(np.abs(band))
        ax.plot(imx, 0, 'o')
        ax.set_yticklabels([])
        [ax.set_xticklabels([]) for ax in axs[:-1]]

    axs[0].set_title('Impulse responses of gammatone bands')

    fig, ax = plt.subplots()

    def plotfun(x, y):
        ax.semilogx(x, 20 * np.log10(np.abs(y) ** 2))

    gfb.freqz(nfft=2 * 4096, plotfun=plotfun)
    plt.grid(True)
    plt.title('Absolute spectra of gammatone bands.')
    plt.xlabel('Normalized Frequency (log)')
    plt.ylabel('Attenuation /dB(FS)')
    plt.axis('Tight')
    plt.ylim([-90, 1])
    plt.show()

    return gfb


def example_gammatone_filter():
    from pylab import plt, np
    sample_rate = 44100
    order = 4
    b, a = design_filter(
        sample_rate=sample_rate,
        order=order,
        centerfrequency=1000.0,
        attenuation_half_bandwidth_db=-3,
        band_width_factor=1.0)

    x = _create_impulse(1000)
    y, states = fosfilter(b, a, order, x)
    y = y[:800]
    plt.plot(np.real(y), label='Re(z)')
    plt.plot(np.imag(y), label='Im(z)')
    plt.plot(np.abs(y), label='|z|')
    plt.legend()
    plt.show()
    return y, b, a


if __name__ == '__main__':
    gfb = example_filterbank()
    y = example_gammatone_filter()

    plt.plot(gfb)
    plt.show()
