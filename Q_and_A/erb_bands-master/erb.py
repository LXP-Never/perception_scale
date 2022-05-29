"""
基于Josh McDermott的Matlab滤波器组代码:
https://github.com/wil-j-wil/py_bank
"""

import numpy as np


class FilterBank(object):
    def __init__(self, len_signal, sample_rate, total_erb_bands, low_lim, high_lim):
        """
        :param len_signal: 信号长度
        :param sample_rate: 采样率
        :param total_erb_bands: erb频带数(不包括为完美重构而添加的高通和低通)
        :param low_lim: 第一个(最低)滤波器的中心频率
        :param high_lim: 最后(最高)滤波器的中心频率
        """
        self.len_signal = len_signal
        self.sample_rate = sample_rate
        self.total_erb_bands = total_erb_bands
        self.erb_bands = []
        self.freq_index = []  # 频段索引，以Hz为单位
        self.bandwidths = []  # 带宽
        self.low_lim = low_lim
        # freqs = 频率范围
        # nfreqs = 频点数
        self.high_lim, self.freqs, self.nfreqs = self.build_frequency_limits(len_signal, sample_rate, high_lim)

    def build_frequency_limits(self, len_signal, sample_rate, high_lim):
        """
        Build frequency limits using a linear scale in Hz
        """
        if len_signal % 2 == 0:
            nfreqs = len_signal  # F
            max_freq = sample_rate / 2
        else:
            nfreqs = (len_signal - 1)
            max_freq = sample_rate * (len_signal - 1) / 2 / len_signal
        freqs = np.linspace(0, max_freq, nfreqs + 1)  # 每个STFT频点对应多少Hz
        if high_lim > sample_rate / 2:
            high_lim = max_freq
        return high_lim, freqs, int(nfreqs)


class EquivalentRectangularBandwidth(FilterBank):
    """
    erb_low  = lowest erb band
    erb_high = highest erb band
    erb_lims = limits between erb bands
    cutoffs  = cuts between erb bands
    """
    def __init__(self, len_signal, sample_rate, total_erb_bands, low_lim, high_lim):
        super(EquivalentRectangularBandwidth, self).__init__(len_signal, sample_rate, total_erb_bands, low_lim,
                                                             high_lim)
        # 在ERB刻度上建立均匀间隔
        erb_low = self.freq2erb(self.low_lim)  # 最低 截止频率
        erb_high = self.freq2erb(self.high_lim)  # 最高 截止频率
        # 在ERB频率上均分为（total_erb_bands + ）2个 频带
        erb_lims = np.linspace(erb_low, erb_high, self.total_erb_bands + 2)
        self.cutoffs = self.erb2freq(erb_lims)  # 将 ERB频率再转到 hz频率, 在线性频率Hz上找到ERB截止频率对应的频率
        # self.nfreqs  F
        # self.freqs # 每个STFT频点对应多少Hz
        self.filters = self.get_bands(self.total_erb_bands, self.nfreqs, self.freqs, self.cutoffs)

    def freq2erb(self, freq_Hz):
        """ Convert Hz to ERB number """
        n_erb = 21.4 * np.log10(1 + 0.00437 * freq_Hz)
        return n_erb

    def erb2freq(self, n_erb):
        """ Convert ERB number to Hz """
        freq_Hz = (np.power(10, (n_erb / 21.4)) - 1) / 0.00437
        return freq_Hz

    def get_bands(self, total_erb_bands, nfreqs, freqs, cutoffs):
        """
        获取erb bands、索引、带宽和滤波器形状
        :param erb_bands_num: ERB 频带数
        :param nfreqs: 频点数 F
        :param freqs: 每个STFT频点对应多少Hz
        :param cutoffs: 中心频率 Hz
        :param erb_points: ERB频带界限 列表
        :return:
        """
        cos_filts = np.zeros([nfreqs + 1, total_erb_bands])  # (F, ERB)
        for i in range(total_erb_bands):
            lower_cutoff = cutoffs[i]  # 上限截止频率 Hz
            higher_cutoff = cutoffs[i + 2]  # 下限截止频率 Hz, 相邻filters重叠50%
            freq_bandwidth = higher_cutoff - lower_cutoff  # 带宽 Hz
            erb_center = (self.freq2erb(lower_cutoff) + self.freq2erb(higher_cutoff)) / 2  # ERB轴上的 中心频率 erb
            center_freq = self.erb2freq(erb_center)  # 中心频率 Hz
            q_factor = center_freq / freq_bandwidth
            index = (np.abs(freqs - center_freq)).argmin()  # 返回沿轴的最小值的索引

            self.erb_bands.append(erb_center)  # ERB轴上的 中心频率 erb
            self.freq_index.append(index)  # 返回沿轴的最小值的索引
            self.bandwidths.append(freq_bandwidth)  # 带宽 Hz

            lower_index = np.min(np.where(freqs > lower_cutoff))  # 下限截止频率对应的Hz索引 Hz。np.where 返回满足条件的索引
            higher_index = np.max(np.where(freqs < higher_cutoff))  # 上限截止频率对应的Hz索引
            avg = (self.freq2erb(lower_cutoff) + self.freq2erb(higher_cutoff)) / 2
            rnge = self.freq2erb(higher_cutoff) - self.freq2erb(lower_cutoff)
            cos_filts[lower_index:higher_index + 1, i] = np.cos(
                (self.freq2erb(freqs[lower_index:higher_index + 1]) - avg) / rnge * np.pi)  # w=2*pi*f

        # 加入低通和高通，得到完美的重构
        filters = np.zeros([nfreqs + 1, total_erb_bands + 2])  # (F, ERB)
        filters[:, 1:total_erb_bands + 1] = cos_filts
        # 低通滤波器上升到第一个余cos filter的峰值
        higher_index = np.max(np.where(freqs < cutoffs[1])) # 上限截止频率对应的Hz索引
        filters[:higher_index + 1, 0] = np.sqrt(1 - np.power(filters[:higher_index + 1, 1], 2))
        # 高通滤波器下降到最后一个cos filter的峰值
        lower_index = np.min(np.where(freqs > cutoffs[total_erb_bands]))
        filters[lower_index:nfreqs + 1, total_erb_bands + 1] = np.sqrt(
            1 - np.power(filters[lower_index:nfreqs + 1, total_erb_bands], 2))
        return cos_filts
