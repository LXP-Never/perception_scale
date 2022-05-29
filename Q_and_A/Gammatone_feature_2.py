# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/5/24
"""
参考自：Gammatone filter master
有python和C语言版本，写的很好，还给了公式推导文档，但是增益归一化的那部分代码是C版本中的，python版本没有
但是当我将最高的截止频率改成8000的时候和5000的时候不一样
todo:增益归一化，思考一下如何提取GFCC特征呢？
"""
import numpy as np
import matplotlib.pyplot as plt


class GTF():
    def __init__(self, fs, cfs=None, cf_low=None, cf_high=None,
                 freq_low=None, freq_high=None, n_band=1):
        """
        Args:
            fs: 采样率
            cf_low,cf_high: 最低和最高中心频率
            freq_low,freq_high: 最低和最高截止频率
            n_band: 频带数量
        """
        self.bw_factor = 1.019

        if cfs is None:
            if cf_low is None:
                if freq_low is not None:
                    # 将freq_low设置为最低截止频率
                    cf_low = ((2 * freq_low + self.bw_factor * 24.7)
                              / (2 - self.bw_factor * 24.7 * 4.37 / 1000))
                else:
                    raise Exception('neither cf_low or freq_low is specified')
            if cf_high is None:
                if freq_high is not None:
                    # 将freq_high设置为最高截止频率
                    cf_high = ((2 * freq_high - self.bw_factor * 24.7)
                               / (2 + self.bw_factor * 24.7 * 4.37 / 1000))
                else:
                    raise Exception(
                        'neither cf_high or freq_high is specified')

            # 中心频率
            cfs = self.ERB_space(freq_low=cf_low, freq_high=cf_high, n_band=n_band)
        bws = self.cal_bw(cfs)  # 根据中心频率计算bandwidths

        self.cf_low = cf_low
        self.cf_high = cf_high
        self.fs = fs
        self.cfs = cfs
        self.bws = bws
        self.n_band = n_band

    def ERB_space(self, freq_low, freq_high, n_band):
        """根据ERB scale等分频率范围 (freq_low~freq_high)
        Args:
            freq_low: 最低中心频率
            freq_high: 最高中心频率
            n_band: 频带数量
            divide_type: default to ERB
        """
        if n_band == 1:
            return np.asarray(freq_low, dtype=np.float).reshape(1, )
        low_erb = self.Hz2ERBscal(freq_low)
        high_erb = self.Hz2ERBscal(freq_high)
        erb_elem = (high_erb - low_erb) / (n_band - 1)  # 带宽 数量
        f = self.ERBscal2Hz(low_erb + erb_elem * np.arange(n_band))
        return f

    def Hz2ERBscal(self, freq):
        """convert Hz to ERB scale"""
        return 21.4 * np.log10(4.37 * freq / 1e3 + 1)

    def ERBscal2Hz(self, erb_num):
        """convert ERB scale to Hz"""
        return (10 ** (erb_num / 21.4) - 1) / 4.37 * 1e3

    def cal_ERB(self, cf):
        """根据 Glasberg and Moore 给出的方程计算 中心频率的ERB(Hz)
        cf: 中心频率 Hz
        """
        return 24.7 * (4.37 * cf / 1000 + 1.0)

    def cal_bw(self, cf):
        """计算 3-dB bandwidth
            cf: 中心频率 Hz
        """
        erb = self.cal_ERB(cf)
        return self.bw_factor * erb

    def filter_py(self, x, align_env=False, align_tfs=False, delay_common=None,gain_norm=True):
        """Filters in Python
        Args:
        x: 形状为[x_len,n_band]的信号，如果x只有一维，则将n_band加为1
        align_env: aligned peaks of Gammatone filter impulse response
        align_tfs: aligned the time fine structure
        delay_common: if aligned, give the same delay to all channels,
        default, aligned to the maximum delay
        Returns:
        fitler result with the shape of [n_band,x_len,n_band]
        """
        tpt = 2 * np.pi * (1.0 / self.fs)

        x = x.copy()
        if len(x.shape) > 2:
            raise Exception('two many dimensions for x')
        # ensure x is 2-D array
        x_len = x.shape[0]      # 信号长度
        n_chann = x.shape[1]    # 信号通道数
        if len(x.shape) == 1:
            x = np.reshape(x, [-1, 1])

        if align_env is True:
            delays = np.round(3.0 / (2.0 * np.pi * self.bws) * self.fs) / self.fs
            if delay_common is not None:
                delay_common = np.max(delays)
                delays = np.int(delays - delay_common)
        else:
            delays = np.zeros(self.n_band)

        # IIR and FIR filters outputs
        out_a = np.zeros((5, n_chann), dtype=np.complex)
        coefs_a = np.zeros(5)  # 分子
        out_b = 0
        coefs_b = np.zeros(4)  # 分母

        norm_factors = np.zeros(self.n_band)
        y = np.zeros((self.n_band, x_len, n_chann), dtype=np.float)
        for band_i in range(self.n_band):
            bw = self.bws[band_i]   # 第i个带宽
            cf = self.cfs[band_i]   # 第i个中心频率
            k = np.exp(-tpt * bw)
            # 滤波器系数
            coefs_a = [1, 4 * k, -6 * k ** 2, 4 * k ** 3, -k ** 4]
            coefs_b = [1, 1, 4 * k, k ** 2]

            norm_factors[band_i] = (1 - k) ** 4 / (1 + 4 * k + k ** 2) * 2
            delay_len_band = np.int(delays[band_i] * self.fs)
            if align_tfs:
                phi_init = -3.0 * cf / self.bws[band_i]
            else:
                phi_init = 0
            for sample_i in range(x_len):
                freq_shiftor = np.exp(-1j * (tpt * cf * sample_i))
                # IIR part
                out_a[0, :] = x[sample_i, :] * freq_shiftor * np.exp(1j * phi_init)
                for order_i in range(1, 5):
                    out_a[0, :] = (out_a[0, :]
                                   + coefs_a[order_i] * out_a[order_i, :])
                # FIR part
                out_b = 0
                for order_i in range(1, 4):
                    out_b = out_b + coefs_b[order_i] * out_a[order_i, :]

                if sample_i > delay_len_band:
                    y[band_i, sample_i - delay_len_band, :] = (
                            norm_factors[band_i]
                            * np.real(out_b
                                      * np.conjugate(freq_shiftor)))
                # update IIR output
                for order_i in range(4, 0, -1):
                    out_a[order_i, :] = out_a[order_i - 1, :]
        return np.squeeze(y)

    def plot_delay_gain_cfs(self):
        """
        Plot delay and center-frequency gain of gammatone filter
        before alignment and gain-normalization
        """
        # k = np.exp(-2*np.pi/self.fs*self.bws)
        Q = np.divide(self.cfs, self.bws)  # quality of filter

        temp1 = 8 * Q * (1 - 4 * Q ** 2)
        temp2 = np.multiply(Q ** 2, Q * (16 * Q ** 2 - 24))
        phi_delay = (np.arctan(np.divide(temp1, temp2 + 2))
                     - np.arctan(np.divide(temp1, temp2 + 1)))
        delays = phi_delay / (2 * np.pi) * 1e3

        correct_factor = np.sqrt(((temp2 + 1) ** 2 + temp1 ** 2)
                                 / ((temp2 + 2) ** 2 + temp1 ** 2))
        gains = 10 * np.log10(3 / (2 * np.pi * self.bws)) * correct_factor

        fig = plt.figure(figsize=[8, 3])
        ax = fig.subplots(1, 2)
        # gains at cfs
        ax[0].plot(self.cfs / 1000, gains, linewidth=2)
        ax[0].set_xlabel('Center frequency(kHz)')
        ax[0].set_ylabel('Gain(dB)')
        # delay at cfs
        ax[1].plot(self.cfs / 1000, delays, linewidth=2)
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Center frequency(kHz)')
        ax[1].set_ylabel('Delay(ms)')
        plt.tight_layout()

        return fig

    def plot_filter_spectrum(self, cf=4e3):
        order = 4
        fs = self.fs
        bw = self.cal_ERB(cf)
        freq_bins = np.arange(1, fs / 2)  # frequency resolution: 1Hz
        # n_freq_bin = freq_bins.shape[0]
        gain_func = 6 / (((2 * np.pi * bw) ** order)
                         * (np.divide(1, 1 + 1j * (freq_bins - cf) / bw) ** order
                            + np.divide(1, 1 + 1j * (freq_bins + cf) / bw) ** order))

        amp_spectrum = np.abs(gain_func)

        phase_spectrum = np.angle(gain_func)
        cf_bin_index = np.int16(cf)
        # unwrap based on phase at cf
        phase_spectrum[:cf_bin_index] = np.flip(
            np.unwrap(
                np.flip(
                    phase_spectrum[:cf_bin_index])))
        phase_spectrum[cf_bin_index:] = np.unwrap(
            phase_spectrum[cf_bin_index:])
        # delays = np.divide(phase_spectrum,freq_bins)

        linewidth = 2
        # Amplitude-phase spectrum
        fig = plt.figure()
        ax1 = fig.subplots(1, 1)
        color = 'tab:red'
        ax1.semilogy(freq_bins / 1000, amp_spectrum, color=color,
                     linewidth=linewidth, label='amp')
        ax1.set_ylabel('dB', color=color)
        ax1.set_xlabel('Frequency(kHz)')
        ax1.legend(loc='upper left')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title('cf={}Hz'.format(cf))

        ax2 = ax1.twinx()
        color = 'tab: blue'
        ax2.plot(freq_bins / 1000, phase_spectrum, color=color,
                 linewidth=linewidth, label='phase')
        ax2.legend(loc='upper right')
        ax2.plot([cf / 1000, cf / 1000], [-8, 8], '-.', color='black')
        ax2.plot([0, fs / 2 / 1000], [0, 0], '-.', color='black')
        ax2.set_ylabel('phase(rad)', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        return fig

    def plot_ir_spec(self, ir, fs=None, cfs=None, ax=None, fig=None):
        """plot the waveform and spectrum of given impulse response
        Args:
            ir: impulse response
            fs: sample frequency,use self.fs as default
            fig: handle of matplotlib figure, if not given, not figure will
                be created
            title: title for ir waveform sub-panel
        """

        if fs is None:
            fs = self.fs
        if cfs is None:
            cfs = self.cfs

        ir_len = ir.shape[1]
        N_fft = ir_len
        N_fft_half = np.int(N_fft / 2)

        index = np.flip(np.argsort(cfs))
        cfs = cfs[index]
        ir = ir[index, :]

        spec = np.abs(np.fft.fft(ir, N_fft, axis=1))[:, :N_fft_half]
        spec_dB = 10 * np.log10(spec)

        freq_ticks = np.arange(N_fft_half) / N_fft * self.fs

        plt.plot(freq_ticks, spec_dB.T)
        plt.xlim([self.cf_low / 8.0, self.cf_high * 1.5])
        plt.xlabel('Frequency(Hz)')
        plt.show()

    def get_ir(self, ir_duration=1, env_aligned=False,
               fine_aligned=False, delay_common=-1, gain_norm=False):
        """返回 gammatone filter bank 的脉冲响应(impulse responses)

        :param ir_duration: 脉冲响应时间长度(s)
        :param env_aligned: 是否对齐ir的包络线，default to False
        :param fine_aligned:如果包络对齐，是否对齐精细结构，默认为False
        :param delay_common: 设置滤波器ir的延迟(s)，如果它是对齐的，默认为-1，使所有滤波器对齐到最大峰值位置
        :param gain_norm: 是否归一化中心频率增益
        :return: 滤波器组脉冲响应 [n_band,ir_len]
        """
        n_sample = np.int(self.fs * ir_duration)
        # impulse stimuli
        x = np.zeros((n_sample, 1))
        x[100] = 1  # spike
        irs = self.filter_py(x, env_aligned, fine_aligned, delay_common,gain_norm=gain_norm)
        return irs

    def get_ir_equation(self, t=None):
        """
        """
        if t is None:
            t = np.arange(self.fs) / self.fs

        n_sample = t.shape[0]
        ir = np.zeros((self.n_band, n_sample))

        order = 4

        part1 = t ** (order - 1)
        for band_i in range(self.n_band):
            part2 = np.multiply(np.cos(2 * np.pi * self.cfs[band_i] * t),
                                np.exp(-2 * np.pi * self.bws[band_i] * t))
            ir[band_i] = np.multiply(part1, part2)

        return ir


if __name__ == "__main__":
    fs = 16000
    gt_filter = GTF(fs, freq_low=80, freq_high=5000, n_band=22)

    # gain normalization
    irs_norm = gt_filter.get_ir()
    gt_filter.plot_ir_spec(irs_norm)

