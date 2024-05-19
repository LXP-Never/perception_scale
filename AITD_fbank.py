import numpy as np
import torch
import torch.nn as nn


class AITDFilterBank(nn.Module):
    def __init__(self, band_num=32, freq_bins=257, fs=16000, band_method="erb", norm=False):
        super().__init__()
        self.band_num = band_num
        self.freq_bins = freq_bins
        self.band_method = band_method
        self.fs = fs
        self.band_segment_idx = self.get_segment_index()
        if self.freq_bins == self.band_num:
            to_band_matrix = torch.eye(self.freq_bins, self.band_num, requires_grad=False)
            inv_to_band_matrix = torch.eye(self.freq_bins, self.band_num, requires_grad=False)
        else:
            to_band_matrix = torch.zeros(self.freq_bins, self.band_num, requires_grad=False)  # (F,B)
            inv_to_band_matrix = torch.zeros(self.band_num, self.freq_bins, requires_grad=False)
            for b in range(self.band_num):
                if b < (self.band_num - 1):
                    band_size = self.band_segment_idx[b + 1] - self.band_segment_idx[b]
                    for f in range(band_size):
                        frac = f / band_size
                        to_band_matrix[f + self.band_segment_idx[b], b] = 1.0 - frac  # right of central frequency
                if b > 0.5:
                    band_size = self.band_segment_idx[b] - self.band_segment_idx[b - 1]
                    for f in range(band_size):
                        frac = f / band_size
                        to_band_matrix[f + self.band_segment_idx[b - 1], b] = frac  # left of central frequency
            # to_band_matrix[:, 0] *= 2.0
            # to_band_matrix[:, -1] *= 2.0

            for b in range(self.band_num - 1):
                band_size = self.band_segment_idx[b + 1] - self.band_segment_idx[b]
                for f in range(band_size):
                    frac = float(f) / band_size
                    inv_to_band_matrix[b, self.band_segment_idx[b] + f] = 1.0 - frac
                    inv_to_band_matrix[b + 1, self.band_segment_idx[b] + f] = frac

        self.to_band_matrix = nn.Parameter(torch.FloatTensor(to_band_matrix), requires_grad=False)
        self.inv_to_band_matrix = nn.Parameter(torch.FloatTensor(inv_to_band_matrix), requires_grad=False)
        if norm:
            self.to_band_matrix /= self.to_band_matrix.sum(dim=0, keepdim=True)
            self.inv_to_band_matrix /= self.inv_to_band_matrix.sum(dim=1, keepdim=True)
        print("================ parameters in BandConverter =================")
        # print(f"self.to_band_matrix {self.to_band_matrix}")
        # print(f"self.inv_to_band_matrix {self.inv_to_band_matrix}")
        print("self.band_segment_idx", len(self.band_segment_idx), self.band_segment_idx)
        print(f"self.band_num {self.band_num}")
        print(f"self.freq_bins {self.freq_bins}")
        print(f"self.fs {self.fs}")
        print(f"self.band_method {self.band_method}")
        print("=================================================================")

    def forward_band_mat(self, stft_mag, method="magnitude"):
        """
        :param stft_mag: [B C T F] or [B T F]
        :param method: "magnitude" or "energy"
        :return:
        """
        if method == "energy":
            stft_mag = stft_mag ** 2
        band_mag = torch.matmul(stft_mag, self.to_band_matrix)
        return band_mag

    def inverse_band_mat(self, band_mag, method="magnitude"):
        """
        :param band_mag:
        :param method:
        :return:
        """
        stft_mag = torch.matmul(band_mag, self.inv_to_band_matrix)
        if method == "energy":
            stft_mag = stft_mag ** 0.5
        return stft_mag

    def forward_band(self, stft_mag, method="magnitude"):
        """
        :param stft_mag: [B C T F] or [B T F]
        :param method: "magnitude" or "energy"
        :return:
        """
        with torch.no_grad():
            nframe = stft_mag.shape[1]
            nbatch = stft_mag.shape[0]
            band_mag = torch.zeros(nbatch, nframe, self.band_num, device=stft_mag.device)
            if method == "energy":
                stft_mag = stft_mag ** 2
            for i in range(self.band_num - 1):
                band_size = self.band_segment_idx[i + 1] - self.band_segment_idx[i]
                for j in range(band_size):
                    frac = float(j) / band_size
                    tmp = stft_mag[:, :, self.band_segment_idx[i] + j]
                    band_mag[:, :, i] += (1. - frac) * tmp
                    band_mag[:, :, i + 1] += frac * tmp
            band_mag[:, :, 0] *= 2
            band_mag[:, :, self.band_num - 1] *= 2
            return band_mag

    def inverse_band(self, band_mag, method="magnitude"):
        """
        :param band_mag: [B T F]
        :param method:
        :return:
        """
        with torch.no_grad():
            nframe = band_mag.shape[1]
            nbatch = band_mag.shape[0]
            gain = torch.zeros(nbatch, nframe, self.freq_bins, device=band_mag.device)
            for i in range(self.band_num - 1):
                band_size = self.band_segment_idx[i + 1] - self.band_segment_idx[i]
                for j in range(band_size):
                    frac = float(j) / band_size
                    gain[:, :, self.band_segment_idx[i] + j] = (1 - frac) * band_mag[:, :, i] + frac * band_mag[:, :,
                                                                                                       i + 1]
            if method == "energy":
                gain = gain ** 0.5
            return gain

    def get_segment_index(self):
        """
        This function computes an array of band_num frequencies uniformly spaced on ERB or Bark scale.
        For start_time definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983). "Suggested formulae for
        calculating auditory-filter bandwidths and excitation patterns," J. Acoust. Soc. Am. 74, 750-753
        """
        assert self.band_method in ["bark", "erb"], f"Unknown method {self.method}!"
        if self.band_method == "bark":
            assert self.fs in [16000, 24000], 'bark method is only support 16000 and 24000'
            band_num = self.band_num
            while True:
                if self.fs == 16000:
                    x = 0.9041 * (np.arange(band_num) / (band_num - 1))
                elif self.fs == 24000:
                    x = 0.9802 * (np.arange(band_num) / (band_num - 1))
                cf = [1.552e+05, -3.929e+05, 3.996e+05, -1.936e+05, 4.779e+04, -2855, 128.8]

                fx = x ** 6 * cf[0] + x ** 5 * cf[1] + x ** 4 * cf[2] + x ** 3 * cf[3] + x ** 2 * cf[4] + x * cf[5] + \
                     cf[6]
                band_seg_idx = np.around(fx / self.fs * 2.0 * self.freq_bins).astype(int)
                band_seg_idx[0] = np.array(0).astype(int)
                band_seg_idx[-1] = np.array(self.freq_bins - 1).astype(int)

                if len(np.unique(band_seg_idx)) == self.band_num:  # remove repeat band index
                    band_seg_idx = np.unique(band_seg_idx).astype(int)
                    print(f"real band_num which used for generating band segmentation: {band_num} ")
                    break
                band_num += 1
                assert band_num < self.freq_bins, f"invalid band_num {band_num}"
        else:
            # Change the following three parameters if you wish to use start_time different ERB scale.
            EarQ, minBW = 9.26449, 24.7
            bin_space = self.fs / (self.freq_bins - 1) / 2
            lowFreq, highFreq = 0, self.fs // 2
            # All the followFreqing expressions are derived in Apple TR #35, "An Efficient Implementation
            # of the Patterson-Holdsworth Cochlear Filter Bank."
            band_seg_idx = -(EarQ * minBW) + np.exp(
                (np.arange(1, self.band_num + 1)) * (-np.log(highFreq + EarQ * minBW) + \
                                                     np.log(lowFreq + EarQ * minBW)) / self.band_num) * (
                                   highFreq + EarQ * minBW)
            band_seg_idx = np.round(np.flip(band_seg_idx, axis=0) / bin_space)
            # make each band at least one bin
            for i in range(1, self.band_num):
                if band_seg_idx[i] <= band_seg_idx[i - 1]:
                    band_seg_idx[i] = band_seg_idx[i - 1] + 1
            band_seg_idx[0] = np.array(0).astype(int)
            band_seg_idx[-1] = np.array(self.freq_bins - 1).astype(int)

        return band_seg_idx.astype(int)


if __name__ == '__main__':
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    norm = matplotlib.colors.Normalize(vmin=-100, vmax=40)
    sr = 16000
    filter_num = 32
    window_len = NFFT = 256
    freq_bins = NFFT // 2 + 1
    wav = librosa.load("../../QA/wav_data/clean/TIMIT/TRAIN_DR1_FCJF0_SA1.wav", sr=sr)[0]
    spec = librosa.stft(wav, n_fft=NFFT, hop_length=window_len // 2, win_length=window_len)
    mag = np.abs(spec)  # (F,T)
    print("mag", mag.shape)

    band_method = "erb"
    AITD_filter = AITDFilterBank(band_num=filter_num, freq_bins=freq_bins, fs=sr, band_method=band_method)

    # erb 32
    # [  0   1   2   3   4   5   7   9  11  13  15  18  21  24  28  32  37  42
    #   48  54  61  70  79  89 100 113 127 143 161 181 203 256]
    # bark 32
    # [  0   3   4   6   9  11  14  17  19  22  25  28  31  34  38  42  47  52
    #   58  64  71  79  88  98 109 121 136 152 172 195 222 256]
    # mel_fbank 34
    # [  0   1   3   5   8  10  13  15  18  22  25  29  33  38  42  48  53  59
    #   66  73  80  89  97 107 117 128 140 153 167 183 199 216 235 256]

    FBank = AITD_filter.to_band_matrix.numpy()  # (F,M)[257, 32]
    FBank_inv = AITD_filter.inv_to_band_matrix.numpy()  # (M,F)[32, 257]
    print("FBannk, FBannk_inv", FBank.shape, FBank_inv.shape)
    # FBank = FBank / np.sum(FBank, axis=0, keepdims=True)  # 归一化,能够让Mel频谱在高频符合幅度谱
    # FBank_inv = FBank_inv / np.sum(FBank_inv, axis=1, keepdims=True)
    # 判断FBank_inv和FBank.T是否相等
    print(np.allclose(FBank_inv, FBank.T))  # True

    # 矩阵相乘获得mel频谱 --------------------------------------
    filter_banks = np.matmul(FBank.T, mag)  # (M,F)*(F,T)=(M,T)
    inverse_band = np.matmul(FBank_inv.T, filter_banks)  # (F,M)*(M,T)

    # ================ 画三角滤波器 ===========================
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle('AITD {}_fbank_nfft={}_num_filter={}'.format(band_method, NFFT, filter_num))

    plt.subplot(2, 2, 1)
    plt.title("FBank")
    fs_bin = sr // 2 / (NFFT // 2)  # 一个频点多少Hz
    x = np.linspace(0, freq_bins, freq_bins)
    plt.plot(x * fs_bin, FBank)
    plt.grid(linestyle='--')

    plt.subplot(2, 2, 2)
    plt.title("mag")
    plt.imshow(20 * np.log10(mag), cmap='jet', aspect='auto', origin='lower')  # , norm=norm,inverse_band
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('FreqBins/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(2, 2, 3)
    plt.title("FBank_spec")
    plt.imshow(20 * np.log10(filter_banks), cmap='jet', aspect='auto', origin='lower')  # , norm=norm
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('FreqBands/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(2, 2, 4)
    plt.title("FBank_spec_inverse")
    plt.imshow(20 * np.log10(inverse_band), cmap='jet', aspect='auto', origin='lower')  # , norm=norm
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('FreqBins/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()
