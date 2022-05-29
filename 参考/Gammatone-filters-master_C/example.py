import numpy as np
import matplotlib.pyplot as plt
from GTF import GTF
from BasicTools import wav_tools


def main():

    wav, fs = wav_tools.read('record.wav')
    gt_filter = GTF(fs, freq_low=80, freq_high=5e3, n_band=32)
    wav_band_all_py = gt_filter.filter_py(wav)
    np.save('wav_band_all_py.npy', wav_band_all_py)
    # wav_band_all_py = np.load('wav_band_all_py.npy')
    print(np.max(wav_band_all_py))

    wav_band_all = gt_filter.filter(wav)
    print(np.max(wav_band_all))

    for band_i in range(32):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(wav_band_all[band_i, :, 0].T)
        ax[0].plot(wav_band_all_py[band_i, :, 0].T)
        ax[0].set_xlim([5000, 5050])

        ax[1].plot(wav_band_all_py[band_i, :, 0].T)
        ax[1].plot(wav_band_all[band_i, :, 0].T)
        ax[1].set_xlim([5000, 5050])

        fig.savefig(f'../images/eg_{band_i}.png')
        plt.close(fig)


if __name__ == "__main__":
    main()
