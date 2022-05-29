import argparse
import math
import matplotlib.pyplot as plt
from GTF import GTF
from BasicTools import wav_tools


def filter(x, fs, freq_low, freq_high, n_band, result_dir, fig_path=None):
    gtf_filter = GTF(fs, freq_low=freq_low, freq_high=freq_high, n_band=n_band)
    x_filtered = gtf_filter.filter(x)
    for band_i in range(n_band):
        band_path = f'{result_dir}/{band_i}.wav'
        wav_tools.write_wav(x_filtered[band_i], fs, band_path)

    if fig_path is not None:
        n_band_per_col = math.ceil(n_band/2)
        fig, ax = plt.subplots(n_band_per_col, 2, sharex=True, sharey=True)
        for band_i in range(n_band):
            col_i = math.floor(band_i/n_band_per_col)
            row_i = band_i - col_i*n_band_per_col
            ax[row_i, col_i].plot(x_filtered[band_i])
            ax[row_i, col_i].set_ylabel(f'{band_i}')
            ax[row_i, col_i].tick_params(axis='both', which='both',
                                         bottom=False, top=False,
                                         labelbottom=False, left=False,
                                         right=False, labelleft=False)
            ax[row_i, col_i].spines['top'].set_visible(False)
            ax[row_i, col_i].spines['bottom'].set_visible(False)
            ax[row_i, col_i].spines['left'].set_visible(False)
            ax[row_i, col_i].spines['right'].set_visible(False)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        fig.savefig(fig_path, dpi=200)


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav-path', dest='wav_path', required=True, type=str)
    parser.add_argument('--freq-low', dest='freq_low', type=int, help='')
    parser.add_argument('--freq-high', dest='freq_high', type=int, help='')
    parser.add_argument('--band-num', dest='n_band', type=int, help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    parser.add_argument('--result-dir', dest='result_dir', type=str, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    x, fs = wav_tools.read_wav(args.wav_path)
    filter(x, fs, args.freq_low, args.freq_high, args.n_band, args.result_dir,
           args.fig_path)


if __name__ == '__main__':
    main()
