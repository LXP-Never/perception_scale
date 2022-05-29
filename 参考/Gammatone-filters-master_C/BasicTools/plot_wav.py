"""
terminal interface for plot_tools.plot_wav
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse

from . import wav_tools
from . import plot_tools


def plot_wav(wav_path, frame_len=320, ax=None, fig=None, fig_path=None,
             mix_channel=False, cmap=None, dpi=100, interactive=False):

    wav, fs = wav_tools.read(wav_path)
    # make wav 2d ndarray
    if len(wav.shape) == 1:
        wav = wav[:, np.newaxis]
    n_chann = wav.shape[1]

    if ax is None:
        if mix_channel:
            fig, ax = plot_tools.subplots(2, 1, sharex=True)
        else:
            fig, ax = plot_tools.subplots(2, n_chann, sharex=True)

    if len(ax.shape) == 1:
        ax = ax[:, np.newaxis]
    if mix_channel:
        ax = np.repeat(ax, n_chann, axis=1)

    max_amp = np.max(np.abs(wav))
    for chann_i in range(n_chann):
        plot_tools.plot_wav(
            wav=wav[:, chann_i], fs=fs, frame_len=frame_len,
            label=f'channel_{chann_i}', ax_wav=ax[0, chann_i],
            ax_specgram=ax[1, chann_i], max_amp=max_amp, cmap=cmap)
    if mix_channel:
        ax[0, chann_i].legend()

    if fig_path is not None:
        fig.savefig(fig_path, dpi=dpi)

    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav-path', dest='wav_path', type=str, nargs='+',
                        required=True,  help='path of wav file')
    parser.add_argument('--fig-path', dest='fig_path', type=str,
                        default=None, help='path of figure to be saved')
    parser.add_argument('--plot-spec', dest='plot_spec', type=str,
                        default='True', choices=['true', 'false'],
                        help='whether to plot the spectrum')
    parser.add_argument('--cmap', dest='cmap', type=str, default=None,
                        help='')
    parser.add_argument('--frame-len', dest='frame_len', type=int,
                        default=320, help='')
    parser.add_argument('--mix-channel', dest='mix_channel', type=str,
                        default='false', choices=['true', 'false'], help='')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        default='false', choices=['true', 'false'], help='')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    n_wav = len(args.wav_path)

    cmap = None
    if args.cmap is not None:
        cmap = plt.get_cmap(args.cmap)

    if args.mix_channel == 'true':
        fig, ax = plot_tools.subplots(2, n_wav, sharex=True)
        for wav_i in range(n_wav):
            plot_wav(
                wav_path=args.wav_path[wav_i], ax=ax[:, wav_i],
                frame_len=args.frame_len, mix_channel=True, cmap=cmap)
    else:
        max_chann_num = 1
        for wav_path in args.wav_path:
            tmp_wav, fs = wav_tools.read(wav_path)
            if len(tmp_wav.shape) == 2 and max_chann_num < tmp_wav.shape[1]:
                max_chann_num = tmp_wav.shape[1]
        fig, ax = plot_tools.subplots(2, n_wav*max_chann_num, sharex=True)
        for wav_i in range(n_wav):
            ax_slice = slice(wav_i*max_chann_num, (wav_i+1)*max_chann_num)
            plot_wav(
                wav_path=args.wav_path[wav_i], ax=ax[:, ax_slice],
                frame_len=args.frame_len, mix_channel=False, cmap=cmap)

    n_row, n_col = ax.shape
    if n_row > 1:
        # remove xlabel of wav if spec is ploted below
        for i in range(n_row-1):
            for ax_tmp in ax[i]:
                ax_tmp.set_xlabel('')

    if args.interactive == 'true':
        plt.show()

    if args.fig_path is not None:
        fig.savefig(args.fig_path, dpi=args.dpi)
        print(f'fig is saved to {args.fig_path}')


if __name__ == '__main__':
    main()
