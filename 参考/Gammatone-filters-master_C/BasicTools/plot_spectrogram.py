import argparse
import matplotlib.pyplot as plt


from . import wav_tools
from . import plot_tools


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav-path', dest='wav_path', required=True, type=str,
                        nargs='+', help='')
    parser.add_argument('--label', dest='label', default=None, type=str,
                        nargs='+', help='')
    parser.add_argument('--frame-len', dest='frame_len', type=int, default=20,
                        help='frame length in ms')
    parser.add_argument('--frame-shift', dest='frame_shift', type=int,
                        default=10, help='frame shift in ms')
    parser.add_argument('--linewidth', dest='linewidth', type=int,
                        default=2, help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        default='false', choices=['true', 'false'], help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    n_wav = len(args.wav_path)
    if args.label is None:
        label = [None for i in range(n_wav)]
    else:
        label = args.label

    wav_all = []
    fs_all = []
    max_chann_num = 0
    for wav_path in args.wav_path:
        wav, fs = wav_tools.read(args.wav_path[0])
        wav_all.append(wav)
        fs_all.append(fs)
        if len(wav.shape) > 1 and max_chann_num < wav.shape[1]:
            max_chann_num = wav.shape[1]
    n_wav = len(wav_all)

    fig, ax = plot_tools.subplots(
        max_chann_num, n_wav, sharex=True, sharey=True)

    for wav_i in range(n_wav):
        n_chann = wav_all[wav_i].shape[1]
        ax[0, wav_i].set_title(label[wav_i])
        for chann_i in range(n_chann):
            plot_tools.plot_spectrogram(
                wav=wav_all[wav_i], fs=fs_all[wav_i],
                frame_len=args.frame_len,
                frame_shift=args.frame_shift,
                ax=ax[chann_i, wav_i], fig=fig)

    if args.fig_path is not None:
        fig.savefig(args.fig_path)
        print(f'fig is saved to {args.fig_path}')

    if args.interactive == 'true':
        plt.show()


if __name__ == '__main__':
    main()
