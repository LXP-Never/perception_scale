import argparse
import numpy as np
import matplotlib.pyplot as plt
from GTF import GTF
from BasicTools import wav_tools


def cal_spectrogram(x, frame_len, frame_shift, fs,
                    cfs=None, freq_low=None, freq_high=None, n_band=None,
                    return_cfs=False, fig_path=None, dpi=100):
    """return shape [n_frame, n_band, n_channel]
    """
    if len(x.shape) == 2:
        n_channel = x.shape[1]
    else:
        n_channel = 1
        x = np.reshape(x, [-1, 1])

    gtf_filter = GTF(
        fs, cfs=cfs, freq_low=freq_low, freq_high=freq_high, n_band=n_band)
    # [x_len, n_band, n_channle]
    x_filtered = np.transpose(gtf_filter.filter(x), [1, 0, 2])
    spectrogram = wav_tools.frame_data(
        x_filtered, frame_len=frame_len, frame_shift=frame_shift)
    spectrogram = 10 * np.log10(np.sum(spectrogram ** 2, axis=1))
    value_min, value_max = np.min(spectrogram), np.max(spectrogram)

    if fig_path is not None:
        n_frame = spectrogram.shape[0]
        t_tick = (np.arange(n_frame) * frame_shift + frame_len / 2) / fs
        imshow_settings = {'aspect': 'auto',
                           'cmap': 'jet',
                           'origin': 'lower',
                           'vmin': value_min,
                           'vmax': value_max,
                           'extent': [0, t_tick[-1], 0, n_band - 1]}
        fig, ax = plt.subplots(1, n_channel, figsize=[8, 6],
                               sharex=True, sharey=True)
        if n_channel == 1:
            ax = [ax]
        for channel_i in range(n_channel):
            ax[channel_i].imshow(spectrogram[:, :, channel_i].T,
                                 **imshow_settings)
            ax[channel_i].set_title(f'channel:{channel_i}')

        y_tick_pos_all = np.arange(0, n_band, 4)
        y_tick_label_all = map(lambda x: f'{x / 1000:.2f}',
                               gtf_filter.cfs[y_tick_pos_all])
        ax[0].set_yticks(y_tick_pos_all)
        ax[0].set_yticklabels(y_tick_label_all)
        ax[0].set_ylabel('fs(kHz)')
        ax[0].set_xlabel('t(s)')
        fig.savefig(fig_path)
    if return_cfs:
        return spectrogram, cfs
    else:
        return spectrogram


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav-path', dest='wav_path', required=True, type=str)
    parser.add_argument('--frame-len', dest='frame_len', type=float,
                        default=0.02, help='second')
    parser.add_argument('--shift-len', dest='frame_shift', type=float,
                        default=0.01, help='second')
    parser.add_argument('--freq-low', dest='freq_low', required=True,
                        type=int, help='')
    parser.add_argument('--freq-high', dest='freq_high', required=True,
                        type=int, help='')
    parser.add_argument('--band-num', dest='n_band', required=True,
                        type=int, help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    x, fs = wav_tools.read_wav(args.wav_path)
    frame_len = int(fs * args.frame_len)
    frame_shift = int(fs * args.frame_shift)
    cal_spectrogram(
        x, frame_len, frame_shift, fs,
        args.freq_low, args.freq_high, args.n_band,
        args.fig_path, args.dpi)


if __name__ == '__main__':
    main()
