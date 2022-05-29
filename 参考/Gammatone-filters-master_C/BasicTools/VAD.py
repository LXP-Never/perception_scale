import argparse
import numpy as np
from . import wav_tools
from .parse_file import file2dict
from LocTools.add_log import add_log


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav', dest='wav_path', type=str,
                        default=None, help='path of the input file')
    parser.add_argument('--log', dest='log_path', type=str,
                        default=None, help='')
    parser.add_argument('--frame-len', dest='frame_len', type=float,
                        default=20,
                        help='frame length in millisecond, default to 20 ms')
    parser.add_argument('--shift-len', dest='frame_shift', type=float,
                        default=None, help='shift length in millisecond')
    parser.add_argument('--vad-log', dest='vad_log_path', type=str,
                        default=None, help='vad_log_path')
    args = parser.parse_args()
    return args


def plot_vad(ax, wav, fs, vad_labels, frame_len, frame_shift):
    ax.plot(np.arange(wav.shape[0])/fs, wav)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('amp')

    ax_vad = ax.twinx()
    ax_vad.plot(np.arange(vad_labels.shape[0])*frame_shift/fs+frame_len/fs/2,
                vad_labels, color='red')
    ax_vad.set_ylabel('vad')


def VAD_1wav(wav_path, frame_len, frame_shift):
    wav, fs = wav_tools.read_wav(wav_path)

    # make sure wav has two dimensions for convinient
    if len(wav.shape) == 1:
        wav = wav.reshape([-1, 1])
    n_channel = wav.shape[1]

    frame_len = round(frame_len/1e3)
    if frame_shift is None:
        frame_shift = round(frame_len/2)
    else:
        frame_shift = round(fs*frame_shift/1e3)

    vad_result_all = []
    for channel_i in range(n_channel):
        vad_result = wav_tools.VAD(wav[:, channel_i], frame_len, frame_shift)
        vad_result_all.append(vad_result)
    return vad_result_all


def main():
    args = parse_args()

    # single wav file or log file
    if args.wav_path is not None:
        tasks = [[args.wav_path, args.wav_path]]
    elif args.log_path is not None:
        wav_path_dict = file2dict(args.log_path)
        tasks = [[key, wav_path] for key, wav_path in wav_path_dict.items()]

    if args.log_path is not None:
        logger = open(args.log_path, 'w')
    else:
        logger = None

    for key, wav_path in tasks:
        vad_results = VAD_1wav(wav_path)
        add_log(logger, key, vad_results[0], )
        for i, vad_result in enumerate(vad_results[1:]):
            add_log(logger, key+f'_channel{i+1}', vad_result)

    if logger is not None:
        logger.close()


if __name__ == '__main__':
    main()
