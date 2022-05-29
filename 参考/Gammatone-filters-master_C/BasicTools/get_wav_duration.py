import argparse
import numpy as np

from . import wav_tools
from .get_file_path import get_file_path
from . import plot_tools


def get_wav_duration(wav_path):
    wav, fs = wav_tools.read(wav_path)
    duration = wav.shape[0]/fs
    return duration


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--wav_path', required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.wav_path.endswith('.wav'):
        duration = get_wav_duration(args.wav_path)
        wav_paths = [args.wav_path]
        durations = [duration]
        duration_sum = duration
    else:
        wav_paths = get_file_path(
            args.wav_path, suffix='.wav', is_absolute=True)
        durations = [get_wav_duration(wav_path) for wav_path in wav_paths]
        duration_sum = np.sum(durations)
        fig, ax = plot_tools.subplots(1, 1)
        hist, bin_edges = np.histogram(durations)
        hist = hist/np.sum(hist)
        ax.bar((bin_edges[:-1]+bin_edges[1:])/2, hist)
        fig.savefig('hist.png')

    for wav_path, duration in zip(wav_paths, durations):
        print(f'{wav_path}: {duration:.2f}')
    print('---------------------------------')
    print(f'total duration: {duration_sum}')
