import numpy as np
import argparse

from BasicTools import wav_tools


def spatialize(src_path, brirs_path, azi_i, record_path):
    src, fs = wav_tools.read(src_path)

    brirs = np.load(brirs_path)

    record = wav_tools.brir_filter(x=src, brir=brirs[azi_i])
    wav_tools.write(record, fs, record_path)
    print(f'record is saved to {record_path} with fs={fs}')


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument(
        '--src-path', dest='src_path', required=True, type=str,
        help='path of source')
    parser.add_argument(
        '--brirs-path', dest='brirs_path', required=True, type=str,
        help='path of brirs')
    parser.add_argument(
        '--azi-i', dest='azi_i', required=True, type=int,
        help='index of azimuths')
    parser.add_argument(
        '--record-path', dest='record_path', required=True, type=str,
        help='path of record')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    spatialize(
        src_path=args.src_path,
        brirs_path=args.brirs_path,
        azi_i=args.azi_i,
        record_path=args.record_path)
