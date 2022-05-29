import argparse
from BasicTools import wav_tools


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument(
        '--wav', dest='wav_path', required=True, nargs='+', type=str,
        help='')
    parser.add_argument(
        '--frame-len', dest='frame_len', type=float, default=-1,
        help='frame length in s')
    parser.add_argument(
        '--frame-shiftt', dest='frame_shift', type=float, default=None,
        help='frame length in s')
    parser.add_argument('--log-path', dest='log_path', type=str, default=None,
                        help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    tar_path = args.wav_path[0]
    tar, fs_tar = wav_tools.read(tar_path)

    if len(args.wav_path) > 1:
        inter_path = args.wav_path[1]
        inter, fs_inter = wav_tools.read(inter_path)
        if fs_tar != fs_inter:
            raise Exception('sample frequency do not match')
        else:
            fs = fs_tar
    else:
        tar, inter = tar[:, 0], tar[:, 1]
        fs = fs_tar

    frame_len = int(args.frame_len * fs)
    if args.frame_shift is not None:
        frame_shift = int(args.frame_shift * fs)
    else:
        frame_shift = None

    snrs = wav_tools.cal_snr(
        tar=tar, inter=inter, frame_len=frame_len, frame_shift=frame_shift)

    txt = '; '.join([f'{snr:.2f}' for snr in snrs])
    if args.log_path is not None:
        with open(args.log_path, 'x') as snr_logger:
            snr_logger.write(txt)
    else:
        print(txt)


if __name__ == '__main__':
    main()
