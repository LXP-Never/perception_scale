import argparse
import numpy as np


def cal_DRR(ir, direct_ir, align_ir=False):

    full_energy = np.sum(ir**2)
    direct_energy = np.sum(direct_ir**2)
    reverb_energy = full_energy - direct_energy
    if reverb_energy < 1e-10:
        drr = np.Inf
    else:
        drr = 10*np.log10(direct_energy/reverb_energy)
    return drr


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--ir', dest='ir_path', required=True,
                        type=str, help='')
    parser.add_argument('--direct-ir', dest='direct_ir_path',
                        type=str, default=None, help='direct ir path')
    parser.add_argument('--align-ir', dest='align_ir',
                        type=str, default='false', choices=('true', 'false'),
                        help='')
    args = parser.parse_args()
    return args


def main():
    from .. import wav_tools
    args = parse_args()
    ir, fs = wav_tools.read(args.ir_path)
    direct_ir, fs_1 = wav_tools.read(args.direct_ir_path)
    if fs != fs_1:
        raise Exception('sample rate do not match')
    if len(ir.shape) == 1:
        n_channel = 1
        ir = ir.reshape([-1, 1])
        direct_ir = direct_ir.reshape([-1, 1])
    else:
        n_channel = ir.shape[1]

    drr = []
    for channel_i in range(n_channel):
        drr_tmp = cal_DRR(ir=ir[:, channel_i],
                          direct_ir=direct_ir[:, channel_i],
                          align_ir=args.align_ir == 'true')
        drr.append(drr_tmp)
    return drr


if __name__ == '__main__':
    main()
