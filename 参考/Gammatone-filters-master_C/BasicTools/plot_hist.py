import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from .parse_file import file2dict


def plot_hist(log_path, ax=None, fig=None, n_bin=-1, xlim=None, var_i=0,
              file_name_required=None, x_label=None, fig_path=None,
              interactive=False):
    log = file2dict(log_path, numeric=True)
    values = None
    if file_name_required is None:
        values = np.concatenate(
            list(log.values()),  # log.values [n_sample, n_var]
            axis=0)[:, var_i]
    else:
        for file_path in log.keys():
            file_name = os.path.basename(file_path).split('.')[0]
            if file_name == file_name_required:
                values = log[file_path][:, var_i]
                break
    if values is None:
        print('Data not found')
        return

    if n_bin == -1:  # occurence frequency of each values
        centers, freqs = np.unique(values, return_counts=True)
        freqs = freqs/np.sum(freqs)*100
        bin_widths = centers[1:]-centers[:-1]
        if bin_widths.shape[0] > 0:
            bin_width = np.min(bin_widths)
        else:
            bin_width = 1
    else:  # statistic the number of values within each bin
        if xlim is not None:
            min_value, max_value = xlim
        else:
            min_value, max_value = np.min(values), np.max(values)
        bin_edges = np.linspace(min_value, max_value, n_bin+1)
        bin_width = bin_edges[1] - bin_edges[0]

        freqs, _ = np.histogram(values, bins=bin_edges, density=True)
        freqs = freqs/np.sum(freqs)*100
        centers = (bin_edges[1:] + bin_edges[:-1])/2

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.bar(centers, freqs, width=bin_width, edgecolor='black')
    ax.set_ylabel('Percentage(%)')
    if x_label is not None:
        ax.set_xlabel(x_label)

    if fig_path is not None:
        fig.savefig(fig_path)
        print(f'fig is saved to {fig_path}')

    if interactive:
        plt.show()
    return ax, fig


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', required=True, type=str,
                        help='path of the input file')
    parser.add_argument('--n-bin', dest='n_bin', type=int, default=-1,
                        help='')
    parser.add_argument('--xlim', dest='xlim', type=float, nargs=2,
                        help='')
    parser.add_argument('--file-name', dest='file_name_required', type=str,
                        default=None, help='')
    parser.add_argument('--x-label', dest='x_label', type=str, default=None,
                        help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        choices=['true', 'false'], default='false', help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    plot_hist(log_path=args.log_path,
              n_bin=args.n_bin,
              xlim=args.xlim,
              file_name_required=args.file_name_required,
              x_label=args.x_label,
              fig_path=args.fig_path,
              interactive=args.interactive == 'true')


if __name__ == '__main__':
    main()
