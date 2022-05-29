import numpy as np
import matplotlib.pyplot as plt
import argparse

from .plot_tools import plot_matrix
from .plot_tools import plot_surf
from .plot_tools import get_figsize

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 200


def load_matrix(matrix_path):
    suffix = matrix_path.split('.')[-1]
    if suffix == 'npy':
        matrix = np.load(matrix_path)
    elif suffix == 'txt':
        with open(matrix_path, 'r') as file_obj:
            lines = file_obj.readlines()
        matrix = [[float(item) for item in line.strip().split()]
                  for line in lines if len(line.strip()) > 0]
        matrix = np.asarray(matrix)
    elif suffix == 'dat':
        with open(matrix_path, 'rb') as file_obj:
            lines = file_obj.readlines()
        matrix = [[float(item) for item in line.strip().split()]
                  for line in lines if len(line.strip()) > 0]
        matrix = np.asarray(matrix)

    return matrix


def plot_matrix_image(args):

    n_matrix = len(args.matrix_paths)
    if args.horizon == 'true':
        n_fig_row, n_fig_col = 1, n_matrix
    else:
        n_fig_row, n_fig_col = n_matrix, 1
    figsize = get_figsize(n_fig_row, n_fig_col)
    fig, ax = plt.subplots(
        1, n_matrix, figsize=figsize, constrained_layout=True,
        sharex=True, sharey=True)
    for matrix_i, matrix_path in enumerate(args.matrix_paths):
        matrix = load_matrix(matrix_path)
        if args.transpose == 'true':
            matrix = matrix.T

        if args.xlim is not None:
            x = np.linspace(args.xlim[0], args.xlim[1], matrix.shape[0])
        else:
            x = None
        if args.ylim is not None:
            y = np.linspace(args.ylim[0], args.ylim[1], matrix.shape[1])
        else:
            y = None

        fig_colorbar = None
        if matrix_i == n_matrix-1:
            fig_colorbar = fig
        plot_matrix(
            matrix, fig=fig_colorbar, ax=ax[matrix_i],
            xlabel=args.xlabel, ylabel=args.ylabel,
            x=x, y=y,
            show_value=args.show_value == 'true',
            normalize=args.normalize == 'true',
            vmin=args.vmin, vmax=args.vmax,
            aspect=args.aspect, cmap=plt.get_cmap(args.cmap))
    return fig, ax


def plot_matrix_surf(args):
    for matrix_i, matrix_path in enumerate(args.matrix_paths):
        matrix = load_matrix(matrix_path)
        if args.transpose == 'true':
            matrix = matrix.T

        if args.xlim is not None:
            x = np.linspace(args.xlim[0], args.xlim[1], matrix.shape[0])
        else:
            x = None
        if args.ylim is not None:
            y = np.linspace(args.ylim[0], args.ylim[1], matrix.shape[1])
        else:
            y = None

        fig, ax = plot_surf(
            Z=matrix.T, x=x, y=y, zlim=args.zlim,
            xlabel=args.xlabel, ylabel=args.ylabel,
            vmin=args.vmin, vmax=args.vmax,
            cmap_range=args.cmap_range, zlabel=args.zlabel)
        if args.view is not None:
            ax.view_init(azim=args.view[0], elev=args.view[1])
    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--matrix-path', dest='matrix_paths', nargs='+',
                        type=str, required=True, help='npy, txt, dat')
    parser.add_argument('--type', dest='type', type=str,
                        choices=['image', 'surf'], default='image',
                        help='')
    parser.add_argument('--aspect', dest='aspect', type=str,
                        choices=['equal', 'auto'], default='auto',
                        help='')
    parser.add_argument('--transpose', dest='transpose', type=str,
                        choices=['true', 'false'], default='false',
                        help='')
    parser.add_argument('--vmin', dest='vmin', type=float, default=None,
                        help='')
    parser.add_argument('--vmax', dest='vmax', type=float, default=None,
                        help='')
    parser.add_argument('--cmap', dest='cmap', type=str, default='coolwarm',
                        help='')
    parser.add_argument('--cmap-range', dest='cmap_range', type=float, nargs=2,
                        default=None, help='range for colormap')
    parser.add_argument('--xlim', dest='xlim', type=float, default=None,
                        nargs=2, help='')
    parser.add_argument('--xlabel', dest='xlabel', type=str, default=None,
                        help='')
    parser.add_argument('--ylim', dest='ylim', type=float, default=None,
                        nargs=2, help='')
    parser.add_argument('--ylabel', dest='ylabel', type=str, default=None,
                        help='')
    parser.add_argument('--zlim', dest='zlim', type=float, default=None,
                        nargs=2, help='')
    parser.add_argument('--zlabel', dest='zlabel', type=str, default=None,
                        help='')
    parser.add_argument('--horizon', dest='horizon', type=str, default='true',
                        choices=['true', 'false'], help='')
    parser.add_argument('--view', dest='view', type=float, default=None,
                        nargs='+', help='')
    parser.add_argument('--show-value', dest='show_value', type=str,
                        choices=['true', 'false'], default='false',
                        help='')
    parser.add_argument('--normlize', dest='normalize', type=str,
                        choices=['true', 'false'], default='false',
                        help='')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        choices=['true', 'false'], default='false',
                        help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.type == 'image':
        fig, ax = plot_matrix_image(args)
    else:
        fig, ax = plot_matrix_surf(args)

    if args.fig_path is not None:
        fig.savefig(args.fig_path)

    if args.interactive == 'true':
        plt.show()


if __name__ == '__main__':
    main()
