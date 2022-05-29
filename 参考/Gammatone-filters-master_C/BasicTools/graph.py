import argparse
import numpy as np
import matplotlib.pyplot as plt


def graph(eq_str, xlim=None, step=0.1, ax=None, fig_path=None):

    # eq = parser.expr(eq_str).compile()

    if xlim is None:
        xlim = [-10, 10]
    x_all = np.arange(xlim[0], xlim[1], step)
    y_all = np.zeros(x_all.shape)
    for i, x in enumerate(x_all):
        try:
            y = eval(eq_str)
        except Exception:
            y = np.nan
        y_all[i] = y

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    ax.plot(x_all, y_all, label=eq_str)
    ax.set_xlim(xlim[0:2])
    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--eq', dest='eq_str', required=True, nargs='+',
                        type=str, help='')
    parser.add_argument('--xlim', dest='xlim', nargs='+',
                        type=int, default=[-10, 10], help='')
    parser.add_argument('--step', dest='step', type=float,
                        default=0.1, help='')
    parser.add_argument('--fig-path', dest='fig_path', required=True,
                        type=str, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    fig, ax = plt.subplots(1, 1)
    for eq_str in args.eq_str:
        graph(eq_str, args.xlim, args.step, ax)
    ax.legend()
    fig.savefig(args.fig_path)


if __name__ == '__main__':
    main()
