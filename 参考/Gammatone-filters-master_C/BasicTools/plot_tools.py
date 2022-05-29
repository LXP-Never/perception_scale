import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib import ticker
import numpy as np
import os
import datetime
from PIL import Image
import pathlib
from functools import wraps

import GTF

from .scale import mel, erb
from . import fft

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['legend.fontsize'] = 10

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'Cambria'
plt.rcParams['mathtext.cal'] = 'Cambria'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['mathtext.sf'] = 'Cambria'
plt.rcParams['mathtext.tt'] = 'Cambria'
plt.rcParams['mathtext.it'] = 'Cambria:italic'

plt.rcParams['figure.dpi'] = 200

linestyles = (
    'solid',
    (0, (1, 1)),
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (3, 1, 1, 1, 1, 1)))



def get_figsize(n_row, n_col):
    width = 2.5+2*n_col
    height = 1+2*n_row
    return [width, height]


def subplots(n_row, n_col, **kwargs):
    if 'figsize' not in kwargs.keys():
        kwargs['figsize'] = get_figsize(n_row, n_col)
    fig, ax = plt.subplots(
        n_row, n_col, constrained_layout=True, **kwargs)
    return fig, ax


def line_collector(plot_func):
    @wraps(plot_func)
    def wrapped_plot_func(ax, *args, line_container=None, **kwargs):
        result = plot_func(ax, *args, **kwargs)
        if line_container is not None:
            if plot_func.__name__ == 'plot_contour':
                line_container.extend(result.collections)
                line_container.extend(ax.clabel(result, inline=True))
            else:
                line_container.extend(result)
        return result
    return wrapped_plot_func


@line_collector
def plot_line(ax, *args, line_container=None, **kwargs):
    return ax.plot(*args, **kwargs)


def plot_line2(ax, y1, y2, x1=None, x2=None, **kwargs):
    ax_twin = ax.twinx()
    if x1 is None:
        x1 = np.arange(y1.shape[0])
    ax.plot(x1, y1, **kwargs)

    if x2 is None:
        x2 = np.arange(y2.shape[0])
    ax_twin.plot(x1, y1, **kwargs)


# @line_collector
# def plot_scatter(ax,*args,line_container=None,**kwargs):
#     return ax.scatter(*args,**kwargs)


@line_collector
def plot_contour(ax, *args, is_label=False, line_container=None, **kwargs):
    contour_set = ax.contour(*args, **kwargs)
    return contour_set


def imshow(Z, ax=None, x_lim=None, y_lim=None, vmin=None, vmax=None, **kwargs):
    if ax is None:
        fig, ax = subplots(1, 1)
    else:
        fig = None

    if x_lim is None or y_lim is None:
        x_lim = [0, Z.shape[1]]
        y_lim = [0, Z.shape[0]]

    if vmin is None or vmax is None:
        vmin = np.min(Z)
        vmax = np.max(Z)
    Z_norm = (np.clip(Z, vmin, vmax)-vmin)/(vmax-vmin)

    basic_settings = {'cmap': cm.jet,
                      'aspect': 'auto'}
    basic_settings.update(kwargs)

    ax.imshow(
        Z_norm, extent=[*x_lim, *y_lim], **basic_settings, origin='lower')
    return fig, ax


def scatter_with_trend(x=None, y=None, ax=None, interp_order=2):
    from scipy.interpolate import interp1d
    if not isinstance(x, np.ndarray):
        raise Exception('x is not ndarray')
    if y is None:
        x, y = np.arange(x.shape[0]), x
    ax.plot(x, y, 'x')

    n_sample = x.shape[0]
    n_sample_interp = 10*n_sample
    interp_x = np.linspace(x[0], x[-1], n_sample_interp)
    interp_y = interp1d(x, y, kind=interp_order)(interp_x)
    ax.plot(interp_x, interp_y)


def plot_distribute(x=None, y=None, ax=None, fig=None):
    if x is None:
        x = np.arange(y.shape[0])
    y_mean = np.mean(y, axis=1)
    y_std = np.std(y, axis=1)

    if ax is None:
        fig, ax = subplots(1, 1)

    ax.plot(x, y_mean)
    ax.fill_between(x, y_mean-y_std/2, y_mean+y_std/2, alpha=0.5)
    return fig, ax


def plot_matrix(matrix, x=None, y=None, ax=None, fig=None, xlabel=None,
                ylabel=None, show_value=False, normalize=True, vmin=None,
                vmax=None, aspect='auto', cmap=None):
    """
    This function prints and plots matrix.
    Normalization can be applied by setting `normalize=True`.
    Args
        X: matrix
        xlabel, ylabel: labels of x-axis and y-axis
        show_value: display the correponding values of each square of images
        normalize: normalize Z to the range of [0, 1]
        vmin, vmax: the min- and max values to clip Z
        cmap: color map
    - normalize: whether normalization
    """

    if ax is None:
        fig, ax = subplots(1, 1)

    if x is None:
        x = np.arange(matrix.shape[1])
    if y is None:
        y = np.arange(matrix.shape[0])

    if cmap is not None:
        im = ax.pcolormesh(
            x, y, matrix, vmin=vmin, vmax=vmax, shading='auto',  cmap=cmap,
            rasterized=True)
    else:
        im = ax.pcolormesh(
            x, y, matrix, vmin=vmin, vmax=vmax, shading='auto',
            rasterized=True)

    # im = ax.imshow(matrix, interpolation='nearest', cmap=cmap,
    #                vmin=vmin, vmax=vmax, extent=[x_min, x_max, y_min, y_max],
    #                aspect=aspect, origin='lower')

    if fig is not None:
        plt.colorbar(im, ax=ax, shrink=0.6)

    # x_axis: colum  y_axis: row
    if show_value:
        fmt = '.2f' if normalize else 'd'
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, format(matrix[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if matrix[i, j] > thresh else "black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_surf(Z, x=None, y=None, ax=None, xlabel=None, ylabel=None,
              zlabel=None, zlim=None, vmin=None, vmax=None, cmap_range=None,
              fig=None, figsize=None, **kwargs):
    m, n = Z.shape
    if x is None:
        x = np.arange(Z.shape[0])
    if y is None:
        y = np.arange(Z.shape[1])

    X, Y = np.meshgrid(x, y)

    if cmap_range is not None:
        norm = mlp.colors.Normalize(vmin=cmap_range[0], vmax=cmap_range[1])
    else:
        norm = None
    basic_settings = {'cmap': cm.coolwarm,
                      'vmin': vmin,
                      'vmax': vmax,
                      'norm': norm}
    basic_settings.update(kwargs)

    if ax is None:
        if figsize is None:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        else:
            fig, ax = plt.subplots(
                subplot_kw={'projection': '3d'}, figsize=figsize)

    surf = ax.plot_surface(X, Y, Z.T, **basic_settings)
    ax.set_zlim(zlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if fig is not None:
        plt.colorbar(surf, shrink=0.5, pad=0.1)
        # plt.colorbar(surf, fraction=0.02, pad=0.12)
    return fig, ax


def plot_errorbar(*mean_std, ax=None, fig=None, xlabels=None, legend=None,
                  **kwargs):
    """plot error-bar figure given mean and std values, also support
    matplotlib figure settings
    Args:
    Returns:
        matplotlib figure
    """
    n_set = len(mean_std)
    n_var = mean_std[0][0].shape[0]

    if ax is None:
        fig, ax = subplots(1, 1)

    bar_width = 0.75/n_set
    bar_width_valid = 0.7/n_set
    for i, [mean, std] in enumerate(mean_std):
        x = np.arange(n_var) + bar_width*(i-(n_set-1)/2)
        ax.bar(x, mean, yerr=std, width=bar_width_valid)

    if xlabels is None:
        xlabels = [f'{i}' for i in range(n_var)]
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(n_var)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(xlabels))
    if legend is not None:
        ax.legend(legend)

    ax.set(**kwargs)
    return fig


def break_plot():
    x = np.random.rand(10)
    x[0] = -100
    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 1))
    ax1.plot(x)
    ax1.set_ylim((0, 1))

    ax2 = plt.subplot2grid((2, 2), (1, 1))
    ax2.plot(x)
    ax2.set_ylim(-120, -80)

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in ax coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom ax
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    savefig(fig, 'break_axis')


def plot_wav(wav, fs=None, label=None, ax_wav=None, plot_spec=False,
             ax_specgram=None, frame_len=320, frame_shift=160, yscale='mel',
             max_amp=None, cmap=None):
    """plot spectrogram of given len
    Args:
    """
    from . import fft

    if cmap is None:
        cmap = cm.jet

    wav_len = wav.shape[0]
    n_bin = int(frame_len/2)

    if fs is None:
        fs = 1
        t_scale = 1
        t_label = '采样点(n)'
        freq_label = '归一化频率'
    else:
        if wav_len < fs*0.05:
            t_scale = 1000
            t_label = '时间(ms)'
        else:
            t_scale = 1
            t_label = '时间(s)'
        freq_label = '频率(Hz)'

    fig = None
    # if ax_wav and ax_specgram are not specified
    if plot_spec:
        if ax_wav is None and ax_specgram is None:
            fig, ax = plt.subplots(2, 1, constained_layout=True)
            ax_wav, ax_specgram = ax
    else:
        if ax_wav is None:
            fig, ax_wav = subplots(1, 1)

    if ax_wav is not None:
        t = np.arange(wav_len)/fs*t_scale
        ax_wav.plot(t, wav, label=label, linewidth=1)
        ax_wav.set_xlabel(t_label)
        ax_wav.set_xlim([t[0], t[-1]])
        if max_amp is not None:
            ax_wav.set_ylim((-max_amp, max_amp))

    if ax_specgram is not None:
        specgram = fft.cal_stft(
            np.pad(wav, [frame_len, 0]),
            frame_len=frame_len, frame_shift=frame_shift)[:, :, 0]
        specgram_amp = 20*np.log10(np.abs(specgram)+1e-20)
        max_value = np.max(specgram_amp)
        min_value = max_value-60
        n_frame, n_bin = specgram_amp.shape
        t = np.arange(n_frame)*frame_shift/fs/t_scale
        freq_bins = np.arange(n_bin)/frame_len*fs/1e3
        imshow(ax=ax_specgram, Z=specgram_amp.T,
               x_lim=[0, t[-1]], y_lim=[0, freq_bins[-1]],
               vmin=min_value, vmax=max_value, origin='lower', cmap=cmap)
        ax_specgram.set_yscale('mel')
        ax_specgram.set_xlabel(t_label)
        ax_specgram.set_ylabel(freq_label)
        # ax_specgram.yaxis.set_major_formatter('{x:.1f}')
        #
    return fig, ax_wav, ax_specgram


def plot_spectrogram(wav, frame_len=1024, frame_shift=None, fs=None,
                     use_gtf=False, cfs=None,
                     freq_low=None, freq_high=None, n_band=None,
                     ax=None, fig=None, vmin=None, vmax=None,
                     return_specgram_amp=False):
    """
    Args:
        wav: waveform
        frame_len: frame length of fft, default to be 1024
        frame_shift: default to half of frame_len
    """

    # ensure wav is two-dimension array, [wav_len, n_chann]
    wav = np.squeeze(wav)
    if len(wav.shape) > 1:
        raise Exception('only single channel is supported')
    wav_len = wav.shape[0]

    if frame_shift is None:
        frame_shift = np.int(frame_len/2)

    if fs is None:
        fs = 1
        t_scale = 1
        t_label = 'sample(n)'
        freq_label = 'normalized freq'
    else:
        if wav_len < fs*0.05:
            t_scale = 1000
            t_label = 'time(ms)'
        else:
            t_scale = 1
            t_label = 'time(s)'
        freq_label = 'freq(kHz)'

    if ax is None:
        fig, ax = subplots(1, 1)
    if use_gtf:
        specgram_amp, cfs = GTF.cal_spectrogram(
            wav[:, np.newaxis], frame_len=frame_len, frame_shift=frame_shift,
            fs=fs, cfs=cfs, freq_low=freq_low, freq_high=freq_high,
            n_band=n_band, return_cfs=True)
        specgram_amp = specgram_amp[:, :, 0]
        freqs = cfs
    else:
        specgram = fft.cal_stft(
            np.pad(wav[:, np.newaxis], [[frame_len, frame_len], [0, 0]]),
            frame_len=frame_len, frame_shift=frame_shift)[:, :, 0]
        specgram_amp = 20*np.log10(np.abs(specgram)+1e-20)
        n_freq_bin = specgram_amp.shape[1]
        freqs = np.arange(n_freq_bin)/frame_len*fs

    if vmax is None:
        vmax = np.max(specgram_amp)
    if vmin is None:
        vmin = vmax-80
    n_frame = specgram_amp.shape[0]
    t = np.arange(n_frame)*frame_shift/fs/t_scale
    plot_matrix(
        specgram_amp.T, x=t, y=freqs,
        vmin=vmin, vmax=vmax, cmap=plt.cm.coolwarm,
        ax=ax, fig=fig)
    ax.set_xlabel(t_label)
    ax.set_yscale('mel')
    ax.yaxis.set_major_formatter(lambda x, pos: f'{x*1e-3}')
    ax.set_ylabel(freq_label)
    if return_specgram_amp:
        return fig, ax, specgram_amp
    else:
        return fig, ax


def plot_spectrum(wav, fs=None, ax=None, fig=None):
    """
    Args:
        wav: waveform
    """

    # ensure wav is two-dimension array, [wav_len, n_chann]
    if len(wav.shape) == 1:
        wav = wav[:, np.newaxis]
    wav_len, n_chann = wav.shape

    if fs is None:
        fs = 1
        freq_label = 'normalized freq'
    else:
        freq_label = 'freq(Hz)'

    if ax is None:
        fig, ax = subplots(1, 1)

    spectrum = np.fft.fft(wav)
    spectrum_amp = 20*np.log10(np.abs(spectrum)+1e-20)
    n_freq_bin = spectrum.shape[0]
    n_freq_bin_valid = np.int(np.floor(n_freq_bin/2)+1)
    freq_bins = np.arange(n_freq_bin_valid)/n_freq_bin*fs
    ax.plot(freq_bins, spectrum_amp[n_freq_bin_valid])
    ax.set_xscale('mel')
    ax.set_xlabel(freq_label)
    ax.set_ylabel('dB')
    return fig, ax


def plot_break_axis(x1, x2):
    # how big to make the diagonal lines in axes coordinates
    d = .015
    fig, ax = plt.subplots(2, 1, sharex=True)
    [x1_min, x1_max] = [np.min(x1), np.max(x1)]
    if x1_min == x1_max:
        tmp = x1_max/10.
    else:
        tmp = (x1_max-x1_min)/10
    ax[0].plot(x1)
    ax[0].set_ylim((x1_min-tmp, x1_max+tmp))

    [x2_min, x2_max] = [np.min(x2), np.max(x2)]
    if x2_min == x2_max:
        tmp = x2_max/10.
    else:
        tmp = (x2_max-x2_min)/10
    ax[1].plot(x2)
    ax[1].set_ylim((x2_min-tmp, x2_max+tmp))

    # set spines
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    #
    ax[0].xaxis.tick_top()
    ax[1].xaxis.tick_bottom()

    ax[1].tick_params(labeltop=False)

    # draw diagonal marks where axises break
    kwargs = dict(transform=ax[0].transAxes, color='k', clip_on=False)
    ax[0].plot((-d, +d), (-d, +d), **kwargs)
    ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax[1].transAxes)
    ax[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    return fig


def annoate(ax, text, xy, xytext, xycoords='data', textcoords='data',
            va='center', ha='center', **args):
    #
    raise Exception('unfinished')
    renderer = ax.get_figure().canvas.get_renderer()
    font_props = mlp.font_manager.FontProperties()
    # get all texts
    pre_text_rect_all = []
    for text in ax.texts:
        text_rect = text.get_window_extent()
        pre_text_rect_all.append(text_rect)
    #
    text_rect = renderer.get_text_width_height_descent(
        text, prop=font_props, ismath=None)


class GIF:
    # GIF-making class, a encapsulation of matplotlib functions
    def __init__(self):
        self.artists = []  # list of objects of line, image ...

    def add(self, artist):
        self.artists.append(artist)

    def save(self, fig_path, fig, fps=60):
        """save to gif file
        Args:
            fpath: file path of gif
            fig: figure obj that hold artist on
            fps: frame per second
        Returns:
            None
        """
        ani = animation.ArtistAnimation(fig, self.artists, interval=1./fps*1e3)
        # writer = animation.FFMpegWriter(fps=fps)
        ani.save(fig_path, fps=fps, writer='pillow')


def savefig(fig, fig_name=None, fig_dir='./images'):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # use date as name if name is not defined
    if fig_name is None:
        fig_name = '{0.year}_{0.month}_{0.day}.png'.format(
                                            datetime.date.today())

    # matplotlib_fig_suffixs = ['.eps', '.pdf', '.pgf', '.png', '.ps', '.raw',
    #                           '.rgba', '.svg', '.svgz']

    # check whether name has suffix
    stem = pathlib.PurePath(fig_name).stem
    suffix = pathlib.PurePath(fig_name).suffix
    if suffix == '':
        suffix = '.png'

    if suffix == '.jpg':  # not support in matplotlib
        fig_path = os.path.join(dir, ''.join((stem, '.png')))
        fig.savefig(fig_path)
        Image.open(fig_path).convert('RGB').save(
                                os.path.join(fig_dir, ''.join((stem, '.jpg'))))
        os.remove(fig_path)
    else:
        fig_path = os.path.join(fig_dir, ''.join((stem, suffix)))
        fig.savefig(fig_path)

    if True:
        print('{}{} is saved in {}'.format(stem, suffix, fig_dir))


def test_bar():
    mean_std_all = [[np.random.normal(size=5), np.random.rand(5)]
                    for i in range(5)]
    fig = plot_errorbar(
        *mean_std_all, ylabel='ylabel',
        xticklabels=['label{}'.format(i) for i in range(4)])
    savefig(fig, name='bar.png', dir='images/plot_tools/')


def test_gif():
    gif = GIF()
    line_container = []
    fig, ax = plt.subplots(1, 1)
    for i in range(5):
        # lines = []
        for line_i in range(np.random.randint(1, 4)):
            plot_line(ax, np.random.rand(10),
                      line_container=line_container)
            # lines.extend(line_tmp)
        gif.add(line_container)
    gif.save('images/plot_tools/gif_example.gif', fig, fps=10)


def test_imshow():
    import wav_tools
    import fft
    # from auditory_scale import mel

    x, fs = wav_tools.wav_read('resource/tar.wav')
    stft, t, freq = fft.cal_stft(x, fs=fs, frame_len=np.int(fs*50e-3))
    ax = imshow(x=t, y=freq, Z=20*np.log10(np.abs(stft)))
    ax.set_yscale('mel')
    fig = ax.get_figure()
    savefig(fig, name='imshow', dir='images/plot_tools')


def test_plot_wav_spec():
    import wav_tools
    x1, fs = wav_tools.wav_read('resource/tar.wav')
    x2, fs = wav_tools.wav_read('resource/inter.wav')

    fig = plot_wav(wav_all=[x1, x2], label_all=['tar', 'inter'], fs=fs,
                   frame_len=1024, shift_len=512, yscale='mel')
    savefig(fig, name='wav_spec', dir='./images/plot_tools')


def test_plot_line2():
    y1 = np.random.rand(10)
    y2 = np.random.rand(10)+10
    fig, ax = plt.subplots(1, 1)
    plot_line2(ax, y1, y2)
    savefig(fig, name='plot_line2', dir='images/plot_tools')


if __name__ == "__main__":

    # test_gif()
    # test_bar()
    # test_imshow()
    # break_plot()
    # test_plot_wav_spec()
    test_plot_line2()
