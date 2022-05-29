import numpy as np
import os
import matplotlib.pyplot as plt
from GTF import GTF as gtf_proposed
from gammatone import filters as gtf_ref

fig_dir = 'images/validate/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


def savefig(fig, fig_name):
    fig_fpath = os.path.join(fig_dir, fig_name)
    fig.savefig(fig_fpath)


def compare_cfs():
    fs = 1e3
    n_band = 16
    freq_low = 70
    freq_high = 7000
    gtf_obj = gtf_proposed(fs, cf_low=freq_low, freq_high=freq_high, n_band=n_band)
    cfs_proposed = gtf_obj.cfs
    bws_proposed = gtf_obj.cal_bw(cfs_proposed)

    cfs_ref = gtf_ref.erb_space(low_freq=freq_low, high_freq=freq_high, num=n_band)[::-1]
    bws_ref = gtf_obj.cal_bw(cfs_ref)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(np.arange(n_band), cfs_proposed, yerr=bws_proposed/2, linewidth=2, label='Todd')
    ax.errorbar(np.arange(n_band)+n_band/100, cfs_ref, yerr=bws_ref/2, linewidth=2, label='Detly')
    ax.set_xlabel('freq_band')
    ax.set_ylabel('freq(Hz)')
    ax.legend()
    fig.savefig(f'images/validate/cfs_n{n_band}.png', dpi=100)


def compare_ir():
    fs = 16e3
    # impulse
    x_len = np.int16(fs)
    x = np.zeros(x_len)
    x[1] = 1

    gtf_obj = gtf_proposed(fs, cf_low=100, cf_high=2000, n_band=4)
    irs = gtf_obj.filter(x)
    fig = gtf_obj.plot_ir_spec(irs[:, :1000])
    savefig(fig, 'proposed.png')

    coefs = gtf_ref.make_erb_filters(fs, gtf_obj.cfs)
    irs_ref = gtf_ref.erb_filterbank(x, coefs)
    fig = gtf_obj.plot_ir_spec(irs[:, :1000])
    savefig(fig, "ref.png")

    irs_eq = gtf_obj.get_ir_equation()

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)
    ax[0].plot(irs[3]/np.max(irs[3]), label='Todd')
    ax[0].plot(irs_eq[3]/np.max(irs_eq[3]), label='Equation')
    ax[0].legend()
    ax[0].set_xlim([0, 200])

    ax[1].plot(irs_ref[3]/np.max(irs_ref[3]), label='Detly')
    ax[1].plot(irs_eq[3]/np.max(irs_eq[3]), label='Equation')
    ax[1].legend()
    ax[1].set_xlim([0, 200])
    savefig(fig, 'images/validate/compare.png')


if __name__ == "__main__":
    # compare_ir()
    compare_cfs()
