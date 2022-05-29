import numpy as np
import matplotlib.pyplot as plt
from GTF import GTF


def main():
    fs = 1e3
    n_band = 4
    freq_low = 70
    freq_high = 7000

    fig, ax = plt.subplots(1, 1)

    gtf = GTF(fs, cf_low=freq_low, cf_high=freq_high,
                           n_band=n_band)
    cfs = gtf.cfs
    bws = gtf.cal_bw(cfs)
    ax.errorbar(np.arange(n_band)+n_band/100, cfs, yerr=bws/2,
                linewidth=2,
                label='freq_range_cf')

    gtf_obj = GTF(fs, freq_low=freq_low, freq_high=freq_high, n_band=n_band)
    cfs = gtf_obj.cfs
    bws = gtf_obj.cal_bw(cfs)
    ax.errorbar(np.arange(n_band), cfs, yerr=bws/2,
                linewidth=2, label='freq_rabge_freq')

    ax.set_xlabel('freq_band')
    ax.set_ylabel('freq(Hz)')
    ax.legend()
    fig.savefig(f'images/freq_range.png', dpi=100)


if __name__ == "__main__":
    main()
