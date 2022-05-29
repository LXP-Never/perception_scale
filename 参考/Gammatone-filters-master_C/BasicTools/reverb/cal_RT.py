import numpy as np
import matplotlib.pyplot as plt
from .. import plot_tools


def cal_rt(rir, fs, is_plot=False):
    """estimate reverberation time based on room impulse response time
        Args:
            rir: room impulse response
            fs: sample frequency

        Returns:
            estimated rt60 in second
    """
    rir = np.reshape(rir, newshape=[-1, 1])
    rir_len = rir.shape[0]

    # specify which part of ir is used for rt calculation
    start_pos = 0
    end_pos = rir.shape[0]

    # inverse-integration decay curve
    iidc = np.flipud(np.cumsum(np.flipud(rir**2)))
    iidc_norm = 10*np.log10(iidc/np.max(iidc))

    # linear regression
    db5_pos = np.nonzero(iidc_norm[start_pos:end_pos] <= -5)[0][0]+start_pos
    db35_pos = np.nonzero(iidc_norm[start_pos:end_pos] < -35)[0][0]+start_pos
    coeffs = np.polyfit(np.arange(db5_pos, db35_pos, dtype=np.float32)/fs,
                        iidc_norm[db5_pos:db35_pos],
                        deg=1)
    # finally, rt is derived based on the slope of regressed line
    slope = coeffs[0]
    rt = -60.0/slope

    if is_plot:
        lineswidth = 3
        fig = plt.figure()
        axes = fig.subplots()
        t = np.arange(0, rir_len)/fs
        # inverser-integration delay curve
        axes.plot(t, iidc_norm,
                  linewidth=lineswidth,
                  label='inverse integration')
        # regressed line
        axes.plot(t, t*slope+coeffs[1],
                  linewidth=lineswidth,
                  label='line regression')
        # start and end point of ir used for rt calculation(dB5, dB35)
        axes.plot([db5_pos/fs, db35_pos/fs], [-5, -35], 'kx', markersize=9)
        axes.text(db5_pos/fs, -5, '-5dB')
        axes.text(db35_pos/fs, -35, '-35dB')

        axes.legend()
        axes.set_xlabel('time(s)')
        axes.set_ylabel('amplitude(dB)')
        axes.set_title('$RT_{30}=%.2f$)'.fromat(rt))
        return [rt, fig]
    else:
        return rt


def test():
    rir_fpath = 'resource/rir.npy'
    rir, fs = np.load(rir_fpath)
    rt, fig = cal_rt(rir, fs, is_plot=True)
    plot_tools.savefig(fig, fig_name='rt', fig_dir='./images/reverb_time')


if __name__ == '__main__':
    test()
