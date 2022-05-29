import os
import numpy as np
import scipy.signal as dsp_tools
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import audioread


def read(wav_path, tar_fs=None):
    """ read wav file, implete with soundfile
    args:
        wav_path
        tar_fs
    returns
        waveform, fs
    """
    wav_path = os.path.expanduser(wav_path)
    try:
        wav, fs = sf.read(wav_path)
        if tar_fs is not None and tar_fs != fs:
            wav = resample(wav, fs, tar_fs)
            fs = tar_fs
    except Exception:
        with audioread.audio_open(wav_path) as f:
            n_channel = f.channels
            pcm_16bit_segs = [item for item in f]
            wav_segs = [np.frombuffer(item, dtype=np.int16)
                        for item in pcm_16bit_segs]
            if n_channel > 1:
                wav_segs = [np.reshape(item, [-1, n_channel])
                            for item in wav_segs]
            wav = np.concatenate(wav_segs, axis=0)
    return [wav, fs]


def write(x, fs, wav_path, n_bit=24):
    """ write wav file,  implete with soundfile
    args:
        x, fs, wav_path, n_bit
    """
    wav_path = os.path.expanduser(wav_path)
    subtype = f'PCM_{n_bit}'
    sf.write(file=wav_path, data=x, samplerate=fs, subtype=subtype)


def resample(x, src_fs, tar_fs, axis=0):
    """ resample signal, implete with librosa
    Args:
        x: signal, resampling in the first dimension
        src_fs: original sample frequency
        tar_fs: target sample frequency
    Returns:
        resampled data
    """
    if len(x.shape) > 2:
        raise Exception('only 1d and 2d array are supported')
    elif len(x.shape) == 2:
        x = x.T  # the fist dimension represent channels

    #
    x = np.asfortranarray(x)
    x_resampled = librosa.resample(x, src_fs, tar_fs)
    x_resampled = np.asarray(x_resampled)
    x_resampled = x_resampled.T
    return x_resampled


def brir_filter(x, brir):
    """ synthesize spatial recording
    Args:
        x: single-channel signal
        brir: binaural room impulse response
    Returns:
        spatialized signal
    """
    if (len(x.shape) > 1) and (x.shape[1] > 1):
        raise Exception('x has mutliple channels')
    signal_len = x.shape[0]
    y = np.zeros((signal_len, 2), dtype=np.float64)
    for channel_i in range(2):
        y[:, channel_i] = np.squeeze(
                            dsp_tools.lfilter(brir[:, channel_i], 1,
                                              x, axis=0))
    return y


def cal_power(x):
    """calculate the engergy of given signal
    """
    x_amp = np.abs(x)
    amp_theta = np.max(x_amp)/1e5
    x_len = np.count_nonzero(x_amp > amp_theta)
    power = np.sum(np.square(x))/x_len
    return power


def frame_data(x, frame_len, frame_shift):
    """parse data into frames
    Args:
        x: single/multiple channel data
        frame_len: frame length in sample
        frame_shift: frame_shift in sample
    Returns:
        [n_frame,frame_len,n_chann]
    """
    x = np.asarray(x)

    if frame_len < 1:
        return x
    if frame_len == 1:
        return np.expand_dims(x, axis=1)

    # ensure x is 2d array
    n_dim = len(x.shape)
    if n_dim == 1:
        x = x[:, np.newaxis]

    n_sample, *sample_shape = x.shape
    n_frame = np.int(np.floor(np.float32(n_sample-frame_len)/frame_shift)+1)
    frame_all = np.zeros((n_frame, frame_len, *sample_shape))
    for frame_i in range(n_frame):
        frame_slice = slice(frame_i*frame_shift, frame_i*frame_shift+frame_len)
        frame_all[frame_i] = x[frame_slice]

    if n_dim == 1:
        frame_all = np.squeeze(frame_all)
    return frame_all


def set_snr(x, ref, snr):
    """ scale signal to a certain snr relative to ref
    Args:
        x: signal to be scaled
        ref: reference signal
        snr:
    Returns:
        scaled target signal
    """
    power_x = cal_power(x)
    power_ref = cal_power(ref)
    coef = np.sqrt(np.float_power(10, float(snr)/10)
                   / (power_x / power_ref))
    return coef*x.copy()


def _cal_snr(tar, inter):
    """sub-function of cal_snr"""
    power_tar = cal_power(tar)
    power_inter = cal_power(inter)
    snr = 10*np.log10(power_tar/power_inter+1e-20)
    return snr


def cal_snr(tar, inter, frame_len=None, frame_shift=None, is_plot=None):
    """Calculate snr of entire signal, frames if frame_len is
    specified.
                snr = 10log10(power_tar/power_inter)
    Args:
        tar: target signal, single channel
        inter: interfere signal, single channel
        frame_len:
        frame_shift: if not specified, set to frame_len/2
        if_plot: whether to plot snr of each frames, default to None
    Returns:
        float number or numpy.ndarray
    """

    # single channel
    # if len(tar.shape) > 1 or len(inter.shape) > 1:
    #     raise Exception('input should be single channel')

    if frame_len is None:
        snr = _cal_snr(tar, inter)
    else:
        if frame_shift is None:
            frame_shift = np.int16(frame_len/2)

        # signal length check
        if tar.shape[0] != inter.shape[0]:
            raise Exception('tar and inter do not have the same length,\
                             tar:{}, inter:{}'.format(tar.shape[0],
                                                      inter.shape[0]))

        tar_frames = frame_data(tar, frame_len, frame_shift)
        inter_frames = frame_data(inter, frame_len, frame_shift)
        n_frame = tar_frames.shape[0]
        snr = np.asarray([_cal_snr(tar_frames[i],
                                   inter_frames[i])
                          for i in range(n_frame)])
        if is_plot:
            n_sample = tar.shape[0]
            # waveform of tar and inter
            fig = plt.figure()
            ax1 = fig.subplots(1, 1)
            time_axis = np.arange(n_sample)
            ax1.plot(time_axis, tar[:n_sample], label='tar')
            ax1.plot(time_axis, inter[:n_sample], label='inter')
            ax1.set_xlabel('time(s)')
            ax1.set_ylabel('amp')
            ax1.legend(loc='upper left')

            # snrs of frames
            ax2 = ax1.twinx()
            ax2.set_ylabel('snr(dB)')
            # time: center of frame
            frame_t_all = np.arange(n_frame)*frame_shift+np.int16(frame_len/2)
            ax2.plot(frame_t_all, snr, color='red', linewidth=2,
                     label='snr')
            ax2.legend(loc='upper right')
    return snr


def cal_delay(x1, x2, method='gcc', max_delay=None):
    """
    """
    from .GCC import cal_ccf, cal_gcc_phat

    x1_len = x1.shape[0]
    x2_len = x2.shape[0]

    if method == 'gcc':
        gcc = cal_ccf(x1, x2, max_delay=max_delay)
        if max_delay is not None:
            delay = np.argmax(gcc)-max_delay
        else:
            delay = np.argmax(gcc)-x1_len
    elif method == 'gcc_phat':
        gcc = cal_gcc_phat(x1, x2, max_delay=max_delay)
        if max_delay is not None:
            delay = np.argmax(gcc)-max_delay
        else:
            delay = np.argmax(gcc)-x1_len
    elif method == 'phase':
        x_len = np.max([x1_len, x2_len])
        half_x_len = np.int(np.floor(x_len/2))
        x1_phase = \
            np.unwrap(
                np.angle(
                    np.fft.fft(x1, x_len)))[1:half_x_len]
        x2_phase = \
            np.unwrap(
                np.angle(
                    np.fft.fft(x2, x_len)))[1:half_x_len]
        phase_diff = np.unwrap(x1_phase-x2_phase)
        norm_freqs = np.arange(1, half_x_len)/x_len
        coef = np.polyfit(norm_freqs, phase_diff, 1)
        delay = coef[0]/(2*np.pi)

        # fig, ax = plt.subplots(1, 1)
        # ax.plot(norm_freqs, phase_diff)
        # fig.savefig('tmp.png')
    else:
        raise Exception('not finish')
    return delay


def iterable(x):
    try:
        [tmp for tmp in x]
        return True
    except Exception:
        return False


def gen_wn(shape, ref=None, energy_ratio=0, power=1):
    """Generate Gaussian white noise with either given energy ration related
    to ref signal or given power
    Args:
        shape: the shape of white noise to be generated,
        ref: reference signal
        energy_ratio: energy ration(dB) between white noise and reference
            signal, default to 0 dB
        power:
    Returns:
        white noise
    """
    if not iterable(shape):
        shape = [shape]

    wn = np.random.normal(0, 1, size=shape)
    if ref is not None:
        wn = set_snr(wn, ref, energy_ratio)
    else:
        power_orin = np.sum(wn**2, axis=0)/shape[0]
        coef = np.sqrt(power/power_orin)
        wn = wn*coef
    return wn


def gen_diffuse_wn(brirs_path, record_path, snr, diffuse_wn_path, gpu_id=0):

    from .GPU_Filter import GPU_Filter

    if os.path.exists(diffuse_wn_path):
        return

    gpu_filter = GPU_Filter(gpu_id)
    brirs = np.load(brirs_path)
    n_azi = brirs.shape[0]

    record, fs = read(record_path)
    record_len, n_channel = record.shape
    diffuse_wn = np.zeros([record_len, n_channel], dtype=np.float32)

    for azi_i in range(n_azi):
        wn = gen_wn([record_len])
        wn_record = gpu_filter.brir_filter(wn, brirs[azi_i])
        diffuse_wn = diffuse_wn + wn_record
    diffuse_wn = set_snr(diffuse_wn, ref=record, snr=snr)
    write(diffuse_wn, fs, diffuse_wn_path)


def VAD(x, frame_len, frame_shift=None, theta=40, is_plot=False):
    """ Energy based vad.
        1. Frame data with frame_shift of 0
        2. Calculte the energy of each frame
        3. Frames with energy below max_energy-theta is regarded as
            silent frames
    Args:
        x: input signal. when multiple-channel signal is given, signal from all
            channels are summed after being squared
        frame_len: frame length
        frame_shift: frames shift length in time
        theta: the maximal energy difference between frames, default 40dB
        is_plot: whether to ploting vad result, default False
    Returns:
        vad_flags: True for voice, False for silence
    """
    if frame_shift is None:
        frame_shift = frame_len
    frames = frame_data(x, frame_len, frame_shift)
    n_frame = frames.shape[0]
    energy = np.asarray([np.sum(frames[i]**2) for i in range(n_frame)])
    energy_thd = np.max(energy)/(10**(theta/10.0))
    vad_flags = energy > energy_thd
    return vad_flags


def truncate_silence(x, frame_len, trunc_type="both", theta=40):
    """truncate silence part along the first dimension
    Args:
        x: data to be truncated
        trunc_type: specify which parts to be cliped,
            choices=[begin,end, both], default to both
        eps: amplitude threshold
    Returns:
        data truncated
    """
    vad_flags = VAD(x, frame_len, theta=theta)
    vad_frame_inices = np.nonzero(vad_flags)[0]
    start_pos, end_pos = 0, 0
    if trunc_type in ['begin', 'both']:
        start_pos = np.int(vad_frame_inices[0]*frame_len)
    if trunc_type in ['end', 'both']:
        end_pos = np.int(vad_frame_inices[-1]*frame_len) + frame_len
    return x[start_pos:end_pos]


def hz2erbscal(freq):
    """convert Hz to ERB scale"""
    return 21.4*np.log10(4.37*freq/1e3+1)


def erbscal2hz(erb_num):
    """convert ERB scale to Hz"""
    return (10**(erb_num/21.4)-1)/4.37*1e3


def cal_erb(cf):
    """calculate the ERB(Hz) of given center frequency based on equation
    given by Glasberg and Moore
    Args
        cf: center frequency Hz, single value or numpy array
    """
    return 24.7*(4.37*cf/1000+1.0)


def test():
    print('test')
    wav_path = '../examples/data/binaural_1.wav'
    wav, fs = read(wav_path)
    x = np.zeros([600, 2])
    x[:, 0] = np.pad(wav[100:600, 0], [50, 50])
    x[:, 1] = np.pad(wav[100:600, 0], [20, 80])

    # vad
    # vad_flag_all,fig = vad(data,fs,frame_dur=20e-3,is_plot=True,theta=20)
    # savefig.savefig(fig,fig_name='vad',fig_dir='./images/wav_tools')

    # wave
    # fig = plot_wav_spec(data)
    # plot_tools.savefig(fig,name='wav_spec',dir='./images/wav_tools')

    delay_gcc = cal_delay(x[:, 0], x[:, 1], method='gcc')
    delay_phase = cal_delay(x[:, 0], x[:, 1], method='phase')
    print(delay_gcc, delay_phase)


if __name__ == '__main__':
    test()
