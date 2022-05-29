import numpy as np
from . import wav_tools


def _cal_gcc_phat(x1, x2, max_delay, win_f, snr_thd):
    """subfunction of cal_gcc_phat
        gcc-phat, numpy ndarray with shape of [gcc_len]
    """
    n_sample = np.max((x1.shape[0], x2.shape[0]))
    if max_delay is None:
        max_delay = n_sample-1
    #
    window = win_f(n_sample)
    gcc_len = 2*n_sample-1
    x1_fft = np.fft.fft(np.multiply(x1, window), n=gcc_len)
    x2_fft = np.fft.fft(np.multiply(x2, window), n=gcc_len)
    gcc_fft = np.multiply(x1_fft, np.conj(x2_fft))
    # leave out frequency bins with small amplitude
    gcc_fft_amp = np.abs(gcc_fft)

    # clip small value to zeros
    eps = np.max(gcc_fft_amp)*(10**(snr_thd/10))
    gcc_fft[gcc_fft_amp < eps] = 0
    gcc_fft_amp[gcc_fft_amp < eps] = eps

    # phase transform
    gcc_phat_raw = np.real(np.fft.ifft(np.divide(gcc_fft, gcc_fft_amp),
                                       gcc_len))
    #
    gcc_phat = np.concatenate((gcc_phat_raw[-max_delay:],
                               gcc_phat_raw[:max_delay+1]))
    return gcc_phat


def cal_gcc_phat(x1, x2, win_f=np.hanning, max_delay=None,
                 frame_len=None, shift_len=None, snr_thd=-50):
    """Calculate general corss-correlation phase-transform
    Args:
        x1,x2: single-channel data
        win_f: window function, default to hanning
        max_delay: maximal delay in sample of ccf, if not specified, it will
                     be set to signale length. The relation between max_delay
                     and gcc_len: gcc_len=2*max_delay+1
        frame_len: frame length in sample, if not specified, frame_len is
                   set to be signal length
        shift_len: if not specified, set to frame_len/2
        snr_thd: allowed amp range,default to -50 dB
    Returns:
        gcc-phat with shape of [gcc_len] or [n_frame,gcc_len]
    """
    if frame_len is None:
        gcc_phat_result = _cal_gcc_phat(x1, x2, max_delay, win_f, snr_thd)
    else:
        if shift_len is None:
            shift_len = np.int16(frame_len/2)
        # signal length check
        if x1.shape[0] != x2.shape[0]:
            raise Exception('x1,x2 do not have the same length,\
                             x1:{}, x2:{}'.format(x1.shape[0], x2.shape[0]))
        frames_x1 = wav_tools.frame_data(x1, frame_len, shift_len)
        frames_x2 = wav_tools.frame_data(x2, frame_len, shift_len)
        n_frame = frames_x1.shape[0]
        gcc_phat_result = np.asarray([_cal_gcc_phat(frames_x1[frame_i],
                                                    frames_x2[frame_i],
                                                    max_delay, win_f,
                                                    snr_thd)
                                      for frame_i in range(n_frame)])
    return gcc_phat_result


def _cal_ccf(x1, x2, max_delay, win_f, is_fft):
    """calculate cross-crrelation function in frequency domain
    Args:
        x1,x2: single channel signals
        max_delay: delay range, ccf_len = 2*max_delay+1
    Returns:
        cross-correlation function with shape of [ccf_len]
    """
    n_sample = np.max((x1.shape[0], x2.shape[0]))
    if max_delay is None:
        max_delay = n_sample-1

    if is_fft:
        # add hanning window before fft
        window = win_f(n_sample)
        ccf_len = 2*n_sample-1
        x1_fft = np.fft.fft(np.multiply(x1, window), ccf_len)
        x2_fft = np.fft.fft(np.multiply(x2, window), ccf_len)
        ccf_unshift = np.real(
                        np.fft.ifft(
                            np.multiply(x1_fft, np.conjugate(x2_fft))))
        ccf = np.concatenate([ccf_unshift[-max_delay:],
                              ccf_unshift[:max_delay+1]],
                             axis=0)
    else:
        ccf = np.correlate(x1, x2, 'full')
    return ccf


def cal_ccf(x1, x2, max_delay=None, frame_len=None, shift_len=None,
            win_f=np.ones, is_fft=False):
    """Calculate cross-correlation function of whole signal or frames
    if frame_len is specified
    Args:
        x1: single-channel signal
        x2: single-channel signal
        max_delay: maximal delay in sample of ccf, if not specified, it will
                     be set to x_len-1. The relation between max_delay
                     and ccf_len: ccf_len=2*max_delay+1
        frame_len: frame length, if not specified, treat whole signal
                   as one frame
        shift_len: if not specified, set to frame_len/2
        win_f: window function, default to np.ones, rectangle windows
        is_fft: whether to use fft
    Returns:
        corss-correlation function, with shape of
        - [ccf_len]: ccf of whole signal
        - [n_frame,ccf_len]: ccf of frames
    """
    if frame_len is None:
        ccf = _cal_ccf(x1, x2, max_delay, win_f, is_fft)
    else:
        if shift_len is None:
            shift_len = np.int16(frame_len/2)
        # signal length check
        if x1.shape[0] != x2.shape[0]:
            raise Exception('x1,x2 do not have the same length,\
                             x1:{}, x2:{}'.format(x1.shape[0], x2.shape[0]))
        frames_x1 = wav_tools.frame_data(x1, frame_len, shift_len)
        frames_x2 = wav_tools.frame_data(x2, frame_len, shift_len)
        n_frame = frames_x1.shape[0]
        ccf = np.asarray([_cal_ccf(frames_x1[i], frames_x2[i],
                                   max_delay, win_f, is_fft)
                          for i in range(n_frame)])
    return ccf


def cal_delay(x1, x2, is_fft=False):
    x_len = np.max((x1.shape[0], x2.shape[0]))
    ccf = cal_ccf(x1, x2, is_fft)
    delay = np.argmax(ccf) + 1 - x_len
    return delay


def align(x1, x2):
    delay = cal_delay(x1, x2)
    if delay < 0:  # x1 is hehind
        x2 = np.concatenate((x2[-delay:], np.zeros(-delay)))
    elif delay > 0:  # x1 is ahead
        x2 = np.concatenate((np.zeros(-delay), x2[-delay:]))
