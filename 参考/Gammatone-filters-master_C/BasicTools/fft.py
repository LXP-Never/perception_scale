import numpy as np
from . import wav_tools


def cal_stft(x, win_f=np.hanning, frame_len=1024, frame_shift=None):
    """short-time fast Fourier transform
    Args:
        x: 2d array, [signal_len, n_chann]
        win_f: window function, default to hannning
        frame_len: default to 1024
        frame_shift: default to frame_len/2
    Returns:
        stft: stft result in the shape of [n_frame, n_bin, n_chann]

    """
    # check params
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    elif len(x.shape) > 2:
        raise Exception(f'x has shape: {x.shape}')

    if frame_shift is None:
        frame_shift = np.int(frame_len/2)

    half_frame_len = np.int(np.floor(frame_len/2.0)+1)

    window = win_f(frame_len)
    # frames: [n_frame, frame_len , n_channel]
    frames = \
        (wav_tools.frame_data(x, frame_len, frame_shift)
         * window[np.newaxis, :, np.newaxis])

    fft_frames = np.fft.fft(frames, axis=1)
    stft = fft_frames[:, :half_frame_len]  # only valid frequency bins
    return stft


def cal_istft(stft, win_f=np.hanning, frame_len=1024, frame_shift=None,
              norm_win=False, return_norm_coef=False):
    """inversr short-time fast Fourier transform
    Args:
        stft: 3d ndarray, [n_frame, n_freq_bin, n_chann]
        win_f: window function
        frame_len: default to 1024
        frame_shift: default to frame_len/2
    Returns:
        x: wavform constructed from stft
        norm_ceofs: coef to normalize the effect of window function,
            only returned if specified
    """
    # check input params
    if frame_shift is None:
        frame_shift = np.int(frame_len/2)

    n_frame, _, n_chann = stft.shape

    #
    if np.mod(frame_len, 2) == 0:  # even
        stft_conj_part = np.flip(np.conj(stft[:, 1:-1, :]), axis=1)
    else:
        stft_conj_part = np.flip(np.conj(stft[:, 1:, :]), axis=1)
    stft_full = np.concatenate([stft, stft_conj_part], axis=1)

    window = win_f(frame_len)
    frames = \
        (np.real(np.fft.ifft(stft_full, axis=1))
         * window[np.newaxis, :, np.newaxis])

    x_len = frame_len+frame_shift*(n_frame-1)
    x = np.zeros((x_len, n_chann))
    norm_ceofs = np.zeros(x_len)
    window_square = window**2
    for frame_i in range(n_frame):
        frame_start = frame_i*frame_shift
        frame_slice = slice(frame_start, frame_start+frame_len)
        x[frame_slice] = x[frame_slice]+frames[frame_i]
        norm_ceofs[frame_slice] = norm_ceofs[frame_slice]+window_square

    if norm_win:  # normalized the effect of window function
        norm_ceofs[0], norm_ceofs[-1] = 1, 1
        x = x/norm_ceofs[:, np.newaxis]

    if return_norm_coef:
        return x, norm_ceofs
    else:
        return x
