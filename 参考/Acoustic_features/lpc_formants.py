# -*- coding: utf-8 -*-
"""
LPC 共振峰
"""
import numpy as np
from scipy import signal


def get_formants(sig, fs, meth='AC', nform=3, lpcEnv=False, pre_emph=True, alpha=0.97):
    """
    Compute formants by solving for the roots of A(z)=0, where
    A(z) is the prediction polynomial.
    
    The LPC (A(z)) can be compute using autocorrelation or Burg method
    
    It is recommended to apply windowing (Hann or Hamm) to sig before using this
    function
    

    Parameters
    ----------
    sig : Array
        Speech signal (or short time frame).
    fs : int
        Sampling frequency of the speech signal.
    meth : String, optional
        Select the method used to compute the LPC. The default is 'Burg'.
    nform : Int, optional
        Number of expected formants. The default is 3.
    lpcEnv : Bool, optional
        If True, then the LPC envelope is returned along with the formants. The default is False.
    pre_emph: Bool
        If True, pre-emphasis is applied to the signal (recommended). Default is True
    alpha: Float
        Pre-empahsis coefficient. Default is 0.97

    Returns
    -------
    Formant frequencies
    LPC_env if lpcEnv=True

    """
    # Number of coefficients for LP model
    M = int(2 + fs / 1000)

    if pre_emph == True:
        # Pre-emphasis (Highpass filter at ~50Hz).
        # This process is applied twice in order to
        # foucsed on the high frequency components
        sig = premph(sig, alpha)

    # Select LPC computation method
    if meth == 'AC':
        lpc_coeff = lcp_AC(sig, M)
    elif meth == 'Burg':
        lpc_coeff = lpc_burg(sig, M)

    # Solve for the roots of A(z)=0
    rts = np.roots(lpc_coeff)
    rts = [r for r in rts if np.imag(r) >= 0]
    # Get angles.
    angz = np.arctan2(np.imag(rts.copy()), np.real(rts.copy()))
    frqs = angz * (fs / (2 * np.pi))  # Convert to Hertz

    # Sort
    frqs = sorted(frqs)
    index = np.argsort(frqs)

    # frqs = np.asarray(frqs)
    # Get bandwidths
    rts = np.asarray(rts)
    bw = -0.5 * (fs / (2 * np.pi)) * np.log(np.abs(rts[index]))

    # Get formant frequencies.
    formants = []
    for i in range(len(frqs)):
        if (frqs[i] > 90) & (bw[i] < 400):
            formants.append(frqs[i])
    # Select formants
    formants = np.asarray(formants[0:nform])
    # Get lcp envelope
    if lpcEnv:
        G = 1  # np.sum(sig)**2/len(sig)
        w, h = signal.freqz(G, lpc_coeff)
        # Compute magnitude and convert to dB
        lpc_env = 20 * np.log10(np.abs(h.T))
        return formants, lpc_env
    else:
        return formants


def lcp_AC(sig, M):
    """
    Compute LPC coeficcients using the autocorrelation method

    Parameters
    ----------
    sig : Speech signal (short-time frame or all)
    M : Number of coefficients

    Returns
    -------
    lpc_coeff : LPC coefficients of the prediction polynomial A(z)

    """
    # M-th order correlation
    rx = np.zeros([M + 1, 1])
    nsamples = len(sig)
    for i in range(0, M + 1):
        var1 = sig[0:(nsamples - i)]
        var1 = var1.reshape(1, -1)
        var2 = sig[i:nsamples]
        var2 = var2.reshape(-1, 1)
        var3 = np.dot(var1, var2)[0][0]
        rx[i, 0] = rx[i, 0] + var3
    # #----
    covmat = np.zeros([M, M])
    for i in range(M):
        covmat[i, i:M] = np.squeeze(rx[0:M - i])
        covmat[i:M, i] = np.squeeze(rx[0:M - i])

    # Compute the coefficients of the prediction polynomial A(z)
    lpc_coeff = np.linalg.solve(-covmat, rx[1:M + 1])
    lpc_coeff = np.squeeze(lpc_coeff)
    lpc_coeff = np.hstack([1, lpc_coeff])
    return lpc_coeff


def lpc_burg(y, order):
    """
    Author: https://librosa.org/doc/main/_modules/librosa/core/audio.html#lpc
    """
    # This implementation follows the description of Burg's algorithm given in
    # section III of Marple's paper referenced in the docstring.
    #
    # We use the Levinson-Durbin recursion to compute AR coefficients for each
    # increasing model order by using those from the last. We maintain two
    # arrays and then flip them each time we increase the model order so that
    # we may use all the coefficients from the previous order while we compute
    # those for the new one. These two arrays hold ar_coeffs for order M and
    # order M-1.  (Corresponding to a_{M,k} and a_{M-1,k} in eqn 5)

    dtype = y.dtype.type
    ar_coeffs = np.zeros(order + 1, dtype=dtype)
    ar_coeffs[0] = dtype(1)
    ar_coeffs_prev = np.zeros(order + 1, dtype=dtype)
    ar_coeffs_prev[0] = dtype(1)

    # These two arrays hold the forward and backward prediction error. They
    # correspond to f_{M-1,k} and b_{M-1,k} in eqns 10, 11, 13 and 14 of
    # Marple. First they are used to compute the reflection coefficient at
    # order M from M-1 then are re-used as f_{M,k} and b_{M,k} for each
    # iteration of the below loop
    fwd_pred_error = y[1:]
    bwd_pred_error = y[:-1]

    # DEN_{M} from eqn 16 of Marple.
    den = np.dot(fwd_pred_error, fwd_pred_error) + np.dot(
        bwd_pred_error, bwd_pred_error
    )

    for i in range(order):
        if den <= 0:
            raise FloatingPointError("numerical error, input ill-conditioned?")

        # Eqn 15 of Marple, with fwd_pred_error and bwd_pred_error
        # corresponding to f_{M-1,k+1} and b{M-1,k} and the result as a_{M,M}
        # reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)
        reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)

        # Now we use the reflection coefficient and the AR coefficients from
        # the last model order to compute all of the AR coefficients for the
        # current one.  This is the Levinson-Durbin recursion described in
        # eqn 5.
        # Note 1: We don't have to care about complex conjugates as our signals
        # are all real-valued
        # Note 2: j counts 1..order+1, i-j+1 counts order..0
        # Note 3: The first element of ar_coeffs* is always 1, which copies in
        # the reflection coefficient at the end of the new AR coefficient array
        # after the preceding coefficients
        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]

        # Update the forward and backward prediction errors corresponding to
        # eqns 13 and 14.  We start with f_{M-1,k+1} and b_{M-1,k} and use them
        # to compute f_{M,k} and b_{M,k}
        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

        # SNIP - we are now done with order M and advance. M-1 <- M

        # Compute DEN_{M} using the recursion from eqn 17.
        #
        # reflect_coeff = a_{M-1,M-1}      (we have advanced M)
        # den =  DEN_{M-1}                 (rhs)
        # bwd_pred_error = b_{M-1,N-M+1}   (we have advanced M)
        # fwd_pred_error = f_{M-1,k}       (we have advanced M)
        # den <- DEN_{M}                   (lhs)
        #

        q = dtype(1) - reflect_coeff ** 2
        den = q * den - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2

        # Shift up forward error.
        #
        # fwd_pred_error <- f_{M-1,k+1}
        # bwd_pred_error <- b_{M-1,k}
        #
        # N.B. We do this after computing the denominator using eqn 17 but
        # before using it in the numerator in eqn 15.
        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]

    return ar_coeffs


def premph(sig, alpha=0.97):
    """
    为了抑制低频中的高强度谐波，建议使用预强调(Pre-emphasis)(50Hz高通滤波)，因为它会严重影响整个频谱的共振峰
    :param sig: Speech signal (short-time)
    :param alpha: 预加重系数
    :return: 滤波后的信号
    """

    # Pre-emphasis (highpass) filtering with first-order auto-regressive filter
    sig = np.append(sig[0], sig[1:] - alpha * sig[:-1])
    # For some reason, doing this process twice yields better results for LPC.
    # Is it something about the OS, or Python version?
    #    sig = np.append(sig[0], sig[1:] - alpha* sig[:-1])
    return sig


def formant_post(formants):
    """ 共振峰(formants)的后处理。主要用于计算共振峰后减少异常值 """

    for i in range(formants.shape[1]):
        ssig = smooth_curve(formants[:, i], 5)
        formants[0:len(ssig), i] = ssig
    return formants


def smooth_curve(y, N=5):
    """ 对一个信号应用移动平均

    Parameters
    ----------
    y : Array of values to be smoothed
    N : Size of the moving average filter
    Returns
    -------
    y_fit : Smoothed array

    """
    # Moving average
    cumsum = np.cumsum(np.insert(y, 0, 0))
    y_fit = (cumsum[N:] - cumsum[:-N]) / np.float(N)
    return y_fit
