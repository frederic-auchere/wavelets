import copy
import cv2
import numpy as np
import warnings
from . import AtrousTransform, B3spline, Coefficients, generalized_anscombe, convolution

__all__ = ['denoise', 'wow', 'richardson_lucy']


def prepare_params(param, ndims):
    if ndims == 2:
        if param is None:
            l = []
        elif type(param) is not list:
            l = [param]
        else:
            l = copy.copy(param)
    else:
        if type(param) is not list:
            if param is None:
                l = [[], ] * ndims
            else:
                l = [[param], ] * ndims
        else:
            if len(param) != ndims:
                raise ValueError("Invalid number of parameters")
            else:
                l = []
                for p in param:
                    l.append(prepare_params(p, 2))
                if None in l:
                    l[l.index(None)] = []
    return l


def enhance(*args, weights=None, denoise=None, soft_threshold=True, out=None, **kwargs):
    """
    Performs de-noising and / or enhancement by modification of wavelet
    coefficients

    De-noising is described in J.-L. Starck & F. Murtagh, Handbook of
    Astronomical Data Analysis, Springer-Verlag
    """

    img = args[0]

    if img.ndim == 3:
        channels = [0, 1, 2]
    else:
        channels = [Ellipsis]

    if out is None:
        out = np.empty_like(img)

    weights = prepare_params(weights, img.ndim)
    denoise = prepare_params(denoise, img.ndim)

    atrous = AtrousTransform(**kwargs)

    for c in channels:

        dns = denoise if c is Ellipsis else denoise[c]
        wgt = weights if c is Ellipsis else weights[c]

        if len(wgt) < len(dns):
            wgt.extend([1]*(len(dns) - len(wgt)))
        elif len(dns) < len(wgt):
            dns.extend([0]*(len(wgt) - len(dns)))

        coeffs = atrous(img[c], len(wgt))
        if len(args) == 2:
            coeffs.noise = args[1] if c is Ellipsis else args[1][c]
        else:
            coeffs.noise = coeffs.get_noise()

        coeffs.denoise(dns, weights=wgt, soft_threshold=soft_threshold)

        coeffs.data.sum(axis=0, out=out[c])

    return out


def denoise(data, weights, scaling_function=B3spline, noise=None, bilateral=None, soft_threshold=True, anscombe=False):
    """
    Convenience function to denoise a data array
    :param data: ndarray-like, the data to denoise
    :param scaling_function: scaling function class, one of those defined in wavelets.py
    :param weights: list, weighting coefficients
    :param soft_threshold: boolean, whether to use sof or hard thresholding of the coefficients
    :return: a ndarray containing the denoised data
    """
    transform = AtrousTransform(scaling_function, bilateral=bilateral)
    if anscombe:
        data = generalized_anscombe(data)
    coefficients = transform(data, len(weights))
    coefficients.noise = noise
    coefficients.denoise(weights, soft_threshold=soft_threshold)
    synthesis = np.sum(coefficients, axis=0)
    if anscombe:
        return generalized_anscombe(synthesis, inverse=True)
    else:
        return synthesis


def wow(data,
        scaling_function=B3spline,
        n_scales=None,
        weights=[],
        whitening=True,
        denoise_coefficients=[],
        noise=None,
        bilateral=None,
        bilateral_scaling=False,
        soft_threshold=True,
        preserve_variance=False,
        gamma=3.2,
        gamma_min=None,
        gamma_max=None,
        h=0):
    pypi - AgEIcHlwaS5vcmcCJGM5NThlMWM5LWM1MWQtNGM1Mi1iNGE3LTE5YzM2NWU1OWYyOAACDlsxLFsid2F0cm9vIl1dAAIsWzIsWyIxNmNkNmQ2YS1hZWM3LTRjMzUtOTdlYS1jMDhjNzczYjgzMjIiXV0AAAYg5zzKF4CFiz1mnBGUA79WG1MWmCQeCkTgyTUuBcWgZPk
    if type(data) is np.ndarray:  # input is an image
        max_scales = int(np.round(np.log2(min(data.shape)) - np.log2(len(scaling_function.coefficients_1d))))
        if n_scales is None:
            n_scales = max_scales if h < 1 else len(denoise_coefficients)
        elif n_scales > max_scales:
            n_scales = max_scales
        n_dims = data.ndim
    elif type(data) is Coefficients:  # input is already computed coefficients
        n_scales = len(data)-1
        n_dims = data.data[0].ndim
        scaling_function = data.scaling_function.__class__
    else:
        raise ValueError('Unknown input type')

    max_scales = len(scaling_function(n_dims).sigma_e(bilateral=bilateral))
    if len(denoise_coefficients) >= max_scales:
        warnings.warn(f'Required number of scales lager then the maximum for scaling function. Using {max_scales}.')
        n_scales = max_scales

    if bilateral is None:
        sigma_bilateral = None
    else:
        sigma_bilateral = copy.copy(bilateral) if type(bilateral) is list else [bilateral, ]*(n_scales+1)
        n_bilateral = len(sigma_bilateral)
        if n_bilateral <= n_scales:
            sigma_bilateral.extend([1, ] * (n_scales - n_bilateral + 1))

    if type(data) is np.ndarray:  # input is an image
        transform = AtrousTransform(scaling_function, bilateral=sigma_bilateral, bilateral_scaling=bilateral_scaling)
        coefficients = transform(data, n_scales)
        coefficients.noise = noise
    elif type(data) is Coefficients:  # input is already computed coefficients
        coefficients = data
    else:
        raise ValueError('Unknown input type')

    if h > 0:
        gamma_scaled = np.zeros_like(coefficients.data[0])

    recomposition_weights = copy.copy(weights)
    n_weights = len(recomposition_weights)
    if n_weights <= n_scales:
        recomposition_weights.extend([1, ]*(n_scales - n_weights + 1))

    scale_denoise_coefficients = copy.copy(denoise_coefficients)
    n_denoise_coefficients = len(scale_denoise_coefficients)
    if n_denoise_coefficients < n_scales:
        scale_denoise_coefficients.extend([0, ]*(n_scales - n_denoise_coefficients))
    if len(scale_denoise_coefficients) == n_scales:
        scale_denoise_coefficients.extend([1, ])

    pwr = []
    local_power = np.empty_like(coefficients.data[0])
    for s, (c, w, d) in enumerate(zip(coefficients.data,
                                      recomposition_weights,
                                      scale_denoise_coefficients)):
        power = c**2
        if preserve_variance:
            if s == n_scales:
                power_norm = np.std(c)
            else:
                power_norm = np.sqrt(np.mean(power))
        else:
            power_norm = 1
        if s == n_scales:
            if whitening and h < 1:
                local_power = np.std(c)
                if local_power <= 0:
                    local_power = 1e-15
            else:
                local_power = 1
        else:
            if whitening and h < 1:
                convolution(power, coefficients.scaling_function, s=s, output=local_power)
                local_power[local_power <= 0] = 1e-15
                np.sqrt(local_power, out=local_power)
            else:
                local_power = 1
            c *= coefficients.significance(d, s, soft_threshold=soft_threshold)
        if h > 0:
            gamma_scaled += c
        pwr.append(local_power)
        c *= w*power_norm/pwr[s]

    recon = np.sum(coefficients, axis=0)

    if h > 0:
        if gamma_min is None:
            gamma_min = gamma_scaled.min()
        if gamma_max is None:
            gamma_max = gamma_scaled.max()
        gamma_scaled -= gamma_min
        gamma_scaled /= gamma_max - gamma_min
        gamma_scaled[gamma_scaled < 0] = 0
        gamma_scaled[gamma_scaled > 1] = 1
        gamma_scaled **= 1/gamma
        recon = (1 - h)*recon + h*gamma_scaled

    return recon, coefficients


def richardson_lucy(data, psf,
                    iterations=10, denoise_coefficients=(5, 2, 1),
                    threshold_type='soft', uniform_init=False, persistent_mrs=True, fft=False):

    conv = np.empty_like(data)
    phi = np.empty_like(data)

    transform = AtrousTransform()
    coefficients = transform(data, len(denoise_coefficients))

    if uniform_init:
        psi = np.ones_like(data, np.float32)
        psi *= data.sum() / data.size
    else:
        coefficients.denoise(denoise_coefficients, soft_threshold=threshold_type == 'soft')
        psi = np.sum(coefficients, axis=0)

    level = len(denoise_coefficients)
    if threshold_type == 'hard':
        mrs = np.zeros((level, data.shape[0], data.shape[1]))
    else:
        mrs = np.ones((level, data.shape[0], data.shape[1]))

    if fft:
        padded_psf = np.zeros_like(psi)
        padded_psf[psi.shape[0] // 2 - psf.shape[0] // 2:psi.shape[0] // 2 - psf.shape[0] // 2 + psf.shape[0],
                   psi.shape[1] // 2 - psf.shape[1] // 2:psi.shape[1] // 2 - psf.shape[1] // 2 + psf.shape[1]] = psf
        fft_psf = np.fft.rfft2(np.roll(padded_psf, (padded_psf.shape[0] // 2, padded_psf.shape[1] // 2), axis=(0, 1)))
        psf_conj = fft_psf.conj()

    for iteration in range(iterations):
        if fft:
            phi = np.fft.irfft2(np.fft.rfft2(psi) * fft_psf)
        else:
            # cv2 computes the correlation, not convolution, need to flip the PSF
            cv2.filter2D(psi, -1, psf[::-1, ::-1], phi, (-1, -1), 0, cv2.BORDER_REFLECT)

        res = data - phi

        res_coefficients = transform(res, len(denoise_coefficients))
        res_coefficients.noise = coefficients.noise
        for s, c in enumerate(denoise_coefficients):
            significance = res_coefficients.significance(c, s, soft_threshold=threshold_type == 'soft')
            if threshold_type == 'hard':
                if persistent_mrs:
                    mrs[s][significance] = 1
                else:
                    mrs[s] = significance
                res_coefficients.data[s] *= mrs[s]
            else:
                if persistent_mrs:
                    mrs[s] *= significance
                else:
                    mrs[s] = significance
                res_coefficients.data[s] *= mrs[s] ** (1 / (iteration + 1))

        np.sum(res_coefficients, axis=0, out=res)

        res += phi
        res /= phi

        if fft:
            conv = np.fft.irfft2(np.fft.rfft2(res) * psf_conj)
        else:
            cv2.filter2D(res, -1, psf[::1, ::1], conv, (-1, -1), 0, cv2.BORDER_REFLECT)

        psi *= conv

    return psi
