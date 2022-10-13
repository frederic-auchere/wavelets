import copy
import numpy as np
import warnings
from wavelets import AtrousTransform, B3spline, Coefficients, generalized_anscombe, convolution

__all__ = ['denoise', 'wow']


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


def denoise(data, scaling_function, weights, noise=None, bilateral=None, soft_threshold=True, anscombe=False):
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

    if type(data) is np.ndarray:  # input is an image
        if data.dtype is np.int32 or data.dtype is np.int64 or data.dtype == '>f4':
            data = np.float64(data)
        if n_scales is None:
            n_scales = int(np.log2(min(data.shape)) - np.log2(len(scaling_function.coefficients_1d)))
            print(n_scales)
        n_dims = data.ndim
    elif type(data) is Coefficients:  # input is already computed coefficients
        n_scales = len(data)-1
        n_dims = data[0].ndim
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
            if whitening:
                local_power = np.std(c)
            else:
                local_power = 1
        else:
            if whitening:
                atrous_kernel = coefficients.scaling_function.atrous_kernel(s)
                convolution(power, atrous_kernel, local_power)
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
            gamma_min = data.min()
        if gamma_max is None:
            gamma_max = data.max()
        gamma_scaled -= gamma_min
        gamma_scaled /= gamma_max - gamma_min
        gamma_scaled **= gamma
        recon = (1 - h)*recon + h*gamma_scaled

    return recon, coefficients
