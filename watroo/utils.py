import copy
import numpy as np
import cv2
from . import AtrousTransform, B3spline, Coefficients

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


def denoise(data, scaling_function, weights, soft_threshold=True):
    """
    Convenience function to denoise a data array
    :param data: ndarray-like, the data to denoise
    :param scaling_function: scaling function class, one of those defined in wavelets.py
    :param weights: list, weighting coefficients
    :param soft_threshold: boolean, whether to use sof or hard thresholding of the coefficients
    :return: a ndarray containing the denoised data
    """
    transform = AtrousTransform(scaling_function)
    coefficients = transform(data, len(weights))
    coefficients.denoise(weights, soft_threshold=soft_threshold)
    return np.sum(coefficients, axis=0)


def sdev_loc(image, kernel):
    mean2 = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mean2 **= 2
    std2 = cv2.filter2D(image**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    std2 -= mean2
    std2[std2 <= 0] = 1e-20
    return np.sqrt(std2)


def wow(data,
        scaling_function=B3spline,
        n_scales=None,
        weights=[],
        whitening=True,
        denoise_coefficients=[],
        bilateral=None,
        soft_threshold=True,
        preserve_variance=False):

    if type(data) is np.ndarray:  # input is an image
        if n_scales is None:
            n_scales = int(np.log2(min(data.shape)) - 2)

        transform = AtrousTransform(scaling_function)
        coefficients = transform(data, n_scales, recursive=True, bilateral=bilateral)
    elif type(data) is Coefficients:  # input is already computed coefficients
        coefficients = data
        n_scales = len(coefficients)-1
    else:
        raise ValueError('Unknown input type')

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
    for s, (c, w, d, se) in enumerate(zip(coefficients.data,
                                          recomposition_weights,
                                          scale_denoise_coefficients,
                                          coefficients.scaling_function.sigma_e)):
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
                cv2.filter2D(power, -1, atrous_kernel, dst=local_power, borderType=cv2.BORDER_REFLECT)
                local_power[local_power <= 0] = 1e-15
                np.sqrt(local_power, out=local_power)
            else:
                local_power = 1
            c *= coefficients.significance(d, s, soft_threshold=soft_threshold)
        pwr.append(local_power)
        c *= w*power_norm/pwr[s]

    recon = np.sum(coefficients, axis=0)

    return recon, pwr
