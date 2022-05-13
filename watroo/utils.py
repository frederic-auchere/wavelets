import copy
import numpy as np
import cv2
from . import AtrousTransform, B3spline

__all__ = ['denoise']


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

        # adds 0 to prevent de-noising the last wavelet plane (smoothed image)
        dns.extend([0])
        if len(wgt) < len(dns):
            wgt.extend([1]*(len(dns) - len(wgt)))
        elif len(dns) < len(wgt):
            dns.extend([0]*(len(wgt) - len(dns)))

        coeffs = atrous(img[c], len(wgt)-1)
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


def mgn(img, k=0.7, h=0.7, gamma=3.2, scales=[1.25, 2.5, 5, 10, 20, 40]):
    c = 0
    for s in scales:
        conv = img - cv2.GaussianBlur(img, (0, 0), s)
        std = np.sqrt(cv2.GaussianBlur(conv**2, (0, 0), s))
        gd = std > 0
        conv[gd] /= std[gd]
        conv = np.arctan(k*conv)
        c += conv
    c /= len(scales)

    g = np.copy(img)
    g -= g.min()
    g /= g.max()
    g **= 1/gamma

    return h*g + (1-h)*c


def wow(image, scaling_function=B3spline, n_scales=None):

    if n_scales is None:
        n_scales = int(np.log2(min(image.shape)) - 2)

    transform = AtrousTransform(scaling_function)
    coefficients = transform(image, n_scales)
    for s, c in enumerate(coefficients.data):
        atrous_kernel = transform.scaling_function_class(2).atrous_kernel(s)
        mean = np.empty_like(c)
        cv2.filter2D(c,
                     -1,  # Same pixel depth as input
                     atrous_kernel,
                     mean,
                     (-1, -1),  # Anchor is kernel center
                     0,  # Optional offset
                     cv2.BORDER_REFLECT)
        vari = np.empty_like(coefficients.data[s])
        cv2.filter2D((c - mean)**2,
                     -1,  # Same pixel depth as input
                     atrous_kernel,
                     vari,
                     (-1, -1),  # Anchor is kernel center
                     0,  # Optional offset
                     cv2.BORDER_REFLECT)
        vari[vari <= 0] = 1e-20
        c /= np.sqrt(vari)
    return np.sum(coefficients.data[:-1], axis=0)
