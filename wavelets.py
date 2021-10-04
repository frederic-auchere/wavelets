#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:47:33 2018

@author: fauchere
"""

import cv2
import copy
import numpy as np
from scipy import special

sigma_e = [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005,
           0.00280, 0.00135, 0.00085, 0.00029]


class ScalingFunction():

    def __init__(self, name, coefficients, dtype=np.float32):
        self.name = name
        self.coefficients = coefficients(dtype=dtype)


class Triangle2D(ScalingFunction):

    def __init__(self, dtype=np.float32):
        super().__init__('triangle2d',
                         triangle2d,
                         dtype=dtype)


class B3spline2D(ScalingFunction):

    def __init__(self, dtype=np.float32):
        super().__init__('b3spline2d',
                         b3spline2d,
                         dtype=dtype)


class Atrous():

    """
    Performs 'à trous' wavelet transform of input array arr over level scales,
    as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
    Astronomical Data Analysis, Springer-Verlag

    Uses either a recursive algorithm (faster for large numbers of scales) or
    a standard algorithm for typically 2 scales or less.

    arr: input array
    level: desired number of scales. The output has level + 1 planes
    kernel: an ndarray defining the base wavelet. The default is a b3spline.
    recursive_threshold: number of scales (level) at and above which the
                         recursive algorithm is used. To force useage of the
                         recursive algorithm, set recursive_threshold = 0.
                         To force useage of the standard algorithm, set
                         recursive_threshold = level+1
    """

    def __init__(self, scaling_function, level, recursive_threshold=3):

        self.scaling_function = scaling_function()
        self.level = level
        self.recursive_threshold = recursive_threshold

    def __call__(self, arr):

        # If (by default) 3 scales or more, the recursive algorithm is faster
        if self.level >= self.recursive_threshold:
            return atrous_recursive(arr,
                                    self.level,
                                    kernel=self.scaling_function.coefficients)
        else:  # If less than (by default) 3 scales, standard algorithm faster
            return atrous_standard(arr,
                                   self.level,
                                   kernel=self.scaling_function.coefficients)


#def denoise(img, sigma_s, method='hard', level=3):
#
#    sigma_e = [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005]
#
#    if img.ndim == 3:
#        channels = [0, 1, 2]
#    else:
#        channels = [Ellipsis]
#
#    if type(sigma_s) is np.ndarray:
#        if sigma_s.shape != img.shape:
#            raise ValueError('Wavelets.denoise: noise array must have same diemsnions as img')
#    else:
#        sigma_s = np.full(int(1+2*(img.ndim-2)), sigma_s)
#
#    recon = np.empty_like(img)
#    for c in channels:
#        coeffs = atrous(img[c], level=level)
#        recon[c] = np.copy(coeffs[level])
#        for l in range(level):
#            if method == 'hard':
#                coeffs[l][np.abs(coeffs[l]) < (3*sigma_s[c]*sigma_e[l])] = 0
#                recon[c] += coeffs[l]
#            elif method == 'soft':
#                coeffs[l] *= special.erf(np.abs(coeffs[l]/(sigma_s[c]*sigma_e[l]))/np.sqrt(2))
#                recon[c] += coeffs[l]
#
#    return recon


def get_noise(coeffs):
    return np.median(np.abs(coeffs[0]))/0.6745/sigma_e[0]

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
                l = [[],]*ndims
            else:
                l = [[param],]*ndims
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

def signif(coeff, sigma, noise, sigma_e, soft_threshold=True):
    if soft_threshold:
        return special.erf(np.abs(coeff/(sigma*noise*sigma_e))/np.sqrt(2))
    else:
        return np.abs(coeff) < (sigma*noise*sigma_e)

def enhance(*args, weights=None, denoise=None, soft_threshold=True, out=None, mrs=None, **kwargs):
    """
    Performs denoising and / or enhancement by modification of wavelet
    coefficients

    Denoising is described in J.-L. Starck & F. Murtagh, Handbook of
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

    for c in channels:

        dns = denoise if c is Ellipsis else denoise[c]
        wgt = weights if c is Ellipsis else weights[c]

        # adds 0 to prevent denoising the last wavelet plane (smoothed image)
        dns.extend([0])
        if len(wgt) < len(dns):
            wgt.extend([1]*(len(dns) - len(wgt)))
        elif len(dns) < len(wgt):
            dns.extend([0]*(len(wgt) - len(dns)))

        # atrous returns len(wgt)-1+1=len(wgt) coefficients
        coeffs = atrous(img[c], level=len(wgt)-1, **kwargs)
        if len(args) == 2:
            sigma_s = args[1] if c is Ellipsis else args[1][c]
            # if type(sigma_s) is np.ndarray:
            #     print(sigma_s.shape, type(args[1]), len(args[1]))
#            if type(sigma_s) is np.ndarray:
#                if sigma_s.shape != img.shape:
#                    raise ValueError('Wavelets.enhance: noise array must '
#                                     'have same dimensions as img')
        else:
            sigma_s = get_noise(coeffs)
        for coeff, w, d, se in zip(coeffs, wgt, dns, sigma_e):
            # no denoising of planes with d = 0
            if d != 0:
                s = signif(coeff, d, sigma_s, se, soft_threshold=soft_threshold)
                if soft_threshold:
                    coeff *= s#special.erf(np.abs(coeff/(d*sigma_s*se))/np.sqrt(2))
                else:
                    coeff[s] = 0#np.abs(coeff) < (d*sigma_s*se)] = 0
            if w != 1:
                coeff *= w
        coeffs.sum(axis=0, out=out[c])

    return out

#def enhance(*args, weights=None, denoise=None, out=None, chroma=0):
#
#    sigma_e = [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005, 0.00280, 0.00135, 0.00085, 0.00029]
#
#    img = args[0]
#
#    channels = [0, 1, 2] if img.ndim == 3 else [Ellipsis]
#
#    if out is None:
#        out = np.empty_like(img)
#
#    wgt = [] if weights is None else weights
#    dns = [] if denoise is None else denoise
#    # adds 0 to prevent denoising last wavelet plane (smoothed image)
#    dns.extend([0])
#    if len(wgt) < len(dns):
#        wgt.extend([1]*(len(dns) - len(wgt)))
#    elif len(dns) < len(wgt):
#        dns.extend([0]*(len(wgt) - len(dns)))
#
#    # Need all planes up front for color noise
#    if img.ndim == 3:
#        coeffs = np.empty((3, len(wgt),) + img.shape[1:], dtype=img.dtype)
#    else:
#        coeffs = np.empty((len(wgt),) + img.shape, dtype=img.dtype)
#    for c in channels:
#        coeffs[c] = atrous(img[c], level=len(wgt)-1)
#
#    if len(args) == 2:
#        sigma_s = args[1]
#        if type(sigma_s) is np.ndarray:
#            if sigma_s.shape != img.shape:
#                raise ValueError('Wavelets.enhance: noise array must '
#                                 'have same dimensions as img')
#    else:
#        sigma_s = [np.median(np.abs(coeff[0]))/0.6745/sigma_e[0] for coeff in coeffs]
#
#    for c in channels:
#        for ic, w, d, se in zip(range(len(wgt)), wgt, dns, sigma_e):
#            # no denoising of planes with d = 0
#            if d > 0:
#                coeffs[c][ic][np.abs(coeffs[c][ic]) < (d*sigma_s[c]*se)] = 0
#            elif d < 0:
#                dum = np.ones_like(coeffs[0][0])
#                for i in channels:
#                    toto = special.erf(np.abs(coeffs[i][ic]/(d*sigma_s[i]*se))/np.sqrt(2))
#                    if i == c:
#                        dum *= toto
#                    else:
#                        dum *= 1 - chroma*(1 - toto)
#                coeffs[c][ic] *= dum
##            if c == 0 and ic == 2 and weights is None:
##                plt.imshow(coeffs[c][ic], vmax=2)
#            if w != 1:
#                coeffs[c][ic] *= w
#        if c == 0 and weights is None:
#            plt.imshow(coeffs[c][0] + coeffs[c][1] + coeffs[c][2], vmax=2)
#        coeffs[c].sum(axis=0, out=out[c])
#
#    return out


def triangle2d(dtype=np.float32):

    """
    Returns array of coefficients for the 2D triangule scaling function

    See appendix A of J.-L. Starck & F. Murtagh, Handbook of Astronomical Data
    Analysis, Springer-Verlag

    dtype: desired dtype of the ndarray output
    """

    triangle1d = np.array([1/4, 1/1, 1/4], dtype=dtype)
    x = triangle1d.reshape(1, -1)
    return np.dot(x.T, x)


def b3spline2d(dtype=np.float32):

    """
    Returns array of coefficients for the 2D B3-spline scaling function

    See appendix A of J.-L. Starck & F. Murtagh, Handbook of Astronomical Data
    Analysis, Springer-Verlag

    dtype: desired dtype of the ndarray output
    """

    b3spline1d = np.array([1/16, 1/4, 3/8, 1/4, 1/16], dtype=dtype)
    x = b3spline1d.reshape(1, -1)
    return np.dot(x.T, x)


def atrous_recursive(arr, level, kernel=None, recursive_threshold=3):

    """
    Performs 'à trous' wavelet transform of input array arr over level scales,
    as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
    Astronomical Data Analysis, Springer-Verlag

    Uses the recursive algorithm.

    arr: input array
    level: desired number of scales. The output has level + 1 planes
    kernel: an ndarray defining the base wavelet. The default is a b3spline.
    """

    if kernel is None:
        kernel = b3spline2d(dtype=arr.dtype)

    coeffs = np.empty((level+1,) + arr.shape, dtype=arr.dtype)
    coeffs[0] = arr

    def recursive_convolution(conv, s=0, dx=0, dy=0):

        """
        Recursively conputes the 'à trous' convolution of the input array by
        extracting 'à trous' sub-arrays instead of using an 'à trous' kernel.
        This can be faster than convolving with an 'à trous' kernel since the
        latter takes time to compute the contribution of the zeros ('trous')
        of the kernel while it is zero by definition and is thus unnecessary.
        There is overhead due to the necessary copy of the 'à trous' sub-arrays
        before convolution by the base kernel.
        Tests show a gain in speed for decompositions in three scales or more.
        For two scale or less the non-recursive approach has less overhead and
        is faster.

        conv: array or sub-array to be convolved
        s=0: current scale at which the convolution is performed
        dx=0, dy=0: current offsets of the sub-array
        """

        cv2.filter2D(conv,
                     -1,        # Same pixel depth as input
                     kernel,    # Kernel known from outer context
                     conv,      # In place operation
                     (-1, -1),  # Anchor is at kernel center
                     0,         # Optional offset
                     cv2.BORDER_REFLECT)
        coeffs[s+1, dy::2**s, dx::2**s] = conv

        if s == level-1:
            return

        # For given subscale, extracts one pixel out of two on each axis.
        # By default .copy() returns c-ordered arrays (as opposed to np.copy())
        recursive_convolution(conv[0::2, 0::2].copy(), s=s+1, dx=dx, dy=dy)
        recursive_convolution(conv[1::2, 0::2].copy(), s=s+1, dx=dx, dy=dy+2**s)
        recursive_convolution(conv[0::2, 1::2].copy(), s=s+1, dx=dx+2**s, dy=dy)
        recursive_convolution(conv[1::2, 1::2].copy(), s=s+1, dx=dx+2**s, dy=dy+2**s)

    recursive_convolution(arr.copy())  # Input array is modified-> copy

    for s in range(level):  # Computes coefficients from convolved arrays
        coeffs[s] -= coeffs[s+1]

    return coeffs


def atrous_standard(arr, level, kernel=None):

    """
    Performs 'à trous' wavelet transform of input array arr over level scales,
    as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
    Astronomical Data Analysis, Springer-Verlag

    Uses the standard algorithm.

    arr: input array
    level: desired number of scales. The output has level + 1 planes
    kernel: an ndarray defining the base wavelet. The default is a b3spline.
    """

    if kernel is None:
        kernel = b3spline2d(dtype=arr.dtype)

    coeffs = np.empty((level+1,) + arr.shape, dtype=arr.dtype)
    coeffs[0] = arr

    for s in range(level):  # Chained convolution
        atrouskernel = np.zeros([(shp-1)*2**s + 1 for shp in kernel.shape],
                                dtype=kernel.dtype)
        atrouskernel[::2**s, ::2**s] = kernel

        cv2.filter2D(coeffs[s],
                     -1,           # Same pixel depth as input
                     atrouskernel,
                     coeffs[s+1],  # Result goes in next scale
                     (-1, -1),     # Anchor is kernel center
                     0,            # Optional offset
                     cv2.BORDER_REFLECT)
        coeffs[s] -= coeffs[s+1]

    return coeffs


def atrous(arr, level, kernel=None, recursive_threshold=3):

    """
    Performs 'à trous' wavelet transform of input array arr over level scales,
    as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
    Astronomical Data Analysis, Springer-Verlag

    Uses either a recursive algorithm (faster for large numbers of scales) or
    a standard algorithm for typically 2 scales or less.

    arr: input array
    level: desired number of scales. The output has level + 1 planes
    kernel: an ndarray defining the base wavelet. The default is a b3spline.
    recursive_threshold: number of scales (level) at and above which the
                         recursive algorithm is used. To force useage of the
                         recursive algorithm, set recursive_threshold = 0.
                         To force useage of the standard algorithm, set
                         recursive_threshold = level+1
    """

    # If (by default) 3 scales or more, the recursive algorithm is faster?
    if level >= recursive_threshold:
        return atrous_recursive(arr, level, kernel=kernel)
    else:  # If less than (by default) 3 scales, standard algorithm is faster
        return atrous_standard(arr, level, kernel=kernel)
