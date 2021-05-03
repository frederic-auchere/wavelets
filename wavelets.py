#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:47:33 2018

@author: fauchere
"""

import cv2
import numpy as np
from scipy import special


class ScalingFunction():

    """
    Expected amplitude of wavelet coefficients in 2D images containing white noise 
    Must be redefined for each scaling function
    """    
    sigma_e = [1]

    def __init__(self, name, coefficients, dtype=np.float32):
        self.name = name
        self.coefficients = coefficients(dtype=dtype)

    def get_noise(self, coeffs):
        return np.median(np.abs(coeffs[0]))/0.6745/self.sigma_e[0]

class Triangle2D(ScalingFunction):

    def __init__(self, dtype=np.float32):
        super().__init__('triangle2d',
                         triangle2d,
                         dtype=dtype)


class B3spline2D(ScalingFunction):

    sigma_e = [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005,
               0.00280, 0.00135, 0.00085, 0.00029]

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

def enhance(*args, dn2photons=None, weights=None, denoise=None, out=None, mrs=None):
    """
    Performs denoising and / or enhancement by modification of wavelet
    coefficients as described in J.-L. Starck & F. Murtagh, Handbook of
    Astronomical Data Analysis, Springer-Verlag
    
    args: the first parameter is the image to be denoised/enhanced
          the second parameter, if present, is a noise map. If not present and 
          dn2photons is None (see below), the
          noise in the data is estimated using the first scale.
    dn2photons: conversion coefficient to be applied to compute photon noise
                from the input data. If no noise map is provided, this can be
                if the data is shot noise dominated.
    denoise: list of desoising coefficients, one for each scale to be desnoised.
             If coefficients are positive, wavelets coefficients at the
             corresponding scale are preserved if they are greater than 
             this coefficient times sigma_e, the excpeted amplitude for 
             white noise.
             If coefficients are negative, soft threshilding is applied.
             If coefficients are zero, no denoising is applied.
    weights: list of weights to apply to each scale during reconstruction.
             Can be used as an enhancement tool.
    mrs: returns the multi-resolution support of the image
    out: output array if specified

    Example assuming a 2D image in "image.fits" containing values in detected
    photons with a read noise equivalent to 2 detected photons
        from astropy.io import fits
        image = fits.getdata("image.fits")
        photon_noise = np.sqrt(image)
        read_noise = 2
        noise = np.sqrt(read_noise**2 + photon_noise**2)
        output = enhance(image, noise, denoise=[-3, -3])
    """

    img = args[0]

    if img.ndim == 3:
        channels = [0, 1, 2]
    else:
        channels = [Ellipsis]

    if out is None:
        out = np.empty_like(img)

    wgt = [] if weights is None else weights
    dns = [] if denoise is None else denoise
    # adds 0 to prevent denoising last wavelet plane (smoothed image)
    dns.extend([0])
    if len(wgt) < len(dns):
        wgt.extend([1]*(len(dns) - len(wgt)))
    elif len(dns) < len(wgt):
        dns.extend([0]*(len(wgt) - len(dns)))

    kernel = b3spline2d

    for c in channels:
        coeffs = atrous(img[c], level=len(wgt)-1)
        if len(args) == 2:
            sigma_s = args[1][c]
        elif dn2photons is not None:  # See Murtagh et al. 1995 A&A 112, 179, Eq .12
            sigma_s =  2*np.sqrt(img*dn2photons + 3/8)    
        else:
            sigma_s = kernel.get_noise(coeffs)
        for coeff, w, d, se in zip(coeffs, wgt, dns, kernel.sigma_e):
            # no denoising of planes with d = 0
            if d > 0:
                coeff[np.abs(coeff) < (d*sigma_s*se)] = 0
            elif d < 0:
                coeff *= special.erf(np.abs(coeff/(d*sigma_s*se))/np.sqrt(2))
            if w != 1:
                coeff *= w
        coeffs.sum(axis=0, out=out[c])

    return out

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

    # If (by default) 3 scales or more, the recursive algorithm is faster
    if level >= recursive_threshold:
        return atrous_recursive(arr, level, kernel=kernel)
    else:  # If less than (by default) 3 scales, standard algorithm is faster
        return atrous_standard(arr, level, kernel=kernel)
