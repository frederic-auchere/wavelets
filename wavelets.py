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
from scipy.ndimage import convolve
# from scipy.ndimage import label, find_objects

__all__ = ['AtrousTransform', 'B3spline', 'Triangle']
__version__ = '0.0.1'


class Coefficients:

    def __init__(self, data, scaling_function):
        self.data = data
        self.scaling_function = scaling_function
        self.noise = None

    def __len__(self):
        return len(self.data)

    def __array__(self):
        return self.data

    def get_noise(self):
        sigma_e = self.scaling_function.sigma_e[0]
        return np.median(np.abs(self.data[0])) / 0.6745 / sigma_e

    def significance(self, sigma, scale, soft_threshold=True):
        if self.noise is None:
            self.noise = self.get_noise()
        sigma_e = self.scaling_function.sigma_e(scale)
        if soft_threshold:
            r = np.abs(self.data[scale] / (sigma * self.noise * sigma_e))
            return special.erf(r / np.sqrt(2))
        else:
            s = np.abs(self.data[scale]) > (sigma * self.noise * sigma_e)
            # regions, _ = label(s)
            # slices = find_objects(regions)
            # w = np.zeros_like(coeff)
            # for slc in slices:
            #     ok = s[slc]
            #     n_pix = ok.sum()
            #     w[slc][ok] = special.erf((n_pix-1))
            return s

    def de_noise(self, sigma, weights=None, soft_threshold=True):
        if weights is None:
            weights = (1,)*len(sigma)
        for scl, (c, sig, wgt) in enumerate(zip(self.data, sigma, weights)):
            c *= wgt*self.significance(sig, scl, soft_threshold=soft_threshold)


class AbstractScalingFunction:
    """
    Abstract class for scaling functions
    """

    def __init__(self, name, n_dim):
        self.name = name
        self.n_dim = n_dim
        self.kernel = self.make_kernel()

    @property
    def coefficients_1d(self):
        raise NotImplementedError

    @property
    def coefficients_2d(self):
        x = np.expand_dims(self.coefficients_1d, 0)
        return x.T @ x

    @property
    def coefficients_3d(self):
        b = np.expand_dims(self.coefficients_2d, 0)
        x = np.expand_dims(self.coefficients_1d, 0)
        return b.T @ x

    def make_kernel(self):
        if self.n_dim == 1:
            return self.coefficients_1d
        elif self.n_dim == 2:
            return self.coefficients_2d
        elif self.n_dim == 3:
            return self.coefficients_3d
        else:
            raise ValueError("Unsupported number of dimensions")

    def atrous_kernel(self, scale):

        kernel = np.zeros([(shp-1)*2**scale + 1 for shp in self.kernel.shape])
        strides = (slice(None, None, 2**scale),)*self.n_dim
        kernel[strides] = self.kernel

        return kernel

    @property
    def sigma_e(self):
        if self.n_dim == 1:
            sigma_e = self.sigma_e_1d
        elif self.n_dim == 2:
            sigma_e = self.sigma_e_2d
        elif self.n_dim == 3:
            sigma_e = self.sigma_e_3d
        else:
            raise ValueError("Unsupported number of dimensions")
        return sigma_e

    @property
    def sigma_e_1d(self):
        raise NotImplementedError

    @property
    def sigma_e_2d(self):
        raise NotImplementedError

    @property
    def sigma_e_3d(self):
        raise NotImplementedError

    def compute_noise_weights(self, n_scales, n_trials=100):
        transform = AtrousTransform(self.__class__)
        std = np.zeros(n_scales)
        for i in range(n_trials):
            data = np.random.normal(size=(2**n_scales,)*self.n_dim)
            coefficients = transform(data, n_scales)
            std += coefficients.data[:-1].std(axis=tuple(range(1, self.n_dim+1)))
        std /= n_trials
        return std


class Triangle(AbstractScalingFunction):
    """
    Triangle scaling function
    See appendix A of J.-L. Starck & F. Murtagh, Handbook of Astronomical Data
    Analysis, Springer-Verlag
    """

    def __init__(self, *args, **kwargs):
        super().__init__('triangle',
                         *args, **kwargs)

    @property
    def coefficients_1d(self):
        return np.array([1/4, 1/2, 1/4])

    @property
    def sigma_e_2d(self):
        return np.array([0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005, 0.00280, 0.00135, 0.00085, 0.00029])

    @property
    def sigma_e_3d(self):
        return np.array([0.956, 0.120, 0.035, 0.012, 0.004, 0.001, 0.0005])


class B3spline(AbstractScalingFunction):
    """"
    B3spline scaling function
    See appendix A of J.-L. Starck & F. Murtagh, Handbook of Astronomical Data
    Analysis, Springer-Verlag
    """

    def __init__(self, *args, **kwargs):
        super().__init__('b3spline',
                         *args, **kwargs)

    @property
    def coefficients_1d(self):
        return np.array([1/16, 1/4, 3/8, 1/4, 1/16])

    @property
    def sigma_e_1d(self):
        return np.array([0.72514976, 0.28538683, 0.17901161, 0.12222841, 0.08469601,
                         0.06027006, 0.04242257, 0.02919823, 0.01805671, 0.01383672])

    @property
    def sigma_e_2d(self):
        return np.array([0.8907e-01, 2.0072e-01, 8.5551e-02, 4.1261e-02, 2.0470e-02,
                         1.0232e-02, 5.1435e-03, 2.6008e-03, 1.3161e-03, 6.7359e-04])

    @property
    def sigma_e_3d(self):
        return np.array([0.956, 0.120, 0.035, 0.012, 0.004, 0.001, 0.0005])


class AtrousTransform:
    """
    Performs 'à trous' wavelet transform of input array arr over level scales,
    as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
    Astronomical Data Analysis, Springer-Verlag

    Uses either a recursive algorithm (faster for large numbers of scales) or
    a standard algorithm for typically 2 scales or less.

    arr: input array
    level: desired number of scales. The output has level + 1 planes
    scaling_function: the base wavelet. The default is a b3spline.
    recursive_threshold: number of scales (level) at and above which the
                         recursive algorithm is used. To force useage of the
                         recursive algorithm, set recursive_threshold = 0.
                         To force usage of the standard algorithm, set
                         recursive_threshold = level+1
    """

    def __init__(self, scaling_function_class=B3spline):

        self.scaling_function_class = scaling_function_class

    def __call__(self, arr, level, recursive=False):

        if arr.ndim > 3:
            raise ValueError("Unsupported number of dimensions")

        scaling_function = self.scaling_function_class(arr.ndim)
        if recursive:
            return Coefficients(
                                self.atrous_recursive(arr,
                                                      level,
                                                      scaling_function),
                                scaling_function
            )
        else:
            return Coefficients(
                                self.atrous_standard(arr,
                                                     level,
                                                     scaling_function),
                                scaling_function
            )

    @staticmethod
    def atrous_recursive(arr, level, scaling_function):
        """
        Performs 'à trous' wavelet transform of input array arr over level scales,
        as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
        Astronomical Data Analysis, Springer-Verlag

        Uses the recursive algorithm.

        arr: input array
        level: desired number of scales. The output has level + 1 planes
        scaling_function: the base wavelet. The default is a b3spline.

        C0  -4  -3  -2  -1   0   1   2   3   4

        C1  -4  -3  -2  -1   0   1   2   3   4

        C2  -4  -3  -2  -1   0   1   2   3   4
        """

        def recursive_convolution(conv, s=0, dx=0, dy=0, dz=0):
            """
            Recursively computes the 'à trous' convolution of the input array by
            extracting 'à trous' sub-arrays instead of using an 'à trous' kernel.
            This can be faster than convolving with an 'à trous' kernel since the
            latter takes time to compute the contribution of the zeros ('trous')
            of the kernel while it is zero by definition and is thus unnecessary.
            There is overhead due to the necessary copy of the 'à trous' sub-arrays
            before convolution by the base kernel.

            conv: array or sub-array to be convolved
            s=0: current scale at which the convolution is performed
            dx=0, dy=0: current offsets of the sub-array
            """
            if conv.ndim == 2:
                cv2.filter2D(conv,
                             -1,        # Same pixel depth as input
                             kernel,    # Kernel known from outer context
                             conv,      # In place operation
                             (-1, -1),  # Anchor is at kernel center
                             0,         # Optional offset
                             cv2.BORDER_REFLECT)
            else:
                convolve(conv,
                         kernel,
                         output=conv,
                         mode='mirror')

            stride = (slice(None, None, 2**s),)*conv.ndim
            coeffs[stride] = conv

            if s == level-1:
                return

            # For given subscale, extracts one pixel out of two on each axis.
            # By default .copy() returns c-ordered arrays (as opposed to np.copy())
            if conv.ndim == 2:
                recursive_convolution(conv[0::2, 0::2].copy(), s=s+1, dx=dx, dy=dy)
                recursive_convolution(conv[1::2, 0::2].copy(), s=s+1, dx=dx, dy=dy+2**s)
                recursive_convolution(conv[0::2, 1::2].copy(), s=s+1, dx=dx+2**s, dy=dy)
                recursive_convolution(conv[1::2, 1::2].copy(), s=s+1, dx=dx+2**s, dy=dy+2**s)
            else:
                recursive_convolution(conv[0::2, 0::2, 0::2].copy(), s=s+1, dx=dx, dy=dy, dz=dz)
                recursive_convolution(conv[0::2, 1::2, 0::2].copy(), s=s+1, dx=dx, dy=dy+2**s, dz=dz)
                recursive_convolution(conv[0::2, 0::2, 1::2].copy(), s=s+1, dx=dx+2**s, dy=dy, dz=dz)
                recursive_convolution(conv[0::2, 1::2, 1::2].copy(), s=s+1, dx=dx+2**s, dy=dy+2**s, dz=dz)

                recursive_convolution(conv[1::2, 0::2, 0::2].copy(), s=s+1, dx=dx, dy=dy, dz=dz+2**s)
                recursive_convolution(conv[1::2, 1::2, 0::2].copy(), s=s+1, dx=dx, dy=dy+2**s, dz=dz+2**s)
                recursive_convolution(conv[1::2, 0::2, 1::2].copy(), s=s+1, dx=dx+2**s, dy=dy, dz=dz+2**s)
                recursive_convolution(conv[1::2, 1::2, 1::2].copy(), s=s+1, dx=dx+2**s, dy=dy+2**s, dz=dz+2**s)

        kernel = scaling_function.kernel

        coeffs = np.empty((level+1,) + arr.shape, dtype=arr.dtype)
        coeffs[0] = arr

        recursive_convolution(arr.copy())  # Input array is modified-> copy

        for s in range(level):  # Computes coefficients from convolved arrays
            coeffs[s] -= coeffs[s+1]

        return coeffs

    @staticmethod
    def atrous_standard(arr, level, scaling_function):
        """
        Performs 'à trous' wavelet transform of input array arr over level scales,
        as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
        Astronomical Data Analysis, Springer-Verlag

        Uses the standard algorithm.

        arr: input array
        level: desired number of scales. The output has level + 1 planes
        scaling_function: the base wavelet. The default is a b3spline.
        """

        coeffs = np.empty((level + 1,) + arr.shape, dtype=arr.dtype)
        coeffs[0] = arr

        for s in range(level):  # Chained convolution
            atrous_kernel = scaling_function.atrous_kernel(s).astype(arr.dtype)

            if arr.ndim == 2:
                cv2.filter2D(coeffs[s],
                             -1,  # Same pixel depth as input
                             atrous_kernel,
                             coeffs[s + 1],  # Result goes in next scale
                             (-1, -1),  # Anchor is kernel center
                             0,  # Optional offset
                             cv2.BORDER_REFLECT)
            else:
                convolve(coeffs[s],
                         atrous_kernel,
                         output=coeffs[s + 1],
                         mode='mirror')

            coeffs[s] -= coeffs[s + 1]

        return coeffs


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

        coeffs.de_noise(dns, weights=wgt, soft_threshold=soft_threshold)

        coeffs.data.sum(axis=0, out=out[c])

    return out
