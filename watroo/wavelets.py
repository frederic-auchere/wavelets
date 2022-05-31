# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import special
from scipy.ndimage import convolve

__all__ = ['AtrousTransform', 'B3spline', 'Triangle', 'Coefficients']


def sdev_loc(image, kernel, variance=False):
    mean2 = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mean2 **= 2
    vari = cv2.filter2D(image**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    vari -= mean2
    vari[vari <= 0] = 1e-20
    if variance:
        return vari
    else:
        return np.sqrt(vari)


def bilateral_filter(image, kernel, variance, mode="symmetric"):

    hwx, hwy = kernel.shape[1]//2, kernel.shape[0]//2
    padded = np.pad(image, (hwy, hwx), mode=mode)
    output = kernel[hwy, hwx]*image
    norm = np.full_like(image, kernel[hwy, hwx])
    shifted = np.empty_like(image)

    y, x, = np.indices(kernel.shape)
    x = kernel.shape[1] - 1 - x
    y = kernel.shape[0] - 1 - y
    mask = np.ones(kernel.shape, dtype=bool)
    mask[hwy, hwx] = False
    for dx, dy, k in zip(x[mask], y[mask], kernel[mask]):
        shifted[:] = padded[dy:dy+image.shape[0], dx:dx+image.shape[1]]
        diff = k*np.exp(-0.5*((image - shifted)**2)/variance)
        norm += diff
        output += shifted*diff
    return output/norm


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
            if self.noise == 0:
                return np.ones_like(self.data[0])
        sigma_e = self.scaling_function.sigma_e[scale]
        if soft_threshold:
            if sigma != 0:
                r = np.abs(self.data[scale] / (sigma * self.noise * sigma_e))
                return special.erf(r / np.sqrt(2))
#                return r / (1 + r)
            else:
                return 1
        else:
            return np.abs(self.data[scale]) > (sigma * self.noise * sigma_e)

    def denoise(self, sigma, weights=None, soft_threshold=True):
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
            data = np.random.normal(size=(2**n_scales,)*self.n_dim).astype(np.float32)
            coefficients = transform(data, n_scales)
            std += coefficients.data[:-1].std(axis=tuple(range(1, self.n_dim+1)))
        std /= n_trials
        return std


class Triangle(AbstractScalingFunction):
    """
    Triangle scaling function (3 x 3 interpolation)
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
    def sigma_e_1d(self):
        return np.array([0.60840933, 0.33000059, 0.21157957, 0.145824, 0.10158388,
                         0.07155912, 0.04902655, 0.03529812, 0.02409187, 0.01722846,
                         0.01144442])

    @property
    def sigma_e_2d(self):
        return np.array([0.7999247, 0.27308452, 0.11998217, 0.05793947, 0.0288104,
                         0.01447795, 0.00733832, 0.0037203, 0.00192882, 0.00098568,
                         0.00048533])

    @property
    def sigma_e_3d(self):
        return np.array([0.89736751, 0.19514386, 0.06239262, 0.02311278, 0.00939645])


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
                         0.06027006, 0.04242257, 0.02919823, 0.01805671, 0.01383672,
                         0.00943623])

    @property
    def sigma_e_2d(self):
        return np.array([8.907e-01, 2.0072e-01, 8.5551e-02, 4.1261e-02, 2.0470e-02,
                         1.0232e-02, 5.1435e-03, 2.6008e-03, 1.3161e-03, 6.7359e-04,
                         4.0040e-04])

    @property
    def sigma_e_3d(self):
        return np.array([0.95633954, 0.12491933, 0.03933029, 0.01489642, 0.0064108])


class AtrousTransform:
    """
    Performs 'à trous' wavelet transform of input array arr over level scales,
    as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
    Astronomical Data Analysis, Springer-Verlag
    """""

    def __init__(self, scaling_function_class=B3spline):
        """
        scaling_function: the base scaling function. The default is a b3spline.
        """
        self.scaling_function_class = scaling_function_class

    def __call__(self, arr, level, recursive=False, bilateral=None):
        """
        Performs the 'à trous' transform.
        Uses either a recursive algorithm or a standard algorithm.

        arr: input array
        level: desired number of scales. The output has level + 1 planes
        recursive: whether or not to use the recursive algorithm
        """
        if arr.ndim > 3:
            raise ValueError("Unsupported number of dimensions")

        scaling_function = self.scaling_function_class(arr.ndim)
        if recursive or bilateral:
            return Coefficients(
                                self.atrous_recursive(arr,
                                                      level,
                                                      scaling_function,
                                                      bilateral=bilateral),
                                scaling_function,
            )
        else:
            return Coefficients(
                                self.atrous_standard(arr,
                                                     level,
                                                     scaling_function),
                                scaling_function
            )

    @staticmethod
    def atrous_recursive(arr, level, scaling_function, bilateral=None):
        """
        Performs 'à trous' wavelet transform of input array arr over level scales,
        as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
        Astronomical Data Analysis, Springer-Verlag

        Uses the recursive algorithm.

        arr: input array
        level: desired number of scales. The output has level + 1 planes
        scaling_function: the base scaling_function.

        C0  -4  -3  -2  -1   0   1   2   3   4

        C1  -4  -3  -2  -1   0   1   2   3   4

        C2  -4  -3  -2  -1   0   1   2   3   4
        """

        def recursive_convolution(conv, s=0, dx=0, dy=0, dz=0, bilateral=bilateral):
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
            if bilateral is None:
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
            else:
                # A Gaussian with d=5 & sigmaSpace=1.1 approximates a B3spline
                # conv[:] = cv2.bilateralFilter(conv, 5, bilateral*2**s, 1.1, borderType=cv2.BORDER_REFLECT)
                # conv[:] = cv2.bilateralFilter(conv, 5, np.std(conv), 1.1, borderType=cv2.BORDER_REFLECT)
                # # sdev = bilateral*2*np.std(conv)
                variance = sdev_loc(conv, kernel, variance=True)
                conv[:] = bilateral_filter(conv, kernel, variance)

            if conv.ndim == 2:
                coeffs[s+1, dy::2**s, dx::2**s] = conv
            else:
                coeffs[s+1, dz::2**s, dy::2**s, dx::2**s] = conv

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
    def atrous_standard(arr, level, scaling_function, bilateral=None):
        """
        Performs 'à trous' wavelet transform of input array arr over level scales,
        as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
        Astronomical Data Analysis, Springer-Verlag

        Uses the standard algorithm.

        arr: input array
        level: desired number of scales. The output has level + 1 planes
        scaling_function: the base scaling function.
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
