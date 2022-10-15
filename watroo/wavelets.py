import copy
import cv2
import numpy as np
from scipy import special
from scipy.ndimage import convolve
import numexpr as ne
from tqdm import tqdm


__all__ = ['AtrousTransform', 'B3spline', 'Triangle', 'Coefficients', 'generalized_anscombe', 'convolution']


def generalized_anscombe(signal, alpha=1, g=0, sigma=0, inverse=False):

    if inverse:
        return ((alpha*signal/2)**2 + alpha*g - sigma**2 - 3*alpha/8)/alpha
    else:
        dum = alpha*signal + 3*alpha**2/8 + sigma**2 - alpha*g
        dum[dum <= 0] = 0
        return 2*np.sqrt(dum)/alpha


def atrous_convolution(image, kernel, s):

    hwx, hwy = kernel.shape[1]//2, kernel.shape[0]//2
    padded = np.pad(image, (hwy*2**s, hwx*2**s), mode='reflect')
    output = np.zeros_like(image)
    norm = np.full_like(image, kernel[hwy, hwx])
    shifted = np.empty_like(image)

    y, x = np.indices(kernel.shape)
    x = (kernel.shape[1] - 1 - x)*2**s
    y = (kernel.shape[0] - 1 - y)*2**s

    for dx, dy, k in zip(x.flatten(), y.flatten(), kernel.flatten()):
        output += k*padded[dy:dy+image.shape[0], dx:dx+image.shape[1]]
    return output


def convolution(arr, kernel, output=None):
    if output is None:
        output = np.empty_like(arr)
    if arr.ndim == 2:
        cv2.filter2D(arr,
                     -1,  # Same pixel depth as input
                     kernel,
                     output,
                     (-1, -1),  # Anchor is kernel center
                     0,  # Optional offset
                     cv2.BORDER_REFLECT)
    else:
        convolve(arr,
                 kernel,
                 output=output,
                 mode='mirror')

    return output


def sdev_loc(image, kernel, s=0, variance=False):
    if s == 0:
        mean2 = convolution(image, kernel)**2
        vari = convolution(image**2, kernel)
    else:
        mean2 = atrous_convolution(image, kernel, s)**2
        vari = atrous_convolution(image**2, kernel, s)
    vari -= mean2
    vari[vari <= 0] = 1e-20
    if variance:
        return vari
    else:
        return np.sqrt(vari)


def bilateral_filter(image, kernel, variance, s=0, mode="reflect"):

    half_widths = tuple([s//2 for s in kernel.shape])  # z, y, x order
    padded = np.pad(image, [(hw*2**s,)*2 for hw in half_widths], mode=mode)
    output = kernel[half_widths]*image
    norm = np.full_like(image, kernel[half_widths])

    shifted = np.empty_like(image)
    diff = np.empty_like(image)

    indices = np.indices(kernel.shape)  # z, y, x order
    mask = np.ones(kernel.shape, dtype=bool)
    mask[half_widths] = False
    indices = [(shape - 1 - index[mask])*2**s for index, shape in zip(indices, kernel.shape)]

    for *deltas, k in zip(*indices, kernel[mask]):
        slc = tuple([slice(d, d+s) for d, s in zip(deltas, image.shape)])
        shifted[:] = padded[slc]
        ne.evaluate('k*exp(-((image - shifted)**2)/variance/2)', out=diff)
        norm += diff
        output += shifted*diff
    output /= norm
    return output
    # def range_weighting(image, padded, slc, k):
    #     shifted = padded[slc]
    #     return ne.evaluate('k*exp(-((image - shifted)**2)/variance/2)')
    # parameters = []
    # for *deltas, k in zip(*indices, kernel[mask]):
    #     slc = tuple([slice(d, d+s) for d, s in zip(deltas, image.shape)])
    #     parameters.append((image, padded, slc, k))
    # diff = pool.starmap(range_weighting, parameters)
    # for *deltas, k, d in zip(*indices, kernel[mask], diff):
    #     norm += d
    #     output += padded[slc]*d


class Coefficients:

    def __init__(self, data, scaling_function, bilateral=None):
        self.data = data
        self.scaling_function = scaling_function
        self.bilateral = bilateral
        self.noise = None

    def __len__(self):
        return len(self.data)

    def __array__(self):
        return self.data

    @property
    def sigma_e(self):
        return self.scaling_function.sigma_e(bilateral=self.bilateral)

    def get_noise(self):
        return np.median(np.abs(self.data[0])) / 0.6745 / self.sigma_e[0]

    def significance(self, sigma, scale, soft_threshold=True):
        if sigma != 0:
            if self.noise is None:
                self.noise = self.get_noise()
                if self.noise == 0:
                    return np.ones_like(self.data[0])
            if soft_threshold:
                r = np.abs(self.data[scale] / (sigma * self.noise * self.sigma_e[scale]))
                return special.erf(r)
                # return r / (1 + r)
            else:
                return np.abs(self.data[scale]) > (sigma * self.noise * self.sigma_e[scale])
        else:
            return np.ones_like(self.data[0])

    def denoise(self, sigma, weights=None, soft_threshold=True):
        if weights is None:
            weights = (1,)*len(sigma)
        for scl, (c, sig, wgt) in enumerate(zip(self.data, sigma, weights)):
            c *= wgt*self.significance(sig, scl, soft_threshold=soft_threshold)


class AbstractScalingFunction:
    """
    Abstract class for scaling functions
    """

    coefficients_1d = None
    sigma_e_1d = None
    sigma_e_2d = None
    sigma_e_3d = None
    sigma_e_1d_bilateral = None
    sigma_e_2d_bilateral = None
    sigma_e_3d_bilateral = None

    def __init__(self, name, n_dim):
        self.name = name
        self.n_dim = n_dim
        self.kernel = self.make_kernel()

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

    def sigma_e(self, bilateral=None):
        if bilateral is None:
            if self.n_dim == 1:
                sigma_e = self.sigma_e_1d
            elif self.n_dim == 2:
                sigma_e = self.sigma_e_2d
            elif self.n_dim == 3:
                sigma_e = self.sigma_e_3d
            else:
                raise ValueError("Unsupported number of dimensions")
            return sigma_e
        else:
            if self.n_dim == 1:
                sigma_e = self.sigma_e_1d_bilateral
            elif self.n_dim == 2:
                sigma_e = self.sigma_e_2d_bilateral
            elif self.n_dim == 3:
                sigma_e = self.sigma_e_3d_bilateral
            else:
                raise ValueError("Unsupported number of dimensions")
            return sigma_e

    def compute_noise_weights(self, n_scales, n_trials=100, bilateral=None):
        transform = AtrousTransform(self.__class__, bilateral=bilateral)
        std = np.zeros(n_scales)
        for i in tqdm(range(n_trials)):
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

    coefficients_1d = np.array([1/4, 1/2, 1/4])

    sigma_e_1d = np.array([0.60840933, 0.33000059, 0.21157957, 0.145824, 0.10158388,
                           0.07155912, 0.04902655, 0.03529812, 0.02409187, 0.01722846,
                           0.01144442])

    sigma_e_2d = np.array([0.7999247, 0.27308452, 0.11998217, 0.05793947, 0.0288104,
                           0.01447795, 0.00733832, 0.0037203, 0.00192882, 0.00098568,
                           0.00048533])

    sigma_e_3d = np.array([0.89736751, 0.19514386, 0.06239262, 0.02311278, 0.00939645])

    sigma_e_2d_bilateral = np.array([0.31063172, 0.34575647, 0.23712331, 0.13559906, 0.07172004, 0.03665405,
                                     0.01850046, 0.00928768, 0.00465967, 0.00234445, 0.00119249])

    sigma_e_3d_bilateral = np.array([0.3828863, 0.36182913, 0.19520299, 0.08498861, 0.03363142])

    def __init__(self, *args, **kwargs):
        super().__init__('triangle',
                         *args, **kwargs)


class B3spline(AbstractScalingFunction):
    """
    B3spline scaling function
    See appendix A of J.-L. Starck & F. Murtagh, Handbook of Astronomical Data
    Analysis, Springer-Verlag
    """

    coefficients_1d = np.array([1/16, 1/4, 3/8, 1/4, 1/16])

    sigma_e_1d = np.array([0.72514976, 0.28538683, 0.17901161, 0.12222841, 0.08469601,
                           0.06027006, 0.04242257, 0.02919823, 0.01805671, 0.01383672,
                           0.00943623])

    sigma_e_2d = np.array([8.907e-01, 2.0072e-01, 8.5551e-02, 4.1261e-02, 2.0470e-02,
                           1.0232e-02, 5.1435e-03, 2.6008e-03, 1.3161e-03, 6.7359e-04,
                           4.0040e-04])

    sigma_e_3d = np.array([0.95633954, 0.12491933, 0.03933029, 0.01489642, 0.0064108])

    sigma_e_2d_bilateral = np.array([0.38234752, 0.24305799, 0.16012153, 0.10633541, 0.07083733,
                                     0.04728659, 0.03163678, 0.02122341, 0.01429102, 0.00952376])

    sigma_e_3d_bilateral = np.array([0.44111772, 0.3552894,  0.16137159, 0.05769064, 0.01932497])

    def __init__(self, *args, **kwargs):
        super().__init__('b3spline',
                         *args, **kwargs)


class AtrousTransform:
    """
    Performs 'à trous' wavelet transform of input array arr over level scales,
    as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
    Astronomical Data Analysis, Springer-Verlag
    """""

    def __init__(self, scaling_function_class=B3spline, bilateral=None, bilateral_scaling=True):
        """
        scaling_function: the base scaling function. The default is a b3spline.
        """
        self.scaling_function_class = scaling_function_class
        self.bilateral = bilateral
        self.bilateral_scaling = bilateral_scaling

    def __call__(self, arr, level, recursive=False):
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
        if recursive:
            return Coefficients(
                                self.atrous_recursive(arr,
                                                      level,
                                                      scaling_function),
                                scaling_function,
                                self.bilateral
            )
        else:
            return Coefficients(
                                self.atrous_standard(arr,
                                                     level,
                                                     scaling_function),
                                scaling_function,
                                self.bilateral
            )

    def atrous_recursive(self, arr, level, scaling_function):
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

        sigma_bilateral = copy.copy(self.bilateral) if type(self.bilateral) is list else [self.bilateral, ]*(level+1)
        n_bilateral = len(sigma_bilateral)
        if n_bilateral <= level:
            sigma_bilateral.extend([1, ] * (level - n_bilateral + 1))

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
            if self.bilateral is None:
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
                variance = sdev_loc(conv, kernel, variance=True)*sigma_bilateral[s]**2
                conv[:] = bilateral_filter(conv, kernel, variance, mode='symmetric')

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

        kernel = scaling_function.kernel.astype(arr.dtype)

        hwy, hwx = (kernel.shape[0]//2)*2**(level-1), (kernel.shape[1]//2)*2**(level-1)
        arr = np.pad(arr, (hwy, hwx), mode='reflect')

        coeffs = np.empty((level+1,) + arr.shape, dtype=arr.dtype)
        coeffs[0] = arr

        recursive_convolution(arr)  # Input array is modified-> copy

        for s in range(level):  # Computes coefficients from convolved arrays
            coeffs[s] -= coeffs[s+1]

        return np.copy(coeffs[:, hwy:-hwy, hwx:-hwx])

    def atrous_standard(self, arr, level, scaling_function):
        """
        Performs 'à trous' wavelet transform of input array arr over level scales,
        as described in Appendix A of J.-L. Starck & F. Murtagh, Handbook of
        Astronomical Data Analysis, Springer-Verlag

        Uses the standard algorithm.

        arr: input array
        level: desired number of scales. The output has level + 1 planes
        scaling_function: the base scaling function.
        """

        sigma_bilateral = copy.copy(self.bilateral) if type(self.bilateral) is list else [self.bilateral, ]*(level+1)
        n_bilateral = len(sigma_bilateral)
        if n_bilateral <= level:
            sigma_bilateral.extend([1, ] * (level - n_bilateral + 1))

        coeffs = np.empty((level + 1,) + arr.shape, dtype=arr.dtype)
        coeffs[0] = arr

        for s in range(level):  # Chained convolution

            atrous_kernel = scaling_function.atrous_kernel(s).astype(arr.dtype)
            if self.bilateral is None:
                convolution(coeffs[s], atrous_kernel, coeffs[s+1])
            else:
                variance = sdev_loc(coeffs[s], atrous_kernel, variance=True)*sigma_bilateral[s]**2
                if self.bilateral_scaling:
                    variance *= s+1
                kernel = scaling_function.kernel.astype(arr.dtype)
                coeffs[s+1] = bilateral_filter(coeffs[s], kernel, variance, s=s, mode='symmetric')

            coeffs[s] -= coeffs[s + 1]

        return coeffs
