# WATROO

Implements the _à trous_ wavelet transform and associated tools: denoising, enhancement, etc.

# Table of contents

[Installation](#installation)\
[_A trous_ transform](#a-trous-transform)\
[Scaling functions](#scaling-functions)\
[WOW! (Wavelets Optimized Whitening)](#wow-wavelets-optimized-whitening))\
[References](#references)

## Installation

Within the active environment

    python setup.py install

or

    pip install .

or if you want to be able to edit & develop (requires reloading the package)

    pip install -e .

## _A trous_ transform

ATrousTransform implements a dyadic 'à-trous' transform

## Scaling functions

### Triangle

### B3 spline

## Examples

### Denoise an image 

    import numpy as np
    from watroo import AtrousTransform, Triangle

    denoise_sigma = [5, 3]
    transform = AtrousTransform(Triangle)
    img = np.random.normal(size=(512, 512))
    coefficients = transform(img, len(denoise_sigma))
    # coefficients.data is an ndarray that contains the coefficients proper
    coefficients.denoise(denoise_sigma)
    # coeffcients accepts numpy operations
    denoised = np.sum(coefficients, axis=0)
    # which is equivalent to
    denoised = coefficients.data.sum(axis=0)

The same result cam be obtained using the *denoise* convenience function

    from watroo import Triangle, denoise

    img = np.random.normal(size=(512, 512))
    denoise_sigma = [5, 3]
    denoised = denoise(img, Triangle, denoise_sigma)

### Extract significant coefficients at a given scale

    # return a ndarray containing the 3-sigma significance of coefficients
    # at scale 2 with hard thresholding
    s = coefficients.significance(3, 2, soft_threshold=False)

### Compute the standard deviation of Gaussian white noise

    # compute 10 scales of the 2D B3spline
    w = B3spline(2)
    w.compute_noise_weights(10)

This returns a 1-D ndarray containing the normalization
used to estimate the significance of coefficients.

## WOW! (Wavelets Optimized Whitening)

    from watroo import wow
    # read in your image here (must be floating point)
    # ...

Standard enhancement

    wow_image, _ = wow(image)

'bilateral' version, slower but better

    wow_image, _ = wow(image, bilateral=1)

Denoised bilateral enhancement (best results)

    wow_image, _ = wow(image, bilateral=1, denoise_coefficients=[5, 2])

## References

J.-L. Starck & F. Murtagh, Handbook of Astronomical Data 
Analysis, Springer-Verlag
