# WATROO

Implements several of the concepts described in:

J.-L. Starck & F. Murtagh, Handbook of Astronomical Data 
Analysis, Springer-Verlag

## Scaling functions

### Triangle

### B3 spline

## 'A trous' transform

ATrousTransform implements a dyadic 'à-trous' transform

## Utils

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


## Installation

From the active environment

    python setup.py install

or

    pip install .

or if you want to be able to edit & develop (requires reloading the package)

    pip install -e .
