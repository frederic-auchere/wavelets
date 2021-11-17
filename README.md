# WATROO

Implements several of the concepts described in:

J.-L. Starck & F. Murtagh, Handbook of Astronomical Data 
Analysis, Springer-Verlag

## Scaling functions

### Triangle

### B3 spline

## 'A trous' transform

## Utils

## Examples

    from watroo import AtrousTransform, Triangle

    denoise_sigma = [5, 3]
    transform = AtrousTransform(Triangle)
    coefficients = transform(img, len(denoise_sigma))
    # coefficients.data is an ndarray that contains the cofeecients proper
    coefficients.denoise(denoise_sigma)
    # coeffcients accepts numpy operations
    denoised = np.sum(coefficients, axis=0)
    # which is equivalent to
    denoised = coefficients.data.sum(axis=0)

## Installation

From the active environment

    python setup.py install

or

    pip install .

or if you want to be able to edit & develop (requires reloading the package)

    pip install -e .