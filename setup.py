#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Documentation
#  https://packaging.python.org/tutorials/packaging-projects/
#
# To install: "python setup.py install"

from setuptools import setup, find_packages

NAME = 'watroo'
DESCRIPTION = 'A trous wavelets transform package',
URL = ''
EMAIL = 'frederic.auchere@universite-paris-saclay.fr'
AUTHOR = 'Frédéric Auchère'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.1'

REQUIRED = [
    'scipy',
    'numpy',
    'numexpr',
    'opencv-python'
]

# If you do change the License, remember to change the Trove Classifier for that!

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=REQUIRED,
    license='LGPL-v3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    zip_safe=False,
    ext_modules=None,
)
