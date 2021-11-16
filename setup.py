#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# To install: "python setup.py install"

import io
import os
from numpy import get_include as np_get_include
from setuptools import find_packages, setup, Command
from setuptools.extension import Extension

# Package meta-data.
NAME = 'ATrous'
DESCRIPTION = 'A trous wavelets package'
URL = ''
EMAIL = 'frederic.auchere@universite-paris-saclay.fr'
AUTHOR = 'Frédéric Auchère'
REQUIRES_PYTHON = '>=3.9.0'
VERSION = '0.0.1'

# What packages are required for this module to be executed?
REQUIRED = [
    'scipy',     # scientific python
    'numpy',     # basic numerics
    'opencv'     # fast 2D convolution
]

# What packages are optional?
EXTRAS = {
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Set up extensions - trivial with just helpers, but 
# useful later if we have to link in C libraries etc.
extensions = [
    Extension()
    ]

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where='watrous'),
    # If your package is a single module, use this instead of 'packages':
    #py_modules=['atrous'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    requires=REQUIRED,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='LGPL-v3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    zip_safe = False,
    ext_modules = None,
)
