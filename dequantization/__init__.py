#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .algorithm import solve, fill
from .tools import eprint, igcd, collect

__version__ = '0.9'
__author__ = 'David Völgyes'
__license__ = 'LGPL v3'
__maintainer__ = 'David Völgyes'
__email__ = 'david.volgyes@ieee.org'
__status__ = 'Testing'
__title__ = 'dequantization'

__summary__ = (' Image de-quantization using plate bending model')
__uri__ = 'https://github.com/dvolgyes/dequantization'
__license__ = 'AGPL v3'
__doi__ = 'unknown'
__description__ = """
Image de-quantization using plate bending model
"""

__bibtex__ = (
    """Unpublished at this moment.
}"""
)
__reference__ = (
    """Unpublished."""
    + __version__
    + """).
Zenodo. https://doi.org/"""
    + __doi__
)

solve = solve
fill = fill
eprint = eprint
igcd = igcd
collect = collect
