#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Discrete approximations of Laplace, biharmonic and
triharmonic operators in 1D and 2D.
"""

import numpy as np
from scipy.ndimage.filters import convolve

# https://andreacensi.github.io/contracts/
from contracts import contract, new_contract


@contract(x='int')
def odd_integer_contract(x):
    return x % 2 == 1


new_contract('odd', odd_integer_contract)


@contract(name='str', stencil_size='int,>0,odd|None')
def operator(name, stencil_size=None):
    """ Return 1 or 2 dimensional central finite difference stencil
    for laplace, biharmonic and triharmonic operators.

    >>> print(operator("laplace_1d"))
    [ 1 -2  1]

    >>> print(operator("biharmonic_1d"))
    [ 1 -4  6 -4  1]

    >>> print(operator("triharmonic_1d"))
    [  1  -6  15 -20  15  -6   1]

    >>> print(operator("laplace_2d"))
    [[ 0  1  0]
     [ 1 -4  1]
     [ 0  1  0]]

    >>> print(operator("laplace_2d", stencil_size=5))
    [[ 0  0  0  0  0]
     [ 0  0  1  0  0]
     [ 0  1 -4  1  0]
     [ 0  0  1  0  0]
     [ 0  0  0  0  0]]


    >>> print(operator("biharmonic_2d"))
    [[ 0  0  1  0  0]
     [ 0  2 -8  2  0]
     [ 1 -8 20 -8  1]
     [ 0  2 -8  2  0]
     [ 0  0  1  0  0]]

    >>> print(operator("triharmonic_2d"))
    [[   0    0    0    1    0    0    0]
     [   0    0    3  -12    3    0    0]
     [   0    3  -24   57  -24    3    0]
     [   1  -12   57 -112   57  -12    1]
     [   0    3  -24   57  -24    3    0]
     [   0    0    3  -12    3    0    0]
     [   0    0    0    1    0    0    0]]

    """

    # 1D
    if name == 'laplace_1d':
        op = np.asarray([1, -2, 1])

    if name == 'biharmonic_1d':
        laplace_1d = operator(name='laplace_1d', stencil_size=5)
        op = convolve(laplace_1d, laplace_1d)

    if name == 'triharmonic_1d':
        laplace_1d = operator(name='laplace_1d', stencil_size=7)
        biharmonic_1d = operator(name='biharmonic_1d', stencil_size=7)
        op = convolve(laplace_1d, biharmonic_1d)

    # 2D
    if name == 'laplace_2d':
        laplace_1d = np.asarray([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
        op = laplace_1d.T + laplace_1d

    if name == 'biharmonic_2d':
        laplace_2d = operator(name='laplace_2d', stencil_size=5)
        op = convolve(laplace_2d, laplace_2d)

    if name == 'triharmonic_2d':
        laplace_2d = operator(name='laplace_2d', stencil_size=7)
        biharmonic_2d = operator(name='biharmonic_2d', stencil_size=7)
        op = convolve(laplace_2d, biharmonic_2d)

    # resizing operator
    if stencil_size is not None:
        assert(stencil_size >= op.shape[0])
        p = (stencil_size - op.shape[0]) // 2
        op = np.pad(op, p, mode='constant')
    return op


if __name__ == '__main__':
    import doctest
    doctest.testmod()
