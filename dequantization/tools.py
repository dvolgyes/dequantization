#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Various tools. They are not essential for the main algorithm
but they make programming more comfortable and testing more
reliable.
"""

import os
import sys
from functools import wraps
from contextlib import contextmanager
from gc import collect as garbage_collector
from contracts import contract, new_contract
import numpy as np
import time
from functools import reduce
import operator
from math import gcd as greatest_common_divisor
import itertools


@contract(limits='Iterable')
def rangeN(limits):
    return itertools.product(*(range(N) for N in limits))


@contract(it='Iterable')
def igcd(it):
    return reduce(greatest_common_divisor, it)


@contract(it='Iterable')
def product(it):
    return reduce(operator.mul, it)


@contract(array='array', factor='int,>1')
def downscalable(array, factor=2):
    for d in array.shape:
        if not (operator.mod(d, factor) == 0) or d < 31:
            return False
    return True


@contract(arr='array')
def all_positive(arr):
    return np.all(arr > 0)


@contract(arr='array')
def all_non_negative(arr):
    return np.all(arr >= 0)


@contract(x='int')
def odd_integer_contract(x):
    return x % 2 == 1


@contract(x='int')
def even_integer_contract(x):
    return x % 2 == 0


#           new contracts
new_contract('path', os.path.exists)
new_contract('dir', os.path.isdir)
new_contract('filename', os.path.isfile)
new_contract('link', os.path.islink)
new_contract('dtype', lambda x: isinstance(x, np.dtype))
new_contract('positive', all_positive)
new_contract('non_negative', all_non_negative)
new_contract('downscalable', downscalable)
# ~new_contract('odd', odd_integer_contract)
# ~new_contract('even', even_integer_contract)


@contract(it='Iterable')
def remove_none(it):
    return [x for x in it if x is not None]


def min_(*args):
    """
    Return minimum value from an iterable.
    None is ignored.
    """
    return min(remove_none(args))


def max_(*args):
    """
    Return maximum value from an iterable.
    None is ignored.
    """
    return max(remove_none(args))

#           utility functions


def eprint(*args, **kwargs):
    """
    Print to the standard error.
    Arguments are the same as for the print function.
    """
    print(*args, file=sys.stderr, **kwargs)


@contract(func='Callable')
def collect(func):
    """
    A decorator which performs garbage collection
    before and after the execution of the function.

    >>> if True:
    ...     @collect
    ...     def test_function(x):
    ...         return x
    ...     test_function(123)
    123
    """

    @wraps(func)
    def func_wrapper(*args, **kwargs):
        garbage_collector()
        result = func(*args, **kwargs)
        garbage_collector()
        return result
    return func_wrapper


@contract(func='Callable')
def error(func):
    """
    This is a decorator which throws an exception.
    It is meant to be used for debug purposes.
    """
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        def exc(*args, **kwargs):
            raise Exception()
        return exc(*args, **kwargs)
    return func_wrapper


@contextmanager
def timeit():
    start = time.time()
    yield
    end = time.time()
    print(' Execution time is: {:.3f} [sec]'.format(end - start))


@contextmanager
@contract(f='file|filename')
def delete_file_ctx(f):
    """
    Delete a given file after the context.
    For temporary files, you should rather consider to use tmpfile_ctx.

    >>> if True:
    ...     import tempfile
    ...     fn = tempfile.mkstemp()[1]
    ...     with delete_file_ctx(fn):
    ...         print(os.path.exists(fn))
    ...     print(os.path.exists(fn))
    True
    False

    >>> if True:
    ...     import tempfile
    ...     fn = tempfile.mkstemp()[1]
    ...     f = open(fn)
    ...     with delete_file_ctx(f):
    ...         print(os.path.exists(f.name))
    ...     print(os.path.exists(f.name))
    True
    False

    >>> if True:
    ...     import tempfile
    ...     fn = tempfile.mkstemp()[1]
    ...     f = open(fn)
    ...     with delete_file_ctx(f):
    ...         f.close()
    ...         print(os.path.exists(f.name))
    ...     print(os.path.exists(f.name))
    True
    False
    """

    yield
    if isinstance(f, str):
        os.remove(f)
    else:
        name = f.name
        if not f.closed:
            f.close()
        os.remove(name)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
