#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from optparse import OptionParser
import numpy as np
import scipy.misc
from contracts import contract
import sys


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


@contract(image='array[MxN]', bits='int,>=0')
def reduce_bits(image, bits=1):
    if bits < 1:
        return image
    return np.right_shift(image, bits)


@contract(image='array[MxN]', filename='str', ftype='str',
          bits='None|int', verbosity='int')
def save_file(image, filename, ftype='raw', bits=None, verbosity=0):
    fname = f'{filename}_{image.shape[0]}x{image.shape[1]}'
    if bits is not None:
        fname += f'_{bits}bit'
    if ftype == 'raw':
        image.tofile(f'{fname}_{image.dtype}.raw')
        if verbosity > 1:
            print(f'  raw file: {fname}_{image.dtype}.raw')
    else:
        scipy.misc.imsave(fname + '.' + ftype, image)
        if verbosity > 1:
            print(f'  {ftype} file: {fname}.{ftype}')


@contract(image='array[MxN]', bit_depths='Iterable',
          filename='str', verbosity='int,>=0')
def generate_reduced(image, bit_depths, filename, verbosity=0):
    for bits in bit_depths:
        result = reduce_bits(image=image, bits=8 - bits)
        save_file(result, filename, 'raw', bits, verbosity=verbosity)
        save_file(np.left_shift(result, 8 - bits),
                  filename,
                  'png',
                  bits,
                  verbosity=verbosity)


@contract(N='int,>0')
def generate_gaussian(N):
    X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    d2 = X**2 + Y**2
    Z = np.exp(-d2 * 2)
    return Z


@contract(N='int,>0')
def generate_rosenbrock(N):
    a = 1
    b = 100
    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, N), np.linspace(-1, 3, N))
    Z = b * (Y - X ** 2) ** 2 + (a - X) ** 2
    return Z


def generate(bit_depths, verbosity=0, N=64, peak=255.):
    funcs = {'gaussian': generate_gaussian,
             'rosenbrock': generate_rosenbrock}
    for name, func in funcs.items():
        if verbosity > 0:
            print(f'Generated example: {name}')
        Z = func(N=N)
        Z = np.asarray(Z) / np.max(Z) * peak
        save_file(Z, name, 'raw', verbosity=verbosity)
        Z = Z.astype(np.uint8)
        generate_reduced(Z, bit_depths, name, verbosity=verbosity)


if __name__ == '__main__':
    usage = '%prog [options] [FILES]'
    parser = OptionParser(usage=usage)
    parser.add_option(
        '-o',
        '--output',
        action='store',
        type='string',
        dest='output',
        help='prefix for the output filename (default: derived from input)')
    parser.add_option(
        '-b',
        '--bit_depth',
        action='append',
        type='int',
        dest='bit_depth',
        help='required bit depth(s), using bitwise right shift'
        '(default: 2,3,4,5,8)')
    parser.add_option(
        '-g',
        '--generate',
        action='store_true',
        dest='generate',
        default=False,
        help='generate algorithmic examples (Gaussian)')
    parser.add_option(
        '-v',
        '--verbose',
        action='count',
        dest='verbose',
        default=0,
        help='verbosity level; can be repeated, e.g. -vvv.'
             '(default: non-verbose mode)')

    (options, args) = parser.parse_args()

    if options.bit_depth is None:
        options.bit_depth = [2, 3, 4, 5, 8]
        if options.verbose > 1:
            depths = ', '.join(map(str, options.bit_depth),)
            print('Bit depths: {depths}')

    if options.generate:
        generate(options.bit_depth, options.verbose, N=64)
        exit(0)

    if len(args) == 0 and options.generate is False:
        parser.print_help()

    for path in args:
        if not os.path.exists(path):
            eprint(f'  The path does not exist! ({path})')
            continue

        image = scipy.misc.imread(path)
        if len(image.shape) > 2:
            eprint(f'  The image must be grayscale! ({path})')
            image = scipy.misc.imread(path, 'L').astype(np.uint)

        if options.verbose > 0:
            print('Input file: {:<30} height: {:>3} width: {:>3}'.format(
                path, image.shape[0], image.shape[1]))

        _, filename = os.path.split(path)
        fileroot, _ = os.path.splitext(filename)
        output = fileroot if options.output is None else options.output
        save_file(image, output, 'raw', verbosity=options.verbose)
        generate_reduced(image,
                         options.bit_depth,
                         output,
                         verbosity=options.verbose)
