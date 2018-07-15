#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
 Image de-quantization using plate bending model
 Copyright: David Völgyes, 2018
 License: LGPL v3, https://www.gnu.org/licenses/lgpl-3.0.txt
"""

# standard python libraries
import os
from optparse import OptionParser

# Additional libraries
import numpy as np     # http://www.numpy.org/
import scipy as sp   # https://scipy.org/
from contracts import contract  # https://andreacensi.github.io/contracts/

# Libraries implemented for the article
from dequantization import collect, eprint, igcd, solve, fill

dtypes = ['bool', 'uint8', 'uint16', 'uint32', 'uint64',
          'int8', 'int16', 'int32', 'int64', 'float32', 'float64']


@contract(filename='filename')
def read_matrix(filename, dtype=np.float):
    print(filename)
    with open(filename, 'rt') as f:
        lines = [l.strip().split() for l in f.readlines()]
    return np.array(lines).astype(dtype)


@collect
@contract(filename='str', shape='tuple|None')
def read_file(filename, shape=None, dtype=None, value=0):
    if filename is None:
        assert(dtype is not None)
        assert(shape is not None)
        return np.zeros(shape=shape, dtype=dtype).fill(value)
    if shape is not None:
        # ~assert(dtype is not None)
        assert(filename is not None)
        return read_file(filename, dtype=dtype, value=value).reshape(shape)
    _, ext = os.path.splitext(filename)
    ext = ext[1:]
    if ext in dtypes:
        return np.fromfile(filename, dtype=np.dtype[ext])
    if ext in ['text', 'txt']:
        data = read_matrix(filename)
        return data[:, -1]
    return sp.misc.imread(filename)


@collect
@contract(filename='str', array='array')
def write_file(filename, array, prec=3):
    _, ext = os.path.splitext(filename)
    ext = ext[1:]
    if ext in dtypes:
        return array.astype(np.dtype(ext)).tofile(filename)
    if ext in ['text', 'txt']:
        return array.tofile(filename, sep='\n', format=f'%.{prec}f')
    if len(array.shape) == 1:
        array = array.reshape(-1, 1)
    print(array.max())
    return sp.misc.imsave(filename, array)


@contract(path='str')
def path_test(path):
    if not os.path.exists(path):
        eprint(f'  The path does not exist! ({path})')
        exit()


if __name__ == '__main__':
    desc = """The aim is to remove false contour from relatively flat regions,
    such as sky or faces, etc.

    The input and output files can have various formats.

    All the custom binary formats use little endian encoding.
    Next to the regular types, the following are also supported extensions:
    .txt: space separated text array, can be 1D or 2D;
    .bool: np.bool binary file in little end;
    .f32: np.float32 binary array;
    .f64: np.float64 binary array;
    .[u]int[8/16/32]: uint8/int8/uint16/etc binary formats;
    """

    parser = OptionParser(usage='%prog [options] IMAGE',
                          version='%prog {__version__}',
                          epilog='',
                          description=desc)

    parser.add_option('-m',
                      '--method',
                      action='store',
                      type='string',
                      dest='method',
                      default='biharmonic',
                      help='Decontouring method: "laplace"'
                      'or "biharmonic"'
                      'or "triharmonic" (default: biharmonic)')

    parser.add_option('--reflect',
                      action='store',
                      type='string',
                      dest='reflect',
                      default='odd',
                      help='Reflection type at the image edges: "odd"'
                      'or "even" (default: "odd")')

    parser.add_option('-i',
                      '--input',
                      action='store',
                      type='string',
                      dest='input',
                      help='Input file.')

    parser.add_option('-M',
                      '--mask',
                      action='store',
                      type='string',
                      dest='mask',
                      help='Mask file. 0/black points are ignored.')

    parser.add_option('-L',
                      '--low',
                      action='store',
                      type='string',
                      dest='low',
                      help='Lower limits. Default: image_mask-0.5')

    parser.add_option('-U',
                      '--upper',
                      action='store',
                      type='string',
                      dest='high',
                      help='Upper limits. Default: image_mask+0.5')

    parser.add_option('-r',
                      '--relative-limits',
                      action='store_true',
                      dest='relative',
                      default=None,
                      help='Limits are relative. (Default: absolute limits)')

    parser.add_option('-o',
                      '--output',
                      action='store',
                      type='string',
                      dest='output',
                      default=None,
                      help='Output file prefix. Default: derived from input.')

    parser.add_option('-s',
                      '--step-size',
                      action='store',
                      type='int',
                      dest='stepsize',
                      default=None,
                      help='Step size in the image file. Default: auto')

    parser.add_option('-p',
                      '--precision',
                      action='store',
                      type='int',
                      dest='precision',
                      default=3,
                      help='Number of decimals in text output. Default: 3')

    parser.add_option(
        '-v',
        '--verbose',
        action='count',
        dest='verbose',
        default=0,
        help='verbosity level; can be repeated, e.g. -vvv.'
             '(default: non-verbose mode)')

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        exit()

    for path in args:
        path_test(path)

        original_image = read_file(path)
        shape = original_image.shape

        if len(original_image.shape) > 2:
            eprint(f'  The image must be grayscale! ({path})')
            continue

        if options.verbose > 0:
            print('Input file: {:<30} height: {:>3} width: {:>3}'.format(
                path, shape[0], shape[1]))

        if options.mask is None:
            mask = np.ones(shape=shape, dtype=np.bool)
            if options.verbose > 0:
                print('No mask file was provided.')
        else:
            mask = read_file(options.mask, shape, value=1)
            mask = mask > (mask.max() / 2.0)

        if options.stepsize is None:
            if original_image.dtype.kind == 'f':
                options.stepsize = 1
            else:
                options.stepsize = igcd(np.unique(original_image).tolist())

        # preprocess: unknown regions: fill with nearest neighbour
        if options.mask is not None:
            image = fill(original_image / options.stepsize, ~mask)
        else:
            image = original_image.copy() / options.stepsize

        if options.low is not None:
            lower = read_file(options.low, shape, np.float32)
        else:
            if options.relative:
                eprint('Conflicting options! You must provide'
                       'limits file if you use relative limits!')
                continue
            lower = image.astype(np.float32) - 0
            if options.verbose > 1:
                eprint('No lower limit file was provided.')

        if options.high is not None:
            upper = read_file(options.high, shape, np.float32)
        else:
            if options.relative:
                eprint('Conflicting options! You must provide'
                       'limits file if you use relative limits!')
                continue
            upper = image.astype(np.float32) + 1.5
            if options.verbose > 1:
                eprint('No upper limit file was provided.')

        # if limits are relative, they must be rebased
        if options.relative:
            lower += image
            upper += image

        if options.output is None:
            _, filename = os.path.split(path)
            fileroot, ext = os.path.splitext(filename)
            output = fileroot
        else:
            output = options.output

        method = options.method

        image = (lower + upper) / 2.0  # initialize from center

        # solve the problem
        solution = solve(image, lower, upper, ~mask,
                         method=method, reflect_type=options.reflect)

        # rescale the result
        solution *= options.stepsize

        # save output
        write_file(f'{output}_{method}_restored.png', solution)
        write_file(f'{output}_{method}_restored.float32', solution)
        if len(solution.shape) == 1:
            write_file(f'{output}_{method}_restored.txt', solution)

        if options.mask is not None:
            solution[~mask] = original_image[~mask]
            write_file(f'{output}_{method}_restored_with_mask.png',
                       solution)
            write_file(f'{output}_{method}_restored_with_mask.float32',
                       solution)
