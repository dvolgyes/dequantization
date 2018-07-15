#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contracts import contract
import numpy as np
import scipy as sp
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import distance_transform_edt as dist_transform
import importlib
from skimage.util import view_as_windows
from .tools import collect, eprint, downscalable, product, rangeN
from .polyharmonic import operator


if importlib.find_loader('numba') is None:
    def jit(x):
        return x
    eprint("Numba isn't working! Try to fix it for faster execution speed.")
    eprint('As a workaround, the code will'
           ' use a pure-python fallback but it is slow.')
else:
    from numba import jit


@collect
@jit
def fill(img, mask):
    # fill with nearest neighbor value
    result = img.copy()
    if np.all(mask):
        return result
    result = result.ravel()
    _, tmp = dist_transform(mask, return_indices=True)
    indices = tmp.reshape(2, -1)
    nonzero = np.nonzero(mask.ravel())
    for idx in nonzero:
        x, y = indices[:, idx]
        result[idx] = img[x, y]
    return result.reshape(img.shape)


@jit
def downscale_all(image, low, up, mask):
    shape = image.shape[0] // 2, image.shape[1] // 2
    dimg = np.zeros(shape=shape, dtype=np.float)
    dlow = np.ones(shape=shape, dtype=low.dtype) * low.max()
    dup = np.ones(shape=shape, dtype=up.dtype) * up.min()
    dmask = np.zeros(shape=shape, dtype=mask.dtype)
    for i in range(dimg.shape[0]):
        for j in range(dimg.shape[1]):
            for k in range(2):
                for l in range(2):
                    a, b = i * 2 + k, j * 2 + l
                    dlow[i, j] = min(dlow[i, j], low[a, b])
                    dup[i, j] = max(dup[i, j], up[a, b])
                    dmask[i, j] = dmask[i, j] or mask[a, b]
                    dimg[i, j] += image[a, b] * 0.25
    return dimg, dlow, dup, dmask


@contract(array='array', factor='float,>0')
def upscale(array, factor=2.0):
    return zoom(array, factor, order=3, mode='reflect')


@collect
@contract(data='Iterable', x='Iterable', y='Iterable', shape='tuple')
def to_csr_matrix(data, x, y, dtype, shape):
    return sp.sparse.coo_matrix((np.asarray(data),
                                 (np.asarray(x),
                                  np.asarray(y))),
                                dtype=dtype,
                                shape=shape).tocsr()


@collect
def generate_sparse_matrix(shape, op):
    count = product(shape)
    indices = np.arange(count).reshape(shape)
    view = view_as_windows(indices, op.shape)
    window_shape = view.shape[0:len(shape)]
    X, Y, W = [], [], []
    cnt = op.size
    center = cnt // 2
    opr = op.ravel()

    for idx in rangeN(window_shape):
        v = view[idx].ravel()
        for p in range(cnt):
            if opr[p] != 0:
                X.append(v[center])
                Y.append(v[p])
                W.append(opr[p])
    inside = to_csr_matrix(W, X, Y, dtype=np.float, shape=(count, count))
    return inside


@collect
def reorder_operator(op):
    center = op.size // 2
    opr = op.copy().flatten()
    scale = -opr[center]
    opr[center] = 0
    opr = opr / float(scale)
    return opr.reshape(op.shape)


def mix(a, b, alpha=0.5):
    return alpha * a + (1 - alpha) * b


@collect
@contract(method='str', dims='int,>=1,<=2')
def generate_operators(method='laplace', dims=2):
    op1 = operator(f'{method}_{dims}d')
    return op1


@collect
def solve(image,
          l_limit,
          u_limit,
          mask,
          center=None,
          method='laplace',
          reflect_type='odd',
          grid_levels=None):
    # This method is a bit complicated, but the main concept is:
    # First, downsample the problem with factor of 2.
    # Repeat this 'grid_levels' times, e.g. 5
    # Solve the problem on the small grid
    # Upscale the solution, solve the larger problem
    #
    # At a given grid: laplace phi = 0 -> phi = average_of_neighbors
    # Biharmonic equation gets a similar reordering
    # Both of them get a relaxation parameter,

    dims = len(image.shape)

    op = reorder_operator(generate_operators(method=method, dims=dims))
    opsize = op.shape[0] // 2

    if grid_levels is None:
        grid_levels = int(np.log2(image.shape[0]))

    if dims == 2 and downscalable(image) and grid_levels > 0:
        dimg, dlow, dup, dmask = downscale_all(image, l_limit, u_limit, mask)

        img = solve(dimg, dlow, dup, dmask,
                    method=method,
                    reflect_type=reflect_type,
                    grid_levels=grid_levels - 1)
        img = img.reshape(dimg.shape)
        image = np.clip(upscale(img), l_limit, u_limit)

    eprint(f'  current image size: {image.shape} ')

    low = np.pad(l_limit, opsize, mode='symmetric', reflect_type=reflect_type)
    up = np.pad(u_limit, opsize, mode='symmetric', reflect_type=reflect_type)
    img = np.pad(image, opsize, mode='symmetric', reflect_type=reflect_type)

    img = np.clip(img, low, up)

    extended_shape = img.shape
    m_inside = generate_sparse_matrix(extended_shape, op)

    img, low, up = img.ravel(), low.ravel(), up.ravel()

    a = opsize
    b = -opsize
    c = opsize
    d = -opsize

    if center is None:
        center = (low+up) / 2.0

    relax = 0.2

    for _ in range(4):
        #  inner loop, proportional with number of pixels, could be merged
        #  with the outer loop
        for _ in range(10 * image.shape[0]):
            delta = m_inside @ img
            diff = np.clip(delta, low, up)

            img = mix(diff, img, relax).reshape(extended_shape)

            #  periodic boundary conditions:
            if len(img.shape) == 2:
                img = np.pad(img[a:b, c:d],
                             opsize,
                             mode='symmetric',
                             reflect_type=reflect_type).ravel()
            else:
                img = np.pad(img[a:b],
                             opsize,
                             mode='symmetric',
                             reflect_type=reflect_type).ravel()

    img = img.reshape(extended_shape)
    #  keep only the center of the image (cut the extrapolated boundaries).
    if len(img.shape) == 2:
        img = img[a:b, c:d]
    else:
        img = img[a:b]

    return img


if __name__ == '__main__':
    import doctest
    doctest.testmod()
