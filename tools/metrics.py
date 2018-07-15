#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" Image quality metrics based on scikit-image.

Releavant publications:
@DOI 10.7717/peerj.453
@DOI 10.1109/TIP.2003.819861
@DOI 10.1007/s10043-009-0119-z
"""


from optparse import OptionParser
import imageio
import skimage.measure
from contracts import contract, new_contract
import os
import sys


new_contract('filename', os.path.isfile)


@contract(filename='filename')
def read_file(filename):
    if filename.endswith('png') or filename.endswith('jpg'):
        return imageio.imread(filename)


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option('-r', '--reference',
                      action='store',
                      type='string',
                      dest='reference',
                      help='Path to the reference file.')

    parser.add_option('-m', '--metrics',
                      action='append',
                      type='string',
                      dest='metrics',
                      help='Metrics: PSNR, SSIM, MSE (default: all)')

    parser.add_option('-s', '--silent',
                      action='store_true',
                      dest='silent',
                      default=False,
                      help='Silent: metrics names are suppressed'
                      ' to improve parsability.')

    parser.add_option('-w', '--window_size',
                      dest='window',
                      default=7,
                      help='Window size for SSIM, must be odd.'
                      '(Default: 7)')

    (options, args) = parser.parse_args()

    metrics_list = ['PSNR', 'SSIM', 'MSE', 'NRMSE']

    if options.reference is None:
        print('Reference file must be provided!', file=sys.stderr)
        exit()

    reference = read_file(options.reference)

    if options.metrics is None:
        options.metrics = metrics_list

    for path in args:
        input_image = read_file(path)
        if options.silent:
            result = ''
        else:
            result = f'{path:<40}'
        for metric in options.metrics:
            metric = metric.upper()
            if metric == 'SSIM':
                value = skimage.measure.compare_ssim(reference, input_image,
                                                     win_size=options.window)
            if metric == 'MSE':
                value = skimage.measure.compare_mse(reference, input_image)
            if metric == 'NRMSE':
                value = skimage.measure.compare_nrmse(reference, input_image)
            if metric == 'PSNR':
                value = skimage.measure.compare_psnr(reference, input_image)
            if options.silent:
                result += '{:{w}.{p}f}  '.format(value, w=5, p=4)
            else:
                result += '{:>6}: {:{w}.{p}f} '.format(metric, value, w=5, p=4)
        print(result)
