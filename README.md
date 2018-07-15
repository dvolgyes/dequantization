Image de-quantization using plate bending model
===============================================

Travis:[![Build Status](https://travis-ci.org/dvolgyes/dequantization.svg?branch=master)](https://travis-ci.org/dvolgyes/dequantization)
Coveralls:[![Coverage Status](https://coveralls.io/repos/github/dvolgyes/dequantization/badge.svg?branch=master)](https://coveralls.io/github/dvolgyes/dequantization?branch=master)
Codecov:[![codecov](https://codecov.io/gh/dvolgyes/dequantization/branch/master/graph/badge.svg)](https://codecov.io/gh/dvolgyes/dequantization)

The code is implemented in Python3, and requires Python3.5 or newer.


Installation
------------

The code could be installed directly from Github:
```
pip install git+https://github.com/dvolgyes/dequantization
```
(Tested on Ubuntu 18.04.)

The code and the paper about it  are currently under review.

Usage
-----

Aftern installation, the command line options can be queried:
```
dequantization.py -h
```

The code could be used as a library too, but it is tricky.
(You have to load & pre-process the data, set up the algorithm,
save the results, etc.)
