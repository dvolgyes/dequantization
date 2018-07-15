#!/usr/bin/make

default:
	@echo "There is nothing to do."

ci-test: example
	@python3 -m coverage run -a --source . tools/metrics.py
	@python3 -m coverage run -a --source . tools/metrics.py -h
	@python3 -m coverage run -a --source . tools/data.py
	@python3 -m coverage run -a --source . tools/data.py -h
	@python3 -m coverage run -a --source . tools/data.py -b 2 -b 8 -g
	@python3 -m coverage run -a --source . tools/data.py -v gaussian_64x64_8bit.png -b 3 -o test

	@python3 -m coverage run -a --source . dequantization_cli.py
	@python3 -m coverage run -a --source . dequantization_cli.py -h
	@python3 -m coverage run -a --source . dequantization/tools.py
	@python3 -m coverage run -a --source . dequantization/polyharmonic.py
	@python3 -m coverage run -a --source . dequantization_cli.py  -v gaussian_64x64_2bit.png -o gauss2bit -m laplace
	@python3 -m coverage run -a --source . dequantization_cli.py  -v gaussian_64x64_2bit.png -o gauss2bit -m biharmonic
	@python3 -m coverage run -a --source . tools/metrics.py -r gaussian_64x64_8bit.png gauss2bit_biharmonic_restored.png
	@echo "Testing is finished."
