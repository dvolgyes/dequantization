#!/usr/bin/make

default:
	@echo "There is nothing to do."

ci-test: example
	python3 -m coverage run -a --source . dequantization.py 
	python3 -m coverage run -a --source . dequantization.py -h
	python3 -m coverage run -a --source . dequantization/tools.py
	python3 -m coverage run -a --source . dequantization/polyharmonic.py
	@python3 -m coverage run -a --source . tools/metrics.py 
	@python3 -m coverage run -a --source . tools/metrics.py -h
	@python3 -m coverage run -a --source . tools/data.py
	@python3 -m coverage run -a --source . tools/data.py -h
	@python3  -m coverage run -a --source . dequantization.py  -v test/gaussian_64x64_2bit.png -o gauss2bit -m laplace
	@python3  -m coverage run -a --source . dequantization.py  -v test/gaussian_64x64_2bit.png -o gauss2bit -m biharmonic
	@python3 -m coverage run -a --source . tools/metrics.py -r test/gaussian_64x64_8bit.png gauss2bit_biharmonic_restored.png
	@echo "Testing is finished."

example:
	@make -C test
