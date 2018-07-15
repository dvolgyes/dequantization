#!/usr/bin/make

data:
	@python3 -m coverage run -a --source ..  ../tools/data.py -b 2 -b 8 -g 
	@python3 -m coverage run -a --source ..  ../tools/data.py -v gaussian_64x64_8bit.png -b 3 -o test

