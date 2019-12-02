#!/usr/bin/env python

from transforms.tests import *

# ------------------------------
# transforms and inverse transforms
# ------------------------------

# test_fourier(Cosine())  # fails!
test_fourier(Gaussian())
test_fourier(Gaussian2())
test_fourier(Cauchy())

test_fourier_distributions_univariate()
test_fourier_distributions_bivariate()
test_hankel()

test_laplace()
