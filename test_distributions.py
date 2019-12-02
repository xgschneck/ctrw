#!/usr/bin/env python

from distributions.tests import *

# ------------------------------
# standard visual tests
# ------------------------------

# distributions.Poisson.test(scale=1.7)
# distributions.Geometric.test(p=0.6)
# distributions.GeometricTruncated.test(p=0.6, n=9)
# distributions.Binomial.test(p=0.6, n=9)

# distributions.Normal.test(scale=4.1)
# distributions.LogNormal.test(scale=1.7)
# distributions.ExponentialPower.test(beta=3.5, scale=2.1)
# distributions.Exponential.test(scale=5.0)

# distributions.Laplace.test(scale=2.4)
# distributions.SkewLaplace.test(scale=7.0, loc=0.0, skewness1=-1.0)

# distributions.Lomax.test(beta=3.2, scale=1.4, logplot=True)
# distributions.Pareto.test(scale=7.4)
# distributions.SquarePareto.test(scale=4.6)
# distributions.MittagLeffler.test(scale=0.7, beta=0.67, logplot=True)

# distributions.Stable.test(alpha=1.4, beta=0.8, scale=0.2)

distributions.Normal2D.test(scale=2.9)
distributions.Cauchy2D.test(scale=2.5)
distributions.Stable2D.test(scale=2.1, alpha=1.7)


# ------------------------------
# additional tests
# * asymptotic behavior
# * transforms
# * transitions
# * identities
# * implementations
# * performance
# ------------------------------

# lomax_powerlaw()
# mittagleffler_function_implementation()
# mittagleffler_powerlaw()
# mittagleffler_transition()
# mittagleffler_laplace_transform()

# stable_discontinuity()
# stable_implementations()
# stable_implementations_performance()
# stable_powerlaw()

# bivariate_stable_radial()
# bivariate_stable_distance()
# bivariate_stable_powerlaw()
# bivariate_stable_density_implementations_performance()
# bivariate_stable_density_implementations()
# bivariate_stable_fourier_transform_density()

# bivariate_normal()
# bivariate_cauchy()
# bivariate_scaling()

# geometric_stability("laplace")
# geometric_stability("mittagleffler")
