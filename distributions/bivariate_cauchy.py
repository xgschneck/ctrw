# This file is part of ctrw.
#
# Copyright (C) 2019 Günter Schneckenreither
#
# ctrw is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ctrw is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ctrw.  If not, see <http://www.gnu.org/licenses/>.


import numpy
from scipy.special import gamma

from .base import BivariateDistribution
from .univariate import Uniform
from .univariate_paretian import SquarePareto
from .bivariate_normal import Normal2D


class _Cauchy2D_base(BivariateDistribution):

    alpha = 1.0

    def __init__(self):

        self._plot_support = [-10.0, 10.0]

    def __repr__(self):

        return "Cauchy2D(scale={})".format(self.scale)

    def __str__(self):

        return "Cauchy$_{{2}}(\gamma={:0.02f})$".format(self.scale)

    def covariance_matrix(self):

        raise NotImplementedError

    def cf(self, u, v):

        u2 = numpy.sqrt(u * u + v * v)
        return numpy.exp(- self.scale * u2)

    def cf_distance(self, k):

        return numpy.exp(- self.scale * k)

    def cf_distance_asymptotic(self, k):

        return 1.0 - self.scale * k

    def pdf_asymptotic_constants(self):

        c = 1 / 2.0 / numpy.pi * self.alpha * (2 * self.scale) ** self.alpha
        c *= gamma(1 + 0.5 * self.alpha) / gamma(1 - 0.5 * self.alpha)
        e = - (2 + self.alpha)
        return c, e

    def pdf_distance_asymptotic(self, r):

        c, e = self.pdf_asymptotic_constants()
        return c * r ** e


class Cauchy2D_mv(_Cauchy2D_base):
    """Isotropic bivariate Cauchy distribution.

    Multivariate implementation SciPy.
    """

    def __init__(self, scale=1.0):

        super().__init__()

        self.scale = scale  # = gamma

        self.__distribution_mvnormal = Normal2D(scale=scale)
        from scipy.stats import chi2
        self.__distribution_chisquare = chi2(df=1)  # TODO

    def sample(self, size=None):

        U_sqrt = numpy.sqrt(self.__distribution_chisquare.rvs(size=size))
        return self.__distribution_mvnormal.sample(size=size) / numpy.stack((U_sqrt, U_sqrt)).T

    def pdf(self, x, y):

        return self.scale / (2.0 * numpy.pi) * (x * x + y * y + self.scale * self.scale) ** -1.5

    def pdf_distance(self, r):

        return self.pdf(r, 0.0)


class Cauchy2D_radial(_Cauchy2D_base):
    """Isotropic bivariate Cauchy distribution.

    Separate implementation of radius and angle.
    """

    def __init__(self, scale=1.0):

        super().__init__()

        self.scale = scale

        self.__distribution_radius = SquarePareto(scale)
        self.__distribution_angle = Uniform(left=0.0, right=2.0 * numpy.pi)

    def sample(self, size=None):

        r = self.__distribution_radius.sample(size=size)
        phi = self.__distribution_angle.sample(size=size)

        return numpy.stack((numpy.cos(phi) * r, numpy.sin(phi) * r)).T

    def pdf(self, x, y):

        r = numpy.sqrt(x * x + y * y)
        return self.pdf_distance(r)

    def pdf_distance(self, r):

        if numpy.isscalar(r):

            if numpy.isclose(r, 0.0):
                return 1.0 / (2.0 * numpy.pi * self.scale * self.scale)
            else:
                return self.__distribution_radius.pdf(r) / r / (2.0 * numpy.pi)

        else:

            idx = numpy.isclose(r, 0.0)
            out = numpy.zeros(r.shape)
            out[idx] = 1.0 / (2.0 * numpy.pi * self.scale * self.scale)
            out[~idx] = self.__distribution_radius.pdf(r[~idx]) / r[~idx] / (2.0 * numpy.pi)
            return out

    def pdf_radius(self, r):

        return self.__distribution_radius.pdf(r)

    def cf_radius(self, r):

        return self.__distribution_radius.cf(r)


class Cauchy2D_direct(_Cauchy2D_base):
    """Isotropic bivariate Cauchy distribution.

    Direct implementation.
    """

    def __init__(self, scale=1.0):

        super().__init__()

        self.scale = scale

        self.__distribution_radius = SquarePareto(scale)

    def sample(self, size=None):

        r = self.__distribution_radius.sample(size=size)
        phi = numpy.random.random(size) * 2.0 * numpy.pi

        return numpy.stack((numpy.cos(phi) * r, numpy.sin(phi) * r)).T

    def pdf(self, x, y):

        return self.scale / (2.0 * numpy.pi) * (x * x + y * y + self.scale * self.scale) ** -1.5

    def pdf_distance(self, r):

        return self.pdf(r, 0.0)


class Cauchy2D(BivariateDistribution):
    """Isotropic bivariate Cauchy distribution.

    Prefers the direct implementation.
    """

    alpha = 1.0

    def __new__(cls, direct=0, **kwargs):

        if direct == 0:
            return Cauchy2D_direct(**kwargs)
        elif direct == 1:
            return Cauchy2D_radial(**kwargs)
        else:
            return Cauchy2D_mv(**kwargs)
