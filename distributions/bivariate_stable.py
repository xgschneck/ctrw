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
from scipy.integrate import quad
from scipy.special import gamma
from scipy.special import j0 as bessel

from .base import BivariateDistribution
from .univariate_stable import Stable
from .bivariate_normal import Normal2D


class _Stable2D_base(BivariateDistribution):

    def __repr__(self):

        return "Stable2D(scale={}, alpha={})".format(self.scale, self.alpha)

    def __str__(self):

        return "Stable$_{{2}}(\\alpha={:0.02f}, \gamma={:0.02f})$".format(self.alpha, self.scale)

    def covariance_matrix(self):

        raise NotImplementedError

    def cf(self, u, v):

        u2 = u * u + v * v
        return numpy.exp(- self.scale ** self.alpha * u2 ** (0.5 * self.alpha))

    def cf_distance(self, k):

        return numpy.exp(- (self.scale * k) ** self.alpha)

    def cf_distance_asymptotic(self, k):

        return 1.0 - (self.scale * k) ** self.alpha

    def asymptotic_constant(self):
        """A factor also encountered in Riesz potentials."""

        c_alpha = numpy.pi / 2.0 ** self.alpha * gamma(- 0.5 * self.alpha) / gamma(1.0 + 0.5 * self.alpha)
        return - 1.0 / c_alpha

    def pdf_asymptotic_constants(self):

        c = self.scale ** self.alpha * self.asymptotic_constant()
        e = - (2.0 + self.alpha)
        return c, e

    def pdf_distance_asymptotic(self, r):

        c, e = self.pdf_asymptotic_constants()
        return c * r ** e


class Stable2D(_Stable2D_base):
    """Bivariate isotropic stable distribution.

    Implementation according to [D8].
    """

    def __init__(self, scale=1.0, alpha=1.0):

        if alpha <= 0.0 or 2.0 <= alpha:
            raise ValueError("alpha out of bounds")

        self.scale = scale
        self.alpha = alpha

        self.univariate_alpha = 0.5 * alpha
        self.univariate_scale = 2.0 * scale * scale * numpy.cos(numpy.pi * alpha * 0.25) ** (2.0 / alpha)

        self.__distribution_stable_univariate = Stable(alpha=self.univariate_alpha, beta=1.0, scale=self.univariate_scale, loc=0.0)
        self.__distribution_normal = Normal2D(loc=(0.0, 0.0), scale=1.0)

        self._plot_support = [-20.0, 20.0]

        # NOTE returns object, is not compatible with multiprocessing pool
        self._pdf_distance_vectorized = numpy.frompyfunc(self._pdf_distance, 1, 1)
        self._pdf_radius_vectorized = numpy.frompyfunc(self._pdf_radius, 1, 1)

    def sample(self, size=None):

        A = self.__distribution_stable_univariate.sample(size=size)
        Z = self.__distribution_normal.sample(size=size)

        return numpy.multiply(Z.T, numpy.sqrt(A)).T

    def sample_square_radius(self, size=None):

        raise NotImplementedError

        # TODO R*R = A * Chi2(df=2)

        A = self.__distribution_stable_univariate.sample(size=size)
        U = "Chi2(df=2)"

        return A * U

    def pdf_radius(self, r):
        """PDF of bivariate RV for absolute argument.

        Different evaluation techniques (see [D8]) are available
        in the methods `_pdf_radius_*`.
        """

        if numpy.isscalar(r):
            return self._pdf_radius(r)
        else:
            return self._pdf_radius_vectorized(r).astype(float)

    def _pdf_radius(self, r):

        return self._pdf_radius_nolan1(r)

    def _pdf_radius_nolan1(self, r):

        # default implementation in [D8]
        # is rather slow

        def integrand(t, r):
            # NOTE: factor 0.5 removed from integrand and delayed
            return self.__distribution_stable_univariate.pdf(r * r / t) * numpy.exp(- 0.5 * t) / t

        # NOTE: factor 2.0 is missing because eliminated from integrand factor 0.5
        return r * quad(integrand, 0.0, numpy.inf, args=(r,))[0]

    def _pdf_radius_nolan2(self, r):

        # is a transformation of variables of the first form
        # is slowest

        def integrand(s, r):
            return self.__distribution_stable_univariate.pdf(s) * numpy.exp(- 0.5 * r * r / s) / s

        return r * quad(integrand, 0.0, numpy.inf, args=(r,))[0]

    def _pdf_radius_zolotarev(self, r):

        # uses the Bessel function instead of the stable pdf in the integrand
        # has issues with small alpha!
        # is fastest

        def integrand(t, r, scale, alpha):
            # NOTE: factor r was eliminated from integrand
            return t * bessel(r * t) * numpy.exp(- (scale * t) ** alpha)

        return r * quad(integrand, 0.0, numpy.inf, args=(r, self.scale, self.alpha))[0]

    def cf_radius(self, k):

        raise NotImplementedError

    def pdf(self, x, y):

        return self.pdf_distance(numpy.sqrt(x * x + y * y))

    def pdf_distance(self, r):
        """PDF for the distance or amplitude density.

        Different evaluation strategies are discussed in [D8].
        See `_pdf_distance_*`.
        """

        if numpy.isscalar(r):
            return self._pdf_distance(r)
        else:
            return self._pdf_distance_vectorized(r).astype(float)

    def _pdf_distance(self, r):

        return self._pdf_distance_nolan4(r)

    def _pdf_distance_nolan4(self, r):

        # standard evaluation technique in [D8]

        if r == 0.0:
            return gamma(2.0 / self.alpha) / (2.0 * self.alpha * numpy.pi * self.scale * self.scale)
        else:
            return 0.5 / numpy.pi / r * self._pdf_radius(r)

    def _pdf_distance_nolan5(self, r):

        # an alternative integral representation in [D8]

        raise NotImplementedError("not tested")

        def integrand2(t, r, alpha):
            # basically the function g
            def integrand(k, t, alpha):
                return numpy.cos(t * k) * k * numpy.exp(- k ** alpha)
            return quad(integrand, 0.0, numpy.inf, args=(t * r, alpha))[0] / numpy.sqrt(1.0 - t * t)
        return quad(integrand2, 0.0, 1.0, args=(r, self.alpha))[0] / numpy.pi / numpy.pi
