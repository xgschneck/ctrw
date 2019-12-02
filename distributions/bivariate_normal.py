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
from .univariate_exponential import Rayleigh


class _Normal2D_base(BivariateDistribution):

    alpha = 2.0

    def __init__(self):

        self._plot_support = [-7.0, 7.0]

    def __repr__(self):

        return "Normal2D(scale={}, loc={})".format(self.scale, self.loc)

    def __str__(self):

        # do not show location in fancy output
        # return "Normal$_{{2}}(\sigma={:0.02f}, \mu=({:0.02f}, {:0.02f}))$".format(self.scale, *self.loc)
        return "Normal$_{{2}}(\sigma={:0.02f})$".format(self.scale)

    def covariance_matrix(self):

        return numpy.eye(2) * self.scale * self.scale

    def cf(self, u, v):

        u2 = u * u + v * v
        return numpy.exp(- 0.5 * self.scale * self.scale * u2)

    def cf_distance(self, k):

        return numpy.exp(- 0.5 * self.scale * self.scale * k * k)

    def pdf_asymptotic_constants(self):

        # TODO simplify calculation for alpha = 2 !

        c = 1 / 2.0 / numpy.pi * self.alpha * (2 * self.scale) ** self.alpha
        c *= gamma(1 + 0.5 * self.alpha) / gamma(1 - 0.5 * self.alpha)
        e = - (2 + self.alpha)

        return c, e

    def pdf_distance_asymptotic(self, r):

        c, e = self.pdf_asymptotic_constants()
        return c * r ** e


class Normal2D_mv(_Normal2D_base):
    """Isotropic bivariate normal distribution.

    Wrapper for SciPy implementation of multivariate normal.
    """

    def __init__(self, scale=1.0, loc=(0.0, 0.0)):

        super().__init__()

        self.scale = scale
        self.loc = loc

        cov = self.covariance_matrix()
        from scipy.stats import multivariate_normal
        self.__distribution = multivariate_normal(loc, cov)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def pdf(self, x, y):

        pos = numpy.dstack((x, y))
        return self.__distribution.pdf(pos)

    def pdf_distance(self, r):

        return self.pdf(r, 0.0 * r)


class Normal2D_radial(_Normal2D_base):
    """Isotropic bivariate normal distribution.

    Separates amplitude and frequency.
    """

    def __init__(self, scale=1.0, loc=(0.0, 0.0)):

        super().__init__()

        self.scale = scale
        self.loc = loc

        self.__distribution_radius = Rayleigh(scale=scale)
        self.__distribution_angle = Uniform(left=0.0, right=2.0 * numpy.pi)

    def sample(self, size=None):

        r = self.__distribution_radius.sample(size=size)
        phi = self.__distribution_angle.sample(size=size)

        return numpy.stack((self.loc[0] + numpy.cos(phi) * r, self.loc[1] + numpy.sin(phi) * r)).T

    def pdf(self, x, y):

        _x = x - self.loc[0]
        _y = y - self.loc[1]
        r = numpy.sqrt(_x * _x + _y * _y)
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

    def cf_radius(self, k):

        return self.__distribution_radius.cf(k)


class Normal2D_direct(_Normal2D_base):
    """Isotropic bivariate normal distribution.

    Direct implementation (no SciPy).
    """

    def __init__(self, scale=1.0, loc=(0.0, 0.0)):

        super().__init__()

        self.scale = scale
        self.loc = loc

    def sample(self, size=None):

        u = numpy.random.random(size)
        r = numpy.sqrt(- 2.0 * numpy.log(u)) * self.scale

        phi = numpy.random.random(size) * 2.0 * numpy.pi

        return numpy.stack((self.loc[0] + numpy.cos(phi) * r, self.loc[1] + numpy.sin(phi) * r)).T

    def pdf(self, x, y):

        _x = x - self.loc[0]
        _y = y - self.loc[1]
        r = numpy.sqrt(_x * _x + _y * _y)
        return self.pdf_distance(r)

    def pdf_distance(self, r):

        return 1.0 / (2.0 * numpy.pi * self.scale * self.scale) * numpy.exp(- r * r / (2.0 * self.scale * self.scale))

    def pdf_radius(self, r):

        return r / (self.scale * self.scale) * numpy.exp(- r * r / (2.0 * self.scale * self.scale))

    def cf_radius(self, k):

        raise NotImplementedError


class Normal2D(BivariateDistribution):
    """Isotropic bivariate normal distribution.

    Prefers the direct implementation.
    """

    alpha = 2.0

    def __new__(cls, direct=0, **kwargs):

        if direct == 0:
            return Normal2D_direct(**kwargs)
        elif direct == 1:
            return Normal2D_radial(**kwargs)
        else:
            return Normal2D_mv(**kwargs)
