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
import scipy.stats

from .base import ContinuousDistribution


class Laplace(ContinuousDistribution):

    def __init__(self, scale=1.0):

        self.scale = scale
        self.min = -numpy.inf
        self.max = numpy.inf

        self.__distribution = scipy.stats.laplace(scale=scale)

        self._plot_support = [-10.0, 10.0]

    def __repr__(self):

        return "Laplace(scale={})".format(self.scale)

    def __str__(self):

        return "Laplace($\delta={:0.02f}$)".format(self.scale)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def pdf(self, r):

        return self.__distribution.pdf(r)

    def mean(self):

        return self.__distribution.mean()

    def cf(self, t):

        return 1.0 / (1.0 + self.scale * self.scale * t * t)


class SkewLaplace(ContinuousDistribution):

    # skew Laplace distributions as defined in [D7]

    def __init__(self, scale=1.0, loc=0.0, skewness1=None, skewness2=None):

        self.scale = scale
        self.loc = loc

        # in the reference:
        # skewness1 = mu
        # skewness2 = kappa

        if skewness1 is None and skewness2 is None:
            skewness1 = 0.0
            skewness2 = 1.0
        elif skewness1 is None:
            self.skewness2 = skewness2
            self.skewness1 = - self.scale * (self.skewness2 - 1.0) / 2.0 / self.skewness2
        elif skewness2 is None:
            self.skewness1 = skewness1
            self.skewness2 = self.scale / \
                (self.skewness1 + numpy.sqrt(self.scale * self.scale + self.skewness1 * self.skewness1))
        else:
            raise ValueError("both skewness parameters set")

        self.min = -numpy.inf
        self.max = numpy.inf

        self.__distribution_exponential = scipy.stats.expon()
        self.__skewness_factor = self.skewness2 / (1.0 + self.skewness2 * self.skewness2)

        self._plot_support = [-100.0, 100.0]

    def __repr__(self):

        return "SkewLaplace(scale={}, loc={}, skewness1={}, skewness2={})".format(
            self.scale, self.loc, self.skewness1, self.skewness2)

    def __str__(self):

        return "SkewLaplace($\delta={:0.02f}, \\theta={:0.02f}, \kappa_{{1}}={:0.02f}, \kappa_{{2}}={:0.02f}$)".format(
            self.scale, self.loc, self.skewness1, self.skewness2)

    def sample(self, size=None):

        E1, E2 = self.__distribution_exponential.rvs(size=(2, size))
        return (E1 / self.skewness2 - E2 * self.skewness2) * self.scale + self.loc

    def pdf(self, r):

        return self.__skewness_factor / self.scale * numpy.exp(- self.skewness2 ** numpy.sign(r - self.loc) * numpy.abs(r - self.loc) / self.scale)

    def mean(self):

        raise NotImplementedError

    def cf(self, t):

        raise NotImplementedError
