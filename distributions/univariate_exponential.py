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


class Normal(ContinuousDistribution):

    def __init__(self, loc=0.0, scale=1.0):

        self.loc = loc
        self.scale = scale
        self.min = -numpy.inf
        self.max = numpy.inf

        self.__distribution = scipy.stats.norm(loc, scale)

        self._plot_support = [-10.0, 10.0]

    def __repr__(self):

        return "Normal(loc={}, scale={})".format(self.loc, self.scale)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def pdf(self, r):

        return self.__distribution.pdf(r)

    def mean(self):

        return self.__distribution.mean()

    def cf(self, t):

        return numpy.exp(- 1j * self.loc * t - 0.5 * (self.scale * t) ** 2.0)


class LogNormal(ContinuousDistribution):

    def __init__(self, loc=0.0, scale=1.0):

        self.loc = loc
        self.scale = scale
        self.min = 0.0
        self.max = numpy.inf

        mean = numpy.exp(loc + 0.5 * scale * scale)
        var = (numpy.exp(scale * scale) - 1.0) * numpy.exp(2.0 * loc + scale * scale)
        loc2 = 0.0
        scale2 = numpy.exp(loc)

        self.__distribution = scipy.stats.lognorm(scale, loc2, scale2)

        assert numpy.isclose(mean, self.__distribution.mean())
        assert numpy.isclose(var, self.__distribution.var())

        self._plot_support = [0.0, 10.0]

    def __repr__(self):

        return "LogNormal(loc={}, scale={})".format(self.loc, self.scale)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def pdf(self, r):

        return self.__distribution.pdf(r)

    def mean(self):

        return self.__distribution.mean()


class ExponentialPower(ContinuousDistribution):

    def __init__(self, beta=1.0, scale=1.0):

        self.beta = beta
        self.scale = scale
        self.min = -numpy.inf
        self.max = numpy.inf

        self.__distribution = scipy.stats.gennorm(beta, scale=scale)

        self._plot_support = [-10.0, 10.0]

    def __repr__(self):

        return "ExponentialPower(beta={}, scale={})".format(self.beta, self.scale)

    def __str__(self):

        return "ExpPow($\\beta={:0.02f}, \delta={:0.02f}$)".format(self.beta, self.scale)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def pdf(self, r):

        return self.__distribution.pdf(r)

    def mean(self):

        return self.__distribution.mean()

    def cf(self, t):

        raise NotImplementedError


class Exponential(ContinuousDistribution):

    def __init__(self, scale=1.0):

        self.scale = scale
        self.min = 0.0
        self.max = numpy.inf

        self.__distribution = scipy.stats.expon(loc=0.0, scale=self.scale)

        self._plot_support = [0.0, 20.0]

    def __repr__(self):

        return "Exponential(scale={})".format(self.scale)

    def __str__(self):

        return "Exp($\delta={:0.02f}$)".format(self.scale)

    def pdf(self, t):

        return self.__distribution.pdf(t)

    def sample(self, size=None):

        return self.__distribution.rvs(size)


class Rayleigh(ContinuousDistribution):

    def __init__(self, scale=1.0):

        self.scale = scale
        self.min = 0.0
        self.max = numpy.inf

        self.__distribution = scipy.stats.rayleigh(loc=0.0, scale=scale)  # scale_R = sigma = scale_N

        self._plot_support = [0.0, 20.0]

    def __repr__(self):

        return "Rayleigh(scale={})".format(self.scale)

    def __str__(self):

        return "Rayleigh($\sigma={:0.02f}$)".format(self.scale)

    def pdf(self, r):

        return self.__distribution.pdf(r)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def cf(self, t):

        C = self.scale * t / numpy.sqrt(2.0)
        return 1.0 - C * numpy.sqrt(numpy.pi) * numpy.exp(- C * C) * (scipy.special.erfi(C) - 1j)


class Gamma(ContinuousDistribution):

    def __init__(self, rate=1.0, shape=10.0):

        self.rate = rate
        self.shape = shape
        self.min = 0.0
        self.max = numpy.inf

        self.__distribution = scipy.stats.gamma(a=shape, loc=0.0, scale=1.0 / rate)

        self._plot_support = [0.0, 20.0]

    def __repr__(self):

        return "Gamma(rate={}, shape={})".format(self.rate, self.shape)

    def __str__(self):

        return "Gamma($\lambda={:0.02f}, s={:0.02f}$)".format(self.rate, self.shape)

    def pdf(self, t):

        return self.__distribution.pdf(t)

    def sample(self, size=None):

        return self.__distribution.rvs(size)
