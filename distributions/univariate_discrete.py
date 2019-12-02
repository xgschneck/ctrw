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

from .base import DiscreteDistribution


class DiscreteConstant(DiscreteDistribution):

    def __init__(self, value):

        assert type(value) is int

        self.value = value
        self.min = value
        self.max = value

    def sample(self, size=None):

        return numpy.full(size, self.value, dtype=int)


class Binomial(DiscreteDistribution):

    def __init__(self, p=0.5, n=10):

        self.n = n
        self.p = p
        self.min = 0
        self.max = n

        self.__distribution = scipy.stats.binom(n=n, p=1.0 - p)

        self._plot_support = [0.0, n]

    def __repr__(self):

        return "Binomial(p={}, n={})".format(self.p, self.n)

    def __str__(self):

        return "Binom($p={:0.02f}, n={}$)".format(self.p, self.n)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def pdf(self, t):

        return self.__distribution.pmf(t)

    def mean(self):

        return self.__distribution.mean()


class Geometric(DiscreteDistribution):

    def __init__(self, p=0.5):

        self.p = p
        self.min = 0
        self.max = numpy.inf

        self.__distribution = scipy.stats.geom(p=p, loc=-1)

        self._plot_support = [0.0, 10.0]

    def __repr__(self):

        return "Geometric(p={})".format(self.p)

    def __str__(self):

        return "Geom($p={:0.02f}$)".format(self.p)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def pdf(self, t):

        return self.__distribution.pmf(t)

    def mean(self):

        return self.__distribution.mean()


class GeometricTruncated(DiscreteDistribution):

    def __init__(self, p=0.5, n=10):

        self.p = p
        self.n = n
        self.min = 0
        self.max = n

        support = list(range(n + 1))
        probabilities = [self.pdf(t) for t in support]
        assert numpy.isclose(sum(probabilities), 1.0)

        self.__distribution = scipy.stats.rv_discrete(name="GeometricTruncated", values=(support, probabilities))

        self._plot_support = [0.0, n]

    def __repr__(self):

        return "GeometricTruncated(p={}, n={})".format(self.p, self.n)

    def __str__(self):

        return "Geom($p={:0.02f}, n={}$)".format(self.p, self.n)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def pdf(self, t):

        return self.p * (1.0 - self.p) ** t / (1.0 - (1.0 - self.p) ** (self.n + 1))

    def mean(self):

        raise NotImplementedError


class Poisson(DiscreteDistribution):

    def __init__(self, scale=1.0):

        # rate = 1/scale is the average number of events per interval
        # expectation = rate

        self.scale = scale
        self.min = 0
        self.max = numpy.inf

        rate = 1.0 / scale
        self.__distribution = scipy.stats.poisson(rate, loc=0)

        self._plot_support = [0.0, 10.0]

    def __repr__(self):

        return "Poisson(scale={})".format(self.scale)

    def __str__(self):

        return "Poisson($\delta={:0.02f}$)".format(self.scale)

    def pdf(self, t):

        return self.__distribution.pmf(t)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)
