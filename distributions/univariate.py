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


class Constant(ContinuousDistribution):
    """Degenerate distribution."""

    def __init__(self, scale=1.0):
        """The `scale` is the constant value returned by this distribution.
        In general, scale is the inverse of rate
        and rate is the number of events per unit interval.
        """

        self.scale = scale
        self.min = scale
        self.max = scale

    def __repr__(self):

        return "Constant(scale={})".format(self.scale)

    def __str__(self):

        return "Const($\delta={:0.02f}$)".format(self.scale)

    def pdf(self, t):

        # simulates delta distribution
        return (numpy.abs(t - self.scale) < 1e-1) * 1000

    def sample(self, size=None):

        return numpy.full(size, self.scale)

    def mean(self):

        return self.scale


class Uniform(ContinuousDistribution):
    """Uniform distribution."""

    def __init__(self, left=0.0, right=1.0):

        self.left = left
        self.right = right

        self.min = left
        self.max = right

        self.__distribution = scipy.stats.uniform(loc=0.0, scale=2.0 * numpy.pi)

    def __repr__(self):

        return "Uniform(left={}, right={})".format(self.left, self.right)

    def __str__(self):

        return "Uniform(${:0.02f}, {:0.02f}$)".format(self.left, self.right)

    def pdf(self, t):

        return numpy.logical_and(self.left <= t, t < self.right)

    def sample(self, size=None):

        return self.__distribution.rvs(size=size)

    def mean(self):

        return (self.left + self.right) * 0.5
