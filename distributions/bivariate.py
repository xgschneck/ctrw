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

from .base import BivariateDistribution


class Uniform2D(BivariateDistribution):
    """Bivariate uniform distribution on bounded domains."""

    def __init__(self, domain_size=1.0):

        self.domain_size = domain_size

    def sample(self, size=None):

        if size is None:
            size = 2
        else:
            size = (size, 2)

        return numpy.random.random(size) * self.domain_size

    def pdf(self, x, y):

        return 1.0 / self.domain_size / self.domain_size

    def pdf_radius(self, r):

        raise NotImplementedError

    def covariance_matrix(self):

        raise NotImplementedError
