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

import ext_levy.levy as levy

from .base import ContinuousDistribution


class _Stable_base(ContinuousDistribution):

    def __init__(self, alpha=1.0, beta=0.0, scale=1.0, loc=0.0):

        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.loc = loc

        if self.beta == 1.0 and self.alpha < 1.0:
            self.min = 0.0
            self.max = numpy.inf
            self._plot_support = [0.0, 20.0]
        else:
            self.min = -numpy.inf
            self.max = numpy.inf
            self._plot_support = [-2.0, 2.0]

    def __repr__(self):

        return "Stable(alpha={}, beta={}, scale={}, loc={})".format(self.alpha, self.beta, self.scale, self.loc)

    def __str__(self):

        # NOTE location parameter skipped
        return "Stable($\\alpha={:0.02f}, \\beta={:0.02f}, \gamma={:0.02f}$)".format(self.alpha, self.beta, self.scale)

    def cf(self, s):

        if self.alpha == 1.0:
            theta = - 2.0 / numpy.pi * numpy.log(numpy.abs(s))
        else:
            theta = numpy.tan(numpy.pi * self.alpha * 0.5)

        return numpy.exp(1j * s * self.loc - numpy.abs(self.scale * s) ** self.alpha * (s - 1j * self.beta * numpy.sign(s) * theta))

    def pdf_asymptotic_constants(self):

        if self.beta != 0.0:
            raise NotImplementedError

        c = self.scale ** self.alpha * numpy.sin(numpy.pi * self.alpha * 0.5) * scipy.special.gamma(1.0 + self.alpha) / numpy.pi
        e = - (1.0 + self.alpha)
        return c, e

    def pdf_asymptotic(self, t):

        c, e = self.pdf_asymptotic_constants()
        return c * t ** e


class Stable_scipy(_Stable_base):
    """Univariate stable distribution.

    Just wraps the SciPy implementation [D4].
    """

    def __init__(self, alpha=1.0, beta=0.0, scale=1.0, loc=0.0):

        super().__init__(alpha=alpha, beta=beta, scale=scale, loc=loc)

        self.__distribution = scipy.stats.levy_stable(alpha=alpha, beta=beta, loc=loc, scale=scale)

    def pdf(self, t):

        return self.__distribution.pdf(t)

    def sample(self, size=None):

        return self.__distribution.rvs(size)


class Stable_extern(_Stable_base):
    """Univariate stable distribution.

    Wrapper for the library implementation [D5].
    """

    def __init__(self, alpha=1.0, beta=0.0, scale=1.0, loc=0.0):

        assert alpha >= 0.5

        super().__init__(alpha=alpha, beta=beta, scale=scale, loc=loc)

        self.__extern_params = levy.Parameters(par="1", alpha=alpha, beta=beta, mu=loc, sigma=scale).get("0")

    def pdf(self, t):

        return levy.levy(t, *self.__extern_params, cdf=False)

    def sample(self, size=None):

        return levy.random(*self.__extern_params, shape=size)


class Stable(ContinuousDistribution):
    """Univariate stable distribution.

    The SciPy implementation [D4] of the PDF has issues
    with small arguments. The library implementation [D5]
    does not suffer from this problem but cannot handle
    small alpha values.
    """

    def __new__(cls, **kwargs):

        if kwargs["alpha"] < 0.5:
            return Stable_scipy(**kwargs)
        else:
            return Stable_extern(**kwargs)
