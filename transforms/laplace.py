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
from scipy.integrate import quad, trapz
import scipy.interpolate
import mpmath


def _laplace_functional(ft, s):

    return quad(lambda t: ft(t) * numpy.exp(- s * t), 0.0, numpy.inf)[0]


laplace_functional = numpy.vectorize(_laplace_functional)
"""Laplace transform

of a function ft(t) evaluated at locations s.
Direct quadrature.
"""


inverse_laplace_functional = numpy.vectorize(mpmath.invertlaplace, excluded=(0, ))
"""Inverse Laplace transform

of a function Fs(s) at locations t.
Numerical implementation [T2].
"""


def laplace(x, fx, t):
    """Laplace transform

    of a function fx with given sample points x,
    evaluated at location t.
    Quadrature.
    """

    assert x[0] >= 0.0
    y = numpy.exp(- numpy.outer(t, x)) * fx
    return trapz(y, x)


def inverse_laplace(t, Ft, x):
    """Inverse Laplace transform

    of a function Ft with given sample points t,
    evaluated at x.
    Reference [T2].
    """

    raise NotImplementedError  # not tested

    assert len(x.shape) == 1

    mpalgorithm = mpmath.calculus.inverselaplace.Stehfest(mpmath.mp)
    # mpalgorithm = mpmath.calculus.inverselaplace.FixedTalbot(mpmath.mp)
    # mpalgorithm = mpmath.calculus.inverselaplace.deHoog(mpmath.mp)

    # use crude interpolation
    F = scipy.interpolate.interp1d(t, Ft, bounds_error=False, fill_value=0.0)

    fx = []

    for _x in x:

        mpalgorithm.calc_laplace_parameter(_x)

        # algorithms need f evaluated at specific points
        p = mpalgorithm.p

        Fp = F(numpy.array(p, dtype="double"))

        _fx = mpalgorithm.calc_time_domain_solution(Fp, _x)
        fx.append(_fx)

    return fx


def laplace_complex(x, fx, dx=None, N=None):

    # use FFT for numerical Laplace transform
    # see e.g. https://stackoverflow.com/questions/38316225/numerical-laplace-transform-python

    raise NotImplementedError  # not tested

    assert x[0] >= 0.0

    x = numpy.concatenate((-x[:0:-1], x))
    fx = numpy.concatenate((numpy.zeros(len(fx) - 1), fx))

    if N is None:
        N = len(x)

    if dx is None:
        dx = x[1] - x[0]

    # real component
    a = numpy.linspace(x[0], x[-1], N - 1)

    # imaginary component
    # b = numpy.fft.fftfreq(N, d=dx) * 2.0 * numpy.pi
    b = numpy.fft.fftshift(numpy.fft.fftfreq(N, d=dx)) * 2.0 * numpy.pi

    exp_factor = numpy.exp(numpy.outer(a, x))
    exp_factor /= numpy.sum(exp_factor)

    # create array for parallel FFT
    fxe = exp_factor * fx

    F = numpy.fft.ifft(fxe, norm="ortho") * dx * numpy.sqrt(N)
    F = numpy.fft.fftshift(numpy.fft.ifft(numpy.fft.fftshift(fxe), norm="ortho")) * dx * numpy.sqrt(N)

    return (a, b, F)
