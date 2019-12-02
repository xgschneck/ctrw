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
import hankel


def hankel_functional(fx, k):
    """Calculate Hankel transform

    of a function fx(x) evaluated at the locations k.
    Reference [T1].
    Maybe log-spaced k improves quality?
    """

    ht = hankel.HankelTransform(nu=0, N=500, h=0.01)
    return ht.transform(fx, k, ret_err=False)


def inverse_hankel_functional(Fk, x):
    """Inverse Hankel transform

    of a function Fk(k) evaluated at locations x.
    Reference [T1].
    """

    ht = hankel.HankelTransform(nu=0, N=500, h=0.01)
    return ht.transform(Fk, x, ret_err=False, inverse=True)


def fourier_2d(x, y, f):
    """2D Fourier transform using FFT

    of a function f given at locations (x, y).
    The evaluation points are determined by the algorithm.
    """

    assert x.shape == y.shape == f.shape
    assert x.shape[0] == x.shape[1]

    N = x.shape[0]
    dx = x[0, 1] - x[0, 0]

    k = numpy.fft.fftshift(numpy.fft.fftfreq(N, d=dx)) * 2.0 * numpy.pi
    Fk = numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(f), norm="ortho")) * dx * dx * N

    return numpy.meshgrid(k, k), Fk


def inverse_fourier_2d(u, v, F):
    """Inverse 2D Fourier transform using FFT

    of the function F given at locations (u, v).
    The evaluation points are determined by the algorithm.
    """

    assert u.shape == v.shape == F.shape
    assert u.shape[0] == u.shape[1]

    N = u.shape[0]
    du = u[0, 1] - u[0, 0]

    x = numpy.fft.fftshift(numpy.fft.fftfreq(N, d=du)) * 2.0 * numpy.pi
    # f = numpy.fft.fftshift(numpy.fft.irfft(F, len(F), norm="ortho")) * dk * numpy.sqrt(N)
    f = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(F), norm="ortho")) * du * du * N / (4.0 * numpy.pi * numpy.pi)

    assert numpy.allclose(f.imag, 0.0)

    return numpy.meshgrid(x, x), f.real


def fourier(x, fx, dx=None, N=None, semi=False):
    """Fourier transform

    of a function fx evaluated at locations x.
    Approximates Integral f(x) * exp(ikx) dx.
    Set `semi=True` if only positive real locations passed.
    The sample points in the frequency domain are determined by the algorithm.
    """

    if semi:
        if x[0] + x[-1] == 0.0:
            raise ValueError("expected positive real axis")
        x = numpy.concatenate((-x[:0:-1], x))
        fx = numpy.concatenate((numpy.zeros(len(fx) - 1), fx))
    else:
        if x[0] + x[-1] != 0.0:
            raise ValueError("expected whole real axis")

    if N is None:
        N = len(x)

    if dx is None:
        dx = x[1] - x[0]

    k = numpy.fft.rfftfreq(N, d=dx) * 2.0 * numpy.pi
    Fk = numpy.fft.rfft(numpy.fft.fftshift(fx), norm="ortho") * dx * numpy.sqrt(N)

    return (k, Fk)


def inverse_fourier(k, Fk, dk=None, N=None, semi=False):
    """Inverse Fourier transform

    of a function Fk evaluated at locations k.
    approximates 1.0 / (2pi) * Integral f(k) * exp(-ikx) dk.
    set `semi=True` if only positive real locations are passed.
    The sample points of the returned transform are determined by the algorithm.
    """

    if N is None:
        N = len(k)

    if dk is None:
        dk = k[1] - k[0]

    if semi:
        x = numpy.fft.fftfreq(N, d=dk)[:N // 2] * 2.0 * numpy.pi
        fx = numpy.fft.irfft(Fk, len(Fk), norm="ortho")[:N // 2] * dk * numpy.sqrt(N) / (2.0 * numpy.pi) * 2.0
    else:
        x = numpy.fft.fftshift(numpy.fft.fftfreq(N, d=dk)) * 2.0 * numpy.pi
        fx = numpy.fft.fftshift(numpy.fft.irfft(Fk, len(Fk), norm="ortho")) * dk * numpy.sqrt(N) / (2.0 * numpy.pi)

    return (x, fx)
