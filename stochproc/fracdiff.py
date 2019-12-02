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
import scipy
import mpmath

import distributions
import transforms

from .ctrw import CTRW


class FractionalDiffusion(CTRW):
    """Provides the identification of CTRW with
    the fractional diffusion equation.
    """

    def __init__(self, **kwargs):
        """If jump and waitingtime distributions are defined manually,
        the analytical results in connection with fractional diffusion may not hold."""

        if "jump" in kwargs.keys():
            raise ValueError("configuration (jump) may not simulate fractional diffusion")
        if "wait" in kwargs.keys():
            raise ValueError("configuration (wait) may not simulate fractional diffusion")

        super().__init__(**kwargs)

    def generalized_diffusion_constant(self):
        """Calculate generalized diffusion constant."""

        return self.DC ** (self.alpha / self.dimensions) / self.TC ** self.beta

    def sample_average_displacement_constants(self, moment=None):
        """Calculate the coefficient and exponent of the expected
        powerlaw in sample average displacement.

        This calculation is only valid if `moment < alpha / beta`.
        The calculation of the coefficient is currently not implemented!
        The 1D case is discussed in [C5] and [C6].
        """

        if moment is None:
            moment = self.alpha / self.beta

        exponent = moment * self.beta / self.alpha
        constant = 1.0  # TODO

        return (constant, exponent)

    def time_average_displacement(self, T, Dt, moment=None):
        """Evaluate the expected time average displacement of a fractional diffusion process.

        Time average displacement is discussed in [C1], [C2] and [C3].
        The parameter `T` is a scalar final time value.
        The parameter `Dt` is the interval length usually denoted Delta.
        The parameter `moment` is the moment of the spatial displacement
        (compare sample average displacement).
        Only `moment=alpha` is implemented!
        The constant should be a random variable!
        """

        if moment != self.alpha:
            raise NotImplementedError("can calculate TA displacement for m == alpha only")

        constant = 1.0  # TODO
        return constant * (T ** (1.0 + self.beta) - Dt ** (1.0 + self.beta) - (T - Dt) ** (1.0 + self.beta)) / (1.0 + self.beta) / (T - Dt)

    def time_average_displacement_asymptotic(self, T, Dt, moment=None):
        """Evaluate the asymptotic approximation of the
        expected time average displacement.

        See `time_average_displacement` and [C1], [C2], [C3].
        """

        if moment != self.alpha:
            raise NotImplementedError("can calculate TA displacement for m == alpha only")

        constant = 1.0  # TODO
        return constant * Dt * T ** (self.beta - 1.0)

    def _green_function_dft(self, t, r, factor=3):
        """Evaluate Green function using FFT.

        FFT works on a regular lattice of sample points.
        Ensure that `r` is regularly spaced.
        If the accuracy is too low, increase the sample points
        or increase the frequency domain (`factor`).
        """

        if self.dimensions == 1:

            assert len(r.shape) == 1  # is 1d
            assert numpy.isclose(r[0], r[-1])  # is symmetric
            assert numpy.isclose(numpy.abs(numpy.diff(r)).std(), 0.0)  # is regular
            assert numpy.isclose(r[(r.size - 1) // 2], 0.0)  # is centered

            N = r.size * factor
            N2 = N // 2
            a = (N2 - (N2 // factor))
            b = (N2 + (N2 // factor)) + 1

            dr = numpy.abs(r[1] - r[0])
            k = numpy.fft.rfftfreq(N, d=dr) * 2.0 * numpy.pi
            Fk = self.green_function_fourier(t, k, parallel=True)

            fr = numpy.fft.irfft(Fk, n=N, norm="ortho")
            fr = numpy.fft.fftshift(fr)[a:b]

            dk = k[1] - k[0]
            rr = numpy.fft.fftshift(numpy.fft.fftfreq(N, d=dk))[a:b] * 2.0 * numpy.pi
            assert r.size == rr.size
            assert numpy.allclose(numpy.abs(rr) - r, 0.0)

            return fr / dr / numpy.sqrt(N)

        else:

            assert len(r.shape) == 2  # is 2d
            assert r.shape[0] == r.shape[1]
            assert numpy.allclose(r[:, 0], r[:, -1])  # is symmetric
            assert numpy.allclose(r[0, :], r[-1, :])  # is symmetric
            assert numpy.isclose(numpy.abs(numpy.diff(r[(r.shape[0] - 1) // 2, :])).std(), 0.0)  # is regular
            assert numpy.isclose(numpy.abs(numpy.diff(r[:, (r.shape[1] - 1) // 2])).std(), 0.0)  # is regular
            assert numpy.isclose(r[(r.shape[0] - 1) // 2, (r.shape[1] - 1) // 2], 0.0)  # is centered

            N = r.shape[0] * factor
            N2 = N // 2
            a = (N2 - (N2 // factor))
            b = (N2 + (N2 // factor)) + 1

            dr = numpy.abs(r[(r.shape[0] - 1) // 2, 1] - r[(r.shape[0] - 1) // 2, 0])
            k = numpy.fft.rfftfreq(N, d=dr) * 2.0 * numpy.pi

            Fk = numpy.zeros((k.size, k.size))
            i, j = numpy.triu_indices(k.size)

            k_triu = numpy.sqrt(k[i] * k[i] + k[j] * k[j])
            Fk[i, j] = self.green_function_fourier(t, k_triu, parallel=True)
            Fk[j, i] = Fk[i, j]
            Fk = numpy.hstack((Fk[:, :0:-1], Fk))
            Fk = numpy.vstack((Fk[:0:-1, :], Fk))
            Fk = numpy.fft.ifftshift(Fk)

            fr = numpy.fft.irfft2(Fk, s=(N, N), norm="ortho")
            fr = numpy.fft.fftshift(fr).take(range(a, b), axis=0).take(range(a, b), axis=1)

            dk = k[1] - k[0]
            rr = numpy.fft.fftshift(numpy.fft.fftfreq(N, d=dk))[a:b] * 2.0 * numpy.pi
            assert r.shape[0] == rr.size
            assert numpy.allclose(numpy.abs(rr) - r[r.shape[0] // 2, :], 0.0)

            return fr / dr / dr / N

    def _green_function_ft(self, t, r):
        """Evaluate Green function using functional Fourier transform."""

        if self.dimensions == 2:

            out = transforms.inverse_hankel_functional(lambda k: self.green_function_fourier(t, k), r)
            out[numpy.isclose(r, 0.0)] = numpy.nan
            return out

        else:

            # TODO Fk = lambda k: lambda k: self.green_function_fourier(t, k)
            raise NotImplementedError("functional inverse Fourier transform not implemented for dim == 1")

    def _green_function_lt(self, t, r):
        """Evaluate Green function using functional Laplace transform."""

        ret = [transforms.inverse_laplace_functional(lambda s: self.green_function_laplace(s, _r, "mpmath"), t) for _r in r.flat]
        return numpy.array(ret, dtype=float).reshape(r.shape)

    def green_function_origin(self, t):
        """Evaulate Green function at origin.

        See [C7] for 1D case.
        """

        if self.dimensions == 1:

            out = 1.0 / numpy.pi / self.alpha * scipy.special.gamma(1.0 / self.alpha)
            if self.beta != 1.0:
                if self.alpha <= 1.0:
                    raise NotImplementedError  # see [C7]
                out *= scipy.special.gamma(1.0 - 1.0 / self.alpha) / scipy.special.gamma(1.0 - self.beta / self.alpha)
            out *= t ** (- self.beta / self.alpha)
            return out

        else:

            raise NotImplementedError("Green function at origin not implemented for 2D")

    def green_function_fourier(self, t, k, parallel=False):
        """Evaluate the Green function in frequency domain."""

        return distributions.MittagLefflerFunction(-(numpy.abs(k) * self.DC ** (1.0 / self.dimensions)) ** self.alpha * (t / self.TC) ** self.beta,
                                                   self.beta, 1.0, parallel=parallel)

    def green_function_laplace(self, s, r, lib="scipy"):
        """Evaulates Laplace transformed Green function.

        Only for `alpha == 2`!
        `lib` can be either `scipy` or `mpmath`, the latter is required
        for use with the mpmath implementation of inverse Laplace transform.
        """

        if self.alpha != 2.0:
            raise NotImplementedError("Laplace transformed evaluation not implemented for alpha != 2")

        if lib == "mpmath":
            if not numpy.isscalar(r):
                raise ValueError("mpmath implementation requires scalar argument r")
        elif lib == "scipy":
            pass
        else:
            raise ValueError("unknown implementation/library selected")

        if self.dimensions == 1:

            ret = (s * self.TC) ** (0.5 * self.beta - 1.0)
            if lib == "scipy":
                ret *= numpy.exp(-r / self.DC * (s * self.TC) ** (0.5 * self.beta))
            elif lib == "mpmath":
                ret *= mpmath.exp(-r / self.DC * (s * self.TC) ** (0.5 * self.beta))
            else:
                raise ValueError
            ret *= 0.5 / self.DC * self.TC

        else:

            # NOTE evaluation at origin is faulty
            # possible approximation of bessel k0 for small arguments
            # - s ** (beta - 1.0) * (numpy.log(0.5 * r * s ** (0.5 * beta)) + numpy.euler_gamma)

            ret = (s * self.TC) ** (self.beta - 1.0)
            if lib == "scipy":
                ret *= scipy.special.k0(r / numpy.sqrt(self.DC) * (s * self.TC) ** (0.5 * self.beta))
            elif lib == "mpmath":
                ret *= mpmath.besselk(0, r / mpmath.sqrt(self.DC) * (s * self.TC) ** (0.5 * self.beta))
            else:
                raise ValueError
            ret *= self.TC / self.DC / (2.0 * numpy.pi)

        return ret

    def green_scale(self, t):
        """Calculates the scale of the spatial distribution of particles for given time t.

        See [C7].
        """

        scale = 1.0
        scale *= self.DC ** (1 / self.dimensions)
        scale *= self.TC ** (- self.beta / self.alpha)
        scale *= t ** (self.beta / self.alpha)
        return scale

    def green_function(self, t, r, method="dft"):
        """Evaluates Green function.

        `r` is the distance from the initial location of trajectories.
        `method` is one of `dft`, `ft`, `lt` and `direct`.
        """

        if not numpy.isscalar(t):
            raise ValueError("Green function takes scalar value t")

        if method == "dft" or method == "fft":
            return self._green_function_dft(t, r)

        elif method == "ft":
            return self._green_function_ft(t, r)

        elif method == "lt":
            return self._green_function_lt(t, r)

        elif method == "direct":

            if (self.beta == 1.0 or self.beta == 0.0) and self.alpha == 2.0:

                scale = self.green_scale(t)
                if self.dimensions == 2:
                    return distributions.Normal2D(scale=scale * numpy.sqrt(2.0)).pdf_distance(r)
                else:
                    return distributions.Normal(scale=scale * numpy.sqrt(2.0)).pdf(r)

            elif (self.beta == 1.0 or self.beta == 0.0) and self.alpha == 1.0:

                raise NotImplementedError  # TODO use Cauchy distribution

            elif self.beta == 1.0 or self.beta == 0.0:

                scale = self.green_scale(t)
                if self.dimensions == 2:
                    return distributions.Stable2D(scale=scale, alpha=self.alpha).pdf_distance(r)
                else:
                    return distributions.Stable(scale=scale, alpha=self.alpha).pdf(r)

            else:

                raise NotImplementedError  # TODO use Fox H-functions

        else:

            raise ValueError("unknown method for evaluating Green function")
