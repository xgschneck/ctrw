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

from ext_mittagleffler.mittag_leffler import ml as mittag_leffler

from .base import ContinuousDistribution


class Lomax(ContinuousDistribution):

    def __init__(self, beta=1.0, scale=1.0):

        self.beta = beta
        self.scale = scale
        self.min = 0.0
        self.max = numpy.inf

        self.__distribution = scipy.stats.lomax(beta, scale=scale)

        self._plot_support = [0.0, 100.0]

    def __repr__(self):

        return "Lomax(beta={}, scale={})".format(self.beta, self.scale)

    def __str__(self):

        return "Lomax($\\beta={:0.02f}, \\tau={:0.02f}$)".format(self.beta, self.scale)

    def pdf(self, t):

        return self.__distribution.pdf(t)

    def pdf_asymptotic(self, t):

        return self.beta * self.scale ** self.beta * t ** - (1.0 + self.beta)

    def sample(self, size=None):

        return self.__distribution.rvs(size)


class Pareto(ContinuousDistribution):

    def __init__(self, beta=1.0, scale=1.0):

        self.beta = beta
        self.scale = scale
        self.min = scale  # TODO
        self.max = numpy.inf

        self.__distribution = scipy.stats.pareto(beta, scale=scale, loc=-1.0)

        self._plot_support = [0.0, 100.0]

    def __repr__(self):

        return "Pareto(beta={}, scale={})".format(self.beta, self.scale)

    def __str__(self):

        return "Pareto($\\beta={:0.02f}, \\tau={:0.02f}$)".format(self.beta, self.scale)

    def pdf(self, t):

        return self.__distribution.pdf(t)

    def sample(self, size=None):

        return self.__distribution.rvs(size)


class SquarePareto(ContinuousDistribution):

    def __init__(self, scale=1.0):

        self.scale = scale
        self.min = 0.0  # TODO
        self.max = numpy.inf

        self.__pareto = scipy.stats.pareto(b=1.0, scale=scale, loc=0.0)

        self._plot_support = [0.0, 20.0]

    def __repr__(self):

        return "SquarePareto(scale={})".format(self.scale)

    def __str__(self):

        return "SquarePareto($\sigma={:0.02f}$)".format(self.scale)

    def pdf(self, r):

        return self.scale * r / (r * r + self.scale * self.scale) ** 1.5

    def sample(self, size=None):

        Z = self.__pareto.rvs(size)
        return numpy.sqrt(Z * Z - self.scale * self.scale)

    def cf(self, t):

        # difficul integral in fourier transform!
        raise NotImplementedError


class MittagLefflerFunction_Quad(object):
    """Mittag-Leffler Function.

    Crude implementation of the algorithm in [D3].
    """

    def __init__(self, alpha, beta, rho=1.0):

        self.alpha = alpha
        self.beta = beta
        self.rho = rho

    def __call__(self, z):

        if 1.0 < self.alpha:

            k0 = numpy.floor(self.alpha) + 1
            k = numpy.arange(0, k0)
            z_ = z ** (1.0 / k0) * numpy.exp(2.0 * numpy.pi * 1j * k / k0)

            mlf = MittagLefflerFunction_Quad(self.alpha / k0, self.beta, self.rho)

            return sum(mlf(z__) for z__ in z_) / k0

        elif z == 0.0:

            return 1.0 / scipy.special.gamma(self.beta)

        elif abs(z) < 1.0:

            k0 = max(
                numpy.ceil((1.0 - self.beta) / self.alpha),
                numpy.ceil(numpy.log(self.rho * (1.0 - abs(z))) / numpy.log(abs(z)))
            )

            k = numpy.arange(0, k0 + 1)

            return (z ** k / scipy.special.gamma(self.beta + self.alpha * k)).sum()

        elif abs(z) > numpy.floor(10.0 + 5.0 * self.alpha):

            k0 = numpy.floor(- numpy.log(self.rho) / numpy.log(abs(z)))
            k = numpy.arange(1, k0 + 1)
            _out = - (z ** -k / scipy.special.gamma(self.beta - self.alpha * k)).sum()

            if numpy.angle(z) < (self.alpha * numpy.pi / 4.0 + 0.5 * numpy.pi * min(1.0, self.alpha)):
                return _out + 1.0 / self.alpha * z ** ((1.0 - self.beta) / self.alpha) * numpy.exp(z ** (1.0 / self.alpha))
            else:
                return _out

        else:

            if self.beta < 0.0:
                chi0 = max(
                    (abs(self.beta) + 1.0) ** self.alpha,
                    2.0 * abs(z),
                    (- 2.0 * numpy.log(numpy.pi * self.rho / (6.0 * (abs(self.beta) + 2.0) * (2.0 * abs(self.beta)) ** abs(self.beta)))) ** self.alpha
                )
            else:
                chi0 = max(
                    1.0,
                    2.0 * abs(z),
                    (- numpy.log(numpy.pi * self.rho / 6.0)) ** self.alpha
                )

            def K(chi):
                out = 1.0 / self.alpha / numpy.pi * chi ** ((1.0 - self.beta) / self.alpha) * numpy.exp(- chi ** (1.0 / self.alpha))
                out *= chi * numpy.sin(numpy.pi * (1.0 - self.beta)) - z * numpy.sin(numpy.pi * (1.0 - self.beta + self.alpha))
                out /= chi * chi - 2.0 * chi * z * numpy.cos(self.alpha * numpy.pi) + z * z
                return out

            def P(phi, eps):
                out = 1.0 / 2.0 / self.alpha / numpy.pi * eps ** (1.0 + (1.0 - self.beta) / self.alpha)
                out *= numpy.exp(eps ** (1.0 / self.alpha) * numpy.cos(phi / self.alpha))
                w = phi * (1.0 + (1.0 - self.beta) / self.alpha) + eps ** (1.0 / self.alpha) * numpy.sin(phi / self.alpha)
                out *= numpy.cos(w) + 1j * numpy.sin(w)
                out /= eps * numpy.exp(1j * phi) - z
                return numpy.real(out)

            from scipy.integrate import quad

            if abs(numpy.angle(z)) > self.alpha * numpy.pi:
                if self.beta <= 1.0:
                    return quad(K, 0.0, chi0)[0]
                else:
                    return quad(K, 1.0, chi0)[0] + \
                        quad(P, - self.alpha * numpy.pi, self.alpha * numpy.pi, args=(1.0,))[0]

            elif abs(numpy.angle(z)) < self.alpha * numpy.pi:
                out = 1.0 / self.alpha * z ** ((1.0 - self.beta) / self.alpha) * numpy.exp(z ** (1.0 / self.alpha))
                if self.beta <= 1.0:
                    return out + quad(K, 0.0, chi0)[0]
                else:
                    return out + quad(K, abs(z) / 2.0, chi0)[0] + \
                        quad(P, - self.alpha * numpy.pi, self.alpha * numpy.pi, args=(abs(z) / 2.0,))[0]

            else:
                return quad(K, (abs(z) + 1.0) / 2.0, chi0)[0] + \
                    quad(P, - self.alpha * numpy.pi, self.alpha * numpy.pi, args=((abs(z) + 1.0) / 2.0,))[0]


def MittagLefflerFunction(x, a, b, parallel=False):
    """Mittag-Leffler Function.

    Wrapper for the implementation in [D2].
    """

    if not parallel:

        return mittag_leffler(x, a, b)

    else:

        args = [(x_, a, b) for x_ in x.flatten()]

        import multiprocessing
        nprocs = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(nprocs, numpy.random.seed)
        return numpy.array(pool.starmap(mittag_leffler, args, len(x) // nprocs)).reshape(x.shape)


class MittagLeffler(ContinuousDistribution):
    """Mittag-Leffler distribution.

    Based on [D2] and [D6].
    """

    def __init__(self, beta=1.0, scale=1.0):

        self.beta = beta
        self.scale = scale
        self.min = 0.0
        self.max = numpy.inf

        self.__ml = MittagLefflerFunction

        # sampling approach 1
        self.__uniform_distribution = scipy.stats.uniform(loc=0.0, scale=1.0)

        # sampling approach 2
        # self.__exponential_distribution = scipy.stats.expon(scale=scale)
        # self.__stable_distribution = scipy.stats.levy_stable(beta, 1.0, loc=0.0, scale=1.0 / 8.0)

        self._plot_support = [0.0, max(10.0, 10.0 * scale * scale)]

    def __repr__(self):

        return "MittagLeffler(beta={}, scale={})".format(self.beta, self.scale)

    def __str__(self):

        return "MiLeff($\\beta={:0.02f}, \\tau={:0.02f}$)".format(self.beta, self.scale)

    def pdf(self, t):

        # - d / dt E_beta ( - (t/scale)^beta )
        # 1 / scale * (t/scale)^(beta - 1) * E_beta,beta ( - (t/scale)^beta )

        out = numpy.zeros(t.shape)
        out[t < 0.0] = 0.0
        out[t >= 0.0] = 1.0 / self.scale * (t[t >= 0.0] / self.scale) ** (self.beta - 1.0) * \
            self.__ml(- (t[t >= 0.0] / self.scale) ** self.beta, self.beta, self.beta)
        return out

        # return 1.0 / self.scale * (t / self.scale) ** (self.beta - 1.0) * self.__ml(- (t / self.scale) ** self.beta, self.beta, self.beta)

    def sample(self, size=None):

        u = self.__uniform_distribution.rvs(size)
        v = self.__uniform_distribution.rvs(size)

        return - self.scale * numpy.log(u) * \
            (numpy.sin(self.beta * numpy.pi) / numpy.tan(self.beta * numpy.pi * v) - numpy.cos(self.beta * numpy.pi)) ** (1.0 / self.beta)

        # return self.__exponential_distribution.rvs(size=size) ** (1.0 / self.beta) * self.__stable_distribution.rvs(size=size)

    def survival(self, t):

        return self.__ml(-(t / self.scale) ** self.beta, self.beta, 1.0)

    def cf(self, s):

        raise NotImplementedError

    def asymptotic_constant(self):

        return numpy.sin(self.beta * numpy.pi) / numpy.pi

    def pdf_asymptotic_constants(self):

        c = self.scale ** self.beta / numpy.abs(scipy.special.gamma(-self.beta))
        e = - (1.0 + self.beta)
        return (c, e)

    def pdf_asymptotic(self, t):

        c, e = self.pdf_asymptotic_constants()
        return c * t ** e
