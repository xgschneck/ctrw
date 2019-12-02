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

import distributions

from .particle import Particle


class Wait(object):
    """Static methods for parameterizing, instantiating and sampling from
    standard waitingtime distributions.

    The standard waitingtime distribution is the Mittag-Leffler distribution.
    For usual diffusion (`beta=1.0`) the exponential distribution is used.
    With `beta=0.0`, a degenerate distribution yields discrete time increments (by convention).
    """

    @staticmethod
    def wait_scale(delta_t=1.0, TC=1.0, beta=1.0):
        """Calculate the scale of waitingtime distributions."""

        if beta == 0.0:
            return TC * delta_t  # constant time increments by convention
        else:
            return TC * delta_t ** (1.0 / beta)

    @staticmethod
    def wait_setup(delta_t=1.0, TC=1.0, beta=1.0):
        """Return instantiated waitingtime distributions."""

        if delta_t <= 0.0:
            raise ValueError

        if TC <= 0.0:
            raise ValueError

        scale = Wait.wait_scale(delta_t, TC, beta)

        if beta > 1.0:
            raise NotImplementedError
        elif beta == 1.0:
            return distributions.Exponential(scale=scale)
        elif beta == 0.0:
            return distributions.Constant(scale=scale)  # by convention
        elif beta < 0.0:
            raise ValueError
        else:
            return distributions.MittagLeffler(beta=beta, scale=scale)

    @staticmethod
    def sample_dt(wait, T=numpy.inf, N=numpy.inf):
        """Sample time increments from a given waitingtime distribution.

        Either sample a number `N` of increments,
        or generate increments that sum up to the final time `T`.
        """

        if numpy.isfinite(N) and numpy.isfinite(T):

            raise ValueError("T and N cannot both be set!")

        elif numpy.isfinite(N):

            return wait.sample(N)

        elif numpy.isfinite(T):

            # start from crude estimate of the number of increments
            # gradually increase the number of increments until final time is reached

            N = int(T / wait.scale)
            inc = int(N * 0.3) + 1

            dt = wait.sample(N + inc)
            T_inc = dt.sum()
            while T_inc < T:
                dt_inc = wait.sample(inc)
                T_inc += dt_inc.sum()
                dt = numpy.concatenate((dt, dt_inc))

            idx = numpy.count_nonzero(numpy.cumsum(dt) <= T) - 1
            idx += 1  # add additional increment (also the last particle location is associated with a waiting time)
            return dt[:idx + 1]

        else:

            raise ValueError("Either T or N must be set!")


class Jump(object):
    """Static methods for parameterizing, instantiating and sampling from
    stable jump distributions.

    The standard jump distributions are Lévy-stable distributions.
    For certain parameter configurations normal (`alpha=2.0`) and
    Cauchy (`alpha=1.0`) distributions can be used.
    """

    @staticmethod
    def jump_scale(delta_t=1.0, DC=1.0, alpha=2.0, dimensions=1):
        """Calculate the scale of jump distributions."""

        return delta_t ** (1.0 / alpha) * DC ** (1.0 / dimensions)

    @staticmethod
    def jump_setup(delta_t=1.0, DC=1.0, alpha=2.0, dimensions=1):
        """Return instantiated jump distributions."""

        if delta_t <= 0.0:
            raise ValueError

        if DC <= 0.0:
            raise ValueError

        scale = Jump.jump_scale(delta_t, DC, alpha, dimensions)

        if dimensions == 1:
            if alpha > 2.0:
                raise NotImplementedError
            elif alpha == 2.0:
                # NOTE the normal distribution has a different parameterization!
                return distributions.Normal(scale=scale * numpy.sqrt(2.0))
            elif alpha <= 0.0:
                raise ValueError
            else:
                return distributions.Stable(alpha=alpha, scale=scale)
        elif dimensions == 2:
            if alpha > 2.0:
                raise NotImplementedError
            elif alpha == 2.0:
                # NOTE the normal distribution has a different parameterization!
                return distributions.Normal2D(scale=scale * numpy.sqrt(2.0))
            elif alpha == 1.0:
                return distributions.Cauchy2D(scale=scale)
            elif alpha <= 0.0:
                raise ValueError
            else:
                return distributions.Stable2D(alpha=alpha, scale=scale)
        else:
            raise NotImplementedError

    @staticmethod
    def sample_dx(jump, N):
        """Sample spatial increments from a given jump distribution."""

        return jump.sample(N)


class CTRW(Jump, Wait):
    """Implementation of continuous time random walks."""

    def __init__(self, delta_t=1.0, alpha=2.0, beta=1.0, DC=None, TC=None, dimensions=2, jump=None, wait=None):
        """The standard jump and waitingtime distributions can be replaced
        by setting the `jump` and `wait` arguments. In this case sampled
        CTRWs do not necessarily simulate the fractional diffusion equation!
        """

        self.dimensions = dimensions
        self.delta_t = delta_t
        self.alpha = alpha
        self.beta = beta

        # time and diffusion constants

        if TC is None:
            self.TC = 1.0
        else:
            self.TC = TC

        if DC is None:
            self.DC = self.TC ** (self.beta / self.alpha)
        else:
            self.DC = DC

        # waiting times

        if isinstance(wait, distributions.base.Distribution):
            # NOTE for generic waiting times analytical results may not hold
            self.wait = wait
        else:
            self.wait = self.wait_setup(self.delta_t, self.TC, self.beta)

        if self.beta == 0.0:
            # fixed step scenario corresponds to exponential waitingtimes
            self.beta = 1.0

        # jumps

        if isinstance(jump, distributions.base.Distribution):
            # NOTE for generic jumps analytical results may not hold
            assert self.dimensions == 1
            self.jump = jump
        elif isinstance(jump, distributions.base.BivariateDistribution):
            # NOTE for generic jumps analytical results may not hold
            assert self.dimensions == 2
            self.jump = jump
        else:
            self.jump = self.jump_setup(self.delta_t, self.DC, self.alpha, self.dimensions)

    def __repr__(self):

        return "CTRW(alpha={}, beta={}, DC={}, TC={}, delta_t={}, dimensions={})".format(
            self.alpha, self.beta, self.DC, self.TC, self.delta_t, self.dimensions)

    def __str__(self):

        return "CTRW($\\alpha={:0.02f}, \\beta={:0.02f}, C_{{D}}={:0.02f}, C_{{T}}={:0.02f}, \Delta t={:0.02f}, d={})$".format(
            self.alpha, self.beta, self.DC, self.TC, self.delta_t, self.dimensions)

    def _sample(self, T=numpy.inf, N=numpy.inf):

        dt = self.sample_dt(self.wait, T, N)
        dx = self.sample_dx(self.jump, len(dt))

        if self.dimensions == 1:
            dx = numpy.atleast_1d(dx)
        elif self.dimensions == 2:
            dx = numpy.atleast_2d(dx)

        return Particle(dt=dt, dx=dx)

    def sample(self, T=numpy.inf, N=numpy.inf, size=1):
        """Sample an ensemble of CTRW trajectories.

        The number of trajectories is determined by `size`.
        Use either `T` to set the final time or
        use `N` to explicitly determine the number of increments.
        """

        return [self._sample(T, N) for i in range(size)]
