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


class Particle(object):
    """Represents a simulated particle/trajectory."""

    def __init__(self, dt=None, dx=None, t=None, x=None):
        """Usually the increments `dt` and `dx` are sampled from
        waitingtime and jump distributions.
        """

        # TODO pass start location to particle constructor ?

        if t is None and dt is None:
            raise ValueError("t and dt cannot both be None")
        elif t is None:
            self.t = numpy.concatenate(([0.0], numpy.cumsum(dt[:-1], axis=0)), axis=0)
            self.dt = dt
        elif dt is None:
            self.t = t
            self.dt = numpy.diff(t, axis=0)
        else:
            # TODO consistency
            self.t = t
            self.dt = dt

        if x is None and dx is None:
            raise ValueError("x and dx cannot both be None")
        elif x is None:
            self.dim = dx.ndim
            if self.dim == 1:
                x0 = numpy.zeros((1,))
            else:
                x0 = numpy.zeros((1, self.dim))
            self.x = numpy.concatenate((x0, numpy.cumsum(dx[:-1], axis=0)), axis=0)
            self.dx = dx
        elif dx is None:
            self.dim = x.ndim
            self.x = x
            self.dx = numpy.diff(x, axis=0)
        else:
            # TODO consistency
            self.dim = dx.ndim
            self.x = x
            self.dx = dx

        assert self.dim == 1 or self.dim == 2
        assert len(self.t) == len(self.dt) == len(self.x) == len(self.dx)

        # calculate square displacement
        if self.dim == 2:
            x = self.x - self.x[0]
            self.r = x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]

    def get_displacement(self, t, moment=None):
        """Calculate fractional displacement at time t.

        If `moment=None`, returns the actual displacement.
        Else, is a generalization of squared displacement.
        """

        if numpy.isscalar(t):
            idx = numpy.count_nonzero(self.t <= t) - 1
        else:
            idx = numpy.searchsorted(self.t, t, side="right") - 1

        if moment is None:
            return self.x[idx] - self.x[0]
        elif self.dim == 1:
            return numpy.abs(self.x[idx]) ** moment
        else:
            return self.r[idx] ** (0.5 * moment)

    def _get_time_average_displacement(self, Dt, T=None, moment=2.0, integration_steps=10000):

        if T is None:
            T = self.t[-1]

        # T ... maximum time
        # Dt ... interval length
        # di ... integration step

        di = T / float(integration_steps)

        ti_range = numpy.arange(0.0, T - Dt, di)
        integral_sum = 0.0

        if self.dim == 1:
            for ti in ti_range:
                idx = (ti <= self.t) * (self.t < ti + Dt)
                integral_sum += numpy.abs(self.dx[idx].sum(axis=0)) ** moment
        else:
            for ti in ti_range:
                idx = (ti <= self.t) * (self.t < ti + Dt)
                m = self.dx[idx].sum(axis=0)
                integral_sum += (m[0] * m[0] + m[1] * m[1]) ** (0.5 * moment)

        return integral_sum * di / (T - Dt)  # multiplied by integration step di

    def get_time_average_displacement(self, Dt, T=None, moment=2.0, integration_steps=10000):
        """Calculate the time average fractional displacement for given interval lengths Dt.

        See [C1], [C2], [C3] for details.
        Compare the parameters in `time_average_displacement`.
        """

        fun_get_time_average_displacement = numpy.vectorize(self._get_time_average_displacement, excluded=["T", "moment", "integration_steps"])
        return fun_get_time_average_displacement(Dt, T, moment, integration_steps)

    def _get_velocity_autocorrelation_version_1(self, Dt, T, integration_steps):

        tau = T / float(integration_steps)
        tau_n = numpy.arange(0.0, T - Dt, tau)

        # calculate I(tau*n), I(tau*n + Dt)
        I_tau_n_0 = numpy.searchsorted(self.t, tau_n, side="right") - 1
        I_tau_n_1 = numpy.searchsorted(self.t, tau_n + Dt, side="right") - 1

        integral_sum = 0.0

        if self.dim == 1:

            raise NotImplementedError  # normalization missing

            for idx1, idx2 in zip(I_tau_n_0, I_tau_n_1):
                integral_sum += (self.dx[idx1] * self.dx[idx2]) / (self.dt[idx1] * self.dt[idx2])

        else:

            for idx1, idx2 in zip(I_tau_n_0, I_tau_n_1):
                # step = numpy.dot(self.dx[idx1], self.dx[idx2]) / (self.dt[idx1] * self.dt[idx2])
                # step /= numpy.sqrt(numpy.dot(self.dx[idx1], self.dx[idx1]) / (self.dt[idx1] * self.dt[idx1]))
                # step /= numpy.sqrt(numpy.dot(self.dx[idx2], self.dx[idx2]) / (self.dt[idx2] * self.dt[idx2]))
                integral_sum += numpy.dot(self.dx[idx1], self.dx[idx2]) \
                    / numpy.sqrt(numpy.dot(self.dx[idx1], self.dx[idx1]) * numpy.dot(self.dx[idx2], self.dx[idx2]))

        return integral_sum * tau / (T - Dt)  # multiplied by integration step di

    def _get_velocity_autocorrelation_version_2(self, Dt, T, integration_steps):

        tau = T / float(integration_steps)
        tau = max(Dt / 10.0, T / float(integration_steps))

        tau_n = numpy.arange(0.0, T - Dt - Dt, tau)

        # calculate I(tau*n), I(tau*n + Dt), I(tau*n + 2*Dt)
        I_tau_n_0 = numpy.searchsorted(self.t, tau_n, side="right") - 1
        I_tau_n_1 = numpy.searchsorted(self.t, tau_n + Dt, side="right") - 1
        I_tau_n_2 = numpy.searchsorted(self.t, tau_n + Dt + Dt, side="right") - 1

        if self.dim == 1:

            raise NotImplementedError

        else:

            v = self.dx / self.dt.reshape(-1, 1)

            # TODO if Dt is a multiple of tau, both calculations can be combined

            v_tau_n_0 = (self.t[I_tau_n_0 + 1] - tau_n).reshape(-1, 1) * v[I_tau_n_0]
            # v_tau_n_0 += numpy.sum(self.dx[I_tau_n_0 + 1:I_tau_n_1 + 1], axis=0)
            for i, (i0, i1) in enumerate(zip(I_tau_n_0, I_tau_n_1)):
                v_tau_n_0[i] += numpy.sum(self.dx[i0 + 1:i1 + 1], axis=0)
            v_tau_n_0 -= (self.t[I_tau_n_1 + 1] - tau_n - Dt).reshape(-1, 1) * v[I_tau_n_1]
            v_tau_n_0 /= Dt
            v_tau_n_0 /= numpy.linalg.norm(v_tau_n_0, axis=1).reshape(-1, 1)

            v_tau_n_1 = (self.t[I_tau_n_1 + 1] - tau_n - Dt).reshape(-1, 1) * v[I_tau_n_1]
            # v_tau_n_1 += numpy.sum(self.dx[I_tau_n_1 + 1:I_tau_n_2 + 1], axis=0)
            for i, (i1, i2) in enumerate(zip(I_tau_n_1, I_tau_n_2)):
                v_tau_n_1[i] += numpy.sum(self.dx[i1 + 1:i2 + 1], axis=0)
            v_tau_n_1 -= (self.t[I_tau_n_2 + 1] - tau_n - Dt - Dt).reshape(-1, 1) * v[I_tau_n_2]
            v_tau_n_1 /= Dt
            v_tau_n_1 /= numpy.linalg.norm(v_tau_n_1, axis=1).reshape(-1, 1)

            return numpy.sum(v_tau_n_0[:, 0] * v_tau_n_1[:, 0] + v_tau_n_0[:, 1] * v_tau_n_1[:, 1]) * tau / (T - Dt - Dt)

    def _get_velocity_autocorrelation(self, Dt, T=None, integration_steps=10000, version=1):

        # T ... maximum time
        # Dt ... interval length
        # tau ... integration step

        if T is None:
            T = self.t[-1]

        if version == 1:
            return self._get_velocity_autocorrelation_version_1(Dt, T, integration_steps)
        elif version == 2:
            return self._get_velocity_autocorrelation_version_2(Dt, T, integration_steps)

    def get_velocity_autocorrelation(self, Dt, T=None, integration_steps=10000, version=1):
        """Calculate the normalized velocity autocorrelation of the trajectory.

        If `version=1`, uses the instantaneous velocity of particles,
        if `version=2`, uses an averaged form of velocity.
        """

        fun = numpy.vectorize(self._get_velocity_autocorrelation, excluded=["T", "integration_steps", "version"])
        return fun(Dt, T, integration_steps, version)

    def get_local_time_histogram_2d(self, bins, T=numpy.inf):
        """Calculate 2D local time profile at time T."""

        if not numpy.isscalar(T):
            raise ValueError

        if self.dim != 2:
            raise RuntimeError

        assert numpy.isclose(numpy.std(numpy.diff(bins[1:-1])), 0.0)  # equally spaced bins required

        idx = numpy.count_nonzero(self.t <= T) - 1

        weights = self.dt[:idx + 1]
        if numpy.isfinite(T):
            # the resting time in the last location of the trajectory may not be captured completely
            weights[idx] = T - self.t[idx]

        histogram, _, _ = numpy.histogram2d(*self.x[:idx + 1].T, bins=bins, weights=weights, density=False)

        # result shall be interpolation
        # dx = numpy.diff(bins)
        # dx2 = numpy.multiply(*numpy.meshgrid(dx, dx))
        # histogram /= dx2
        dx = bins[5] - bins[4]
        histogram /= dx * dx

        return histogram

    def get_local_time_histogram_distance(self, bins, T=numpy.inf):
        """Calculate radial local time profile at time T."""

        if not numpy.isscalar(T):
            raise ValueError

        idx = numpy.count_nonzero(self.t <= T) - 1

        if self.dim == 1:
            displacement = numpy.abs(self.x[:idx + 1])
        else:
            displacement = numpy.sqrt(self.r[:idx + 1])
        weights = self.dt[:idx + 1]

        if numpy.isfinite(T):
            # the resting time in the last location of the trajectory may not be captured completely
            weights[idx] *= (T - self.t[idx]) / self.dt[idx]

        histogram, _ = numpy.histogram(displacement.flat, bins=bins, weights=weights.flat, density=False)

        # result shall be interpolation
        dx = numpy.diff(bins)
        histogram /= dx

        return histogram

    def get_local_time_moment(self, T=numpy.inf, moment=2.0):
        """Calculate a central moment of the local time profile."""

        if numpy.isscalar(T):

            idx = numpy.count_nonzero(self.t <= T) - 1

            if self.sim == 1:
                m = self.dt[:idx + 1] * numpy.abs(self.x[:idx + 1]) ** moment
            else:
                m = self.dt[:idx + 1] * self.r[:idx + 1] ** (0.5 * moment)

            if numpy.isfinite(T):
                # the resting time in the last location of the trajectory may not be captured completely
                m[idx] *= (T - self.t[idx]) / self.dt[idx]

            return numpy.sum(m)

        else:

            idx = numpy.searchsorted(self.t, T, side="right") - 1
            idx_max = idx[-1]

            if self.dim == 1:
                m = self.dt[:idx_max + 1] * numpy.abs(self.x[:idx_max + 1]) ** moment
            else:
                m = self.dt[:idx_max + 1] * self.r[:idx_max + 1] ** (0.5 * moment)

            out = numpy.zeros(len(T))
            for i, (_idx, _t) in enumerate(zip(idx, T)):
                out[i] = numpy.sum(m[:_idx])
                out[i] += m[_idx] * (_t - self.t[_idx]) / self.dt[_idx]

            return out

    def get_increments_histogram(self, bins):
        """Calculate histogram of the absolute spatial increments."""

        if self.dim == 1:
            displacement = numpy.abs(self.dx)
        else:
            displacement = numpy.sqrt(self.dx[:, 0] * self.dx[:, 0] + self.dx[:, 1] * self.dx[:, 1])

        histogram, _ = numpy.histogram(displacement.flat, bins=bins, density=True)

        return histogram
