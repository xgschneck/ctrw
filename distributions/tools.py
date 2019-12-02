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


def fit_powerlaw(t, m, std_t=None, std_m=None, method="opti4", exponent=None):
    """Fit a powerlaw to given data points.

    `method` selects between different NumPy/SciPy optimization methods:
    least squares curve fitting, polynomial fit on log-log scale and
    orthogonal distance regression.

    Returns a coefficient and an exponent.
    """

    if method == "opti2":

        # least squares curve fitting

        from scipy.optimize import curve_fit

        if exponent is not None:

            def powerlaw(t, constant):
                return constant * t ** exponent

            opt, cov = curve_fit(powerlaw, t, m, p0=(1.0,), check_finite=True, method=None)
            constant = opt[0]

        else:

            def powerlaw(t, exponent, constant):
                return constant * t ** exponent

            opt, cov = curve_fit(powerlaw, t, m, p0=(1.0, 1.0), check_finite=True, method=None)
            exponent = opt[0]
            constant = opt[1]
            # error = numpy.sqrt(numpy.diag(cov))

    elif method == "opti3":

        # polynomial fit on loglog scale

        p = numpy.polyfit(numpy.log(t), numpy.log(m), 1)
        exponent = p[-2]
        constant = numpy.exp(p[-1])

    elif method == "opti4":

        # orthogonal distance regression

        from scipy.odr import ODR, Model, RealData

        def fcn(B, t):
            return B[1] * t ** B[0]

        def fjacb(B, t):
            return numpy.array((B[1] * t ** B[0] * numpy.log(t), t ** B[0]))

        def fjacd(B, t):
            return B[1] * B[0] * t ** (B[0] - 1.0)

        data = RealData(t, m, std_t, std_m)
        mdl = Model(fcn, fjacb, fjacd)

        odr = ODR(data, mdl, (1.0, 1.0))
        odr.set_job(fit_type=0, deriv=1)
        output = odr.run()

        exponent = output.beta[0]
        constant = output.beta[1]
        # error = (output.sum_square_delta, output.sum_square_eps)

    else:

        raise ValueError("unknown fitting method")

    return (constant, exponent)


def smoothing(t, m, smoothing_method="none", smoothing_partitions=100):
    """Algorithms for smoothing 1D trajectories.

    Either aggregate fixed size subsets of sample points
    or aggregate sample points from given interval lengths.
    """

    # sort
    idx = numpy.argsort(t)
    t = t[idx]
    m = m[idx]

    if smoothing_method is None or smoothing_method == "none":

        std_t = numpy.ones(m.shape)
        std_m = numpy.ones(m.shape)

    elif smoothing_method == "equally_spaced":

        # split data into junks with equal time span
        # each junk represents the same fraction of time

        dt = t[-1] / smoothing_partitions

        t_ = numpy.zeros(smoothing_partitions)
        m_ = numpy.zeros(smoothing_partitions)
        std_t = numpy.zeros(smoothing_partitions)
        std_m = numpy.zeros(smoothing_partitions)

        for i in range(smoothing_partitions):
            idx = (i * dt <= t) * (t < (i + 1) * dt)
            if (~idx).prod():
                t_[i] = t_[i - 1]
                m_[i] = m_[i - 1]
                std_t[i] = std_t[i - 1]
                std_m[i] = std_m[i - 1]
            else:
                t_[i] = t[idx].mean()
                m_[i] = m[idx].mean()
                std_t[i] = t[idx].std()
                std_m[i] = m[idx].std()

        t = t_
        m = m_

    elif smoothing_method == "equally_sized":

        # split data into equally sized junks
        # each containing the same number of observations

        t_ = numpy.array_split(t, smoothing_partitions)
        m_ = numpy.array_split(m, smoothing_partitions)
        t = numpy.array([array.mean() for array in t_])
        m = numpy.array([array.mean() for array in m_])
        std_t = numpy.array([array.std() for array in t_])
        std_m = numpy.array([array.std() for array in m_])

    else:

        raise ValueError("unknown smoothing method")

    return (t, m, std_t, std_m)


def evaluate_quadrant(function, points):

    lx, ly = points.shape

    assert lx == ly
    assert lx / 2.0 != lx // 2

    l = lx // 2 + 1

    out = function(points[:l, :l])
    out = numpy.concatenate((out, numpy.flip(out[:, :-1], axis=1)), axis=1)
    out = numpy.concatenate((out, numpy.flip(out[:-1, :], axis=0)), axis=0)
    return out
