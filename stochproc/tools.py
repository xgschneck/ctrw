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


def sample_average_displacement(particles, t, moment=2.0):

    return numpy.sum([p.get_displacement(t, moment=moment) for p in particles], axis=0) / len(particles)


def time_average_displacement(particles, Dt, T=None, moment=2.0):

    return numpy.sum([p.get_time_average_displacement(Dt, T=T, moment=moment) for p in particles], axis=0) / len(particles)


def velocity_autocorrelation(particles, Dt, T=None, version=1):

    return numpy.sum([p.get_velocity_autocorrelation(Dt, T=T, version=version) for p in particles], axis=0) / len(particles)


def sample_histogram_2d(particles, T, bins=None):

    if not numpy.isscalar(T):
        raise ValueError

    pos = numpy.array([p.get_displacement(T) for p in particles])

    if bins is not None:
        bins = numpy.concatenate(([-numpy.inf], bins, [numpy.inf]))

    hist, xbins, ybins = numpy.histogram2d(*pos.T, bins=bins, density=True)

    # NOTE hist is in units of [bin_count / sample_count / bin_area] if density=True
    # to sum up to 1.0, multiply with bin_area
    # hist *= dx * dx

    # center bin locations
    xbins = (xbins[1:] + xbins[:-1]) * 0.5
    ybins = (ybins[1:] + ybins[:-1]) * 0.5

    # remove infinite bin locations
    xbins = xbins[1:-1]
    ybins = ybins[1:-1]
    hist = hist[1:-1, :]
    hist = hist[:, 1:-1]

    return xbins, ybins, hist


def sample_histogram_1d(particles, T, bins=None):

    if not numpy.isscalar(T):
        raise ValueError

    pos = [particle.get_displacement(T) for particle in particles]

    if bins is not None:
        bins = numpy.concatenate(([-numpy.inf], bins, [numpy.inf]))

    hist, bins = numpy.histogram(pos, bins=bins, density=True)

    # NOTE hist is in units of [bin_count / sample_count / bin_area] if density=True
    # to get normalization (sum == 1.0), multiply with bin_area or bin_size
    # dx = bins[2] - bins[1]
    # hist *= dx

    # center bin locations
    bins = (bins[1:] + bins[:-1]) * 0.5

    # remove infinite bin locations
    bins = bins[1:-1]
    hist = hist[1:-1]

    return bins, hist


def local_time_histogram_2d(particles, bins, T=numpy.inf):

    return numpy.sum([p.get_local_time_histogram_2d(bins, T=T) for p in particles], axis=0) / len(particles)


def local_time_histogram_distance(particles, bins, T=numpy.inf):

    return numpy.sum([p.get_local_time_histogram_distance(bins, T=T) for p in particles], axis=0) / len(particles)


def local_time_moment(particles, T, moment=2.0):

    return numpy.sum([p.get_local_time_moment(moment=moment, T=T) for p in particles], axis=0) / len(particles)


def spatial_increments_histogram(particles, bins):

    return numpy.sum([p.get_increments_histogram(bins) for p in particles], axis=0) / len(particles)
