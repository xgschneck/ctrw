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


class Distribution(object):

    @classmethod
    def test(cls, logplot=False, **kwargs):

        distribution = cls(**kwargs)
        distribution._test_visual(logplot=logplot)


class ContinuousDistribution(Distribution):

    def _test_visual(self, logplot=False):

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(5, 4))
        ax = fig.gca()

        samples = self.sample(100000)

        if logplot:
            minimum = max(self._plot_support[0], 1e-6)
            bins = numpy.logspace(numpy.log10(minimum), numpy.log10(self._plot_support[1]), 201)
        else:
            bins = numpy.linspace(*self._plot_support, 201)

        bins = numpy.concatenate(([-numpy.inf], bins, [numpy.inf]))
        dens, bins = numpy.histogram(samples, bins=bins, density=True)
        bins = bins[1:-1]
        dens = dens[1:-1]
        bins = (bins[1:] + bins[:-1]) * 0.5
        ax.plot(bins, dens, linestyle="-", label="histogram")

        pdf = self.pdf(bins)
        ax.plot(bins, pdf, linestyle="--", label="density")

        if logplot:
            ax.set(xscale="log", yscale="log")

        ax.set_title(str(self))
        ax.legend()

        fig.savefig("results/distributions_visual_test_{}.pdf".format(self.__class__.__name__.lower()))


class DiscreteDistribution(Distribution):

    def _test_visual(self, logplot=False):

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(5, 4))
        ax = fig.gca()

        samples = self.sample(10000)

        bins = numpy.concatenate((numpy.arange(*self._plot_support), [numpy.inf]))
        dens, bins = numpy.histogram(samples, bins=bins, density=True)
        bins = bins[0:-1]
        ax.plot(bins, dens, linestyle="-", marker="o", label="histogram")

        pdf = self.pdf(bins)
        ax.plot(bins, pdf, linestyle="--", marker="x", label="mass")

        if logplot:
            ax.set(xscale="log", yscale="log")

        ax.set_title(str(self))
        ax.legend()

        fig.savefig("results/distributions_visual_test_{}.pdf".format(self.__class__.__name__.lower()))


class BivariateDistribution(Distribution):

    def _test_visual(self, ax=None, logplot=False):

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 4))
        ax1, ax2, ax3 = fig.subplots(1, 3)

        samples = self.sample(10000)
        # samples = numpy.array([(-7, -7)] * 10)

        nbins = 20
        vmax = 0.02

        bins = numpy.concatenate(([-numpy.inf], numpy.linspace(*self._plot_support, nbins), [numpy.inf]))
        hist, _, _ = numpy.histogram2d(*samples.T, (bins, bins), density=True)
        hist = hist[1:-1, 1:-1]
        bins = 0.5 * (bins[1:-2] + bins[2:-1])
        dx = bins[2] - bins[1]
        extent = [self._plot_support[0] - 0.5 * dx, self._plot_support[1] + 0.5 * dx] * 2
        ax1.imshow(hist, extent=extent, vmin=0.0, vmax=vmax)
        ax1.set_title("histogram")

        from .tools import evaluate_quadrant

        x, y = numpy.meshgrid(bins, bins)
        r = numpy.sqrt(x * x + y * y)
        pdf = evaluate_quadrant(self.pdf_distance, r)

        # pdf2 = self.pdf(x, y)
        # assert numpy.allclose(pdf, pdf2)

        # assert numpy.isclose(numpy.std([
        #     self.pdf(0.0, 0.0),
        #     self.pdf_distance(0.0),
        #     self.pdf(1e-10, 1e-10),
        #     self.pdf_distance(1e-10)]), 0.0)

        ax2.imshow(pdf, extent=extent, vmin=0.0, vmax=vmax)
        ax2.set_title("density")

        ax3.imshow(pdf - hist, extent=extent, cmap="RdYlBu", vmin=-0.1, vmax=0.1)
        ax3.set_title("density - histogram")

        plt.savefig("results/distributions_visual_test_{}.pdf".format(self.__class__.__name__.lower()))
