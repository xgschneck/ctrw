import numpy
from matplotlib import pyplot as plt

import distributions
import stochproc


def _plot(ax1, ax2, ax3, color, delta_t):

    alpha = 1.5
    beta = 0.7
    dimensions = 2
    T = 1e4
    S = int(1e3)

    sp = stochproc.FractionalDiffusion(alpha=alpha, beta=beta, delta_t=delta_t, DC=1.0, TC=1.0, dimensions=dimensions)
    particles = sp.sample(T=T, size=S)

    t = numpy.logspace(1, numpy.log10(T), 5)
    m = stochproc.local_time_moment(particles, t, moment=alpha / beta)
    c, e = distributions.fit_powerlaw(t, m, None, None, method="opti3")

    bins = numpy.linspace(0.0, 1000.0, 50)
    d = stochproc.local_time_histogram_distance(particles, bins, T=T)

    if dimensions == 2:
        bins2d = numpy.linspace(-100.0, 100.0, 51)
        histogram2d = stochproc.local_time_histogram_2d(particles, bins2d, T=T)
        bins2d = 0.5 * (bins2d[1:] + bins2d[:-1])

    ax1.plot(t, m, linestyle="-", color=color, label="$\Delta t = {:0.02f}$".format(delta_t))
    ax1.plot(t, c * t ** e, linestyle=":", color=color, label="fitted ${:0.02f} \cdot t^{{{:0.02f}}}$".format(c, e))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("local time moment")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$l_{\delta}^{m}(t)$")
    ax1.legend()

    ax2.plot(bins[:-1], d, color=color, label="$\Delta t = {:0.02f}$".format(delta_t))
    ax2.set_xscale("log")
    ax2.set_title("local time radial $T={}$".format(T))
    ax2.set_xlabel("$|\mathbf{x}|$")
    ax2.set_ylabel("$L_{\delta}(t=T,\mathbf{x})$")
    ax2.legend()

    if dimensions == 2:
        im = ax3.imshow(histogram2d, extent=(-100.0, 100.0, -100.0, 100.0), origin="lower", cmap=plt.cm.gist_yarg, vmin=0.0, vmax=3.0)
        plt.colorbar(im, ax=ax3)
        ax3.set_title("$\Delta t = {:0.02f}$\nlocal time $T={}$".format(delta_t, T))


def local_time():

    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.4)
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=False, sharey=False)

    _plot(ax1, ax2, ax3, "C0", 0.7)
    _plot(ax1, ax2, ax4, "C1", 2.3)

    fig.savefig("results/ctrw_local_time.pdf")
