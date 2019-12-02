import numpy
from matplotlib import pyplot as plt

import stochproc
import distributions


def waitingtimes():

    beta_list = [0.19, 0.57, 0.99]
    delta_t_list = [0.05, 0.77]

    rows = len(beta_list)
    cols = len(delta_t_list)

    from matplotlib import pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.95, wspace=0.0, hspace=1.0)
    axes = fig.subplots(rows, cols)

    for r in range(rows):
        for c in range(cols):

            ax = axes[r, c]
            ax.set_yscale('symlog', linthreshy=1e1, linscaley=2.0)
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.get_major_locator().set_params(numticks=3)
            ax.set_xlabel("operational time $t_{\\ast}$")

        for c in range(1, cols):

            ax = axes[r, c]
            ax.get_shared_y_axes().join(axes[r, 0], ax)
            ax.set_yticklabels([])
            ax.tick_params(axis="y", which="both", length=0)

        axes[r, 0].set_ylabel("time lag $t-t_{\\ast}$")

    def trajectories(ax, delta_t=1.0, beta=1.0):

        base_rate = 1.0

        scale = delta_t / base_rate
        constant = distributions.Constant(scale=scale)
        exponential = distributions.Exponential(scale=scale)

        scale = beta * delta_t ** (1.0 / beta) / base_rate
        lomax = distributions.Lomax(beta=beta, scale=scale)

        scale = 1.0 * delta_t ** (1.0 / beta) / base_rate
        mittagleffler = distributions.MittagLeffler(beta=beta, scale=scale)

        K = 100

        T = 500.0
        N = int(T / delta_t * base_rate)
        t = numpy.linspace(0.0, T, N)

        alpha_value = 0.02

        t_constant = numpy.cumsum(constant.sample(N))
        ax.plot(t, t_constant - t, label=str(constant), alpha=1.0, color="k")

        t_exponential_mean = numpy.zeros(t.shape)
        for k in range(K):
            t_exponential = numpy.cumsum(exponential.sample(N))
            ax.plot(t, t_exponential - t, alpha=alpha_value, color="C2")
            t_exponential_mean += t_exponential
        t_exponential_mean /= K
        ax.plot(t, t_exponential_mean - t, label=str(exponential), alpha=1.0, color="C2")

        t_lomax_mean = numpy.zeros(t.shape)
        for k in range(K):
            t_lomax = numpy.cumsum(lomax.sample(N))
            ax.plot(t, t_lomax - t, alpha=alpha_value, color="C0")
            t_lomax_mean += t_lomax
        t_lomax_mean /= K
        ax.plot(t, t_lomax_mean - t, label=str(lomax), alpha=1.0, color="C0")

        t_ml_mean = numpy.zeros(t.shape)
        for k in range(K):
            t_ml = numpy.cumsum(mittagleffler.sample(N))
            ax.plot(t, t_ml - t, alpha=alpha_value, color="C1")
            t_ml_mean += t_ml
        t_ml_mean /= K
        ax.plot(t, t_ml_mean - t, label=str(mittagleffler), alpha=1.0, color="C1")

        ax.set_title("$\Delta t = {}$".format(delta_t))
        ax.legend(bbox_to_anchor=(0.5, 0.0), loc=9, borderaxespad=4.0, ncol=2)

    for r, beta in enumerate(beta_list):
        for c, delta_t in enumerate(delta_t_list):
            trajectories(axes[r][c], delta_t=delta_t, beta=beta)

    plt.savefig("results/ctrw_waitingtimes.pdf")


def trajectories():

    T = 1e3
    list_delta_t = [0.05, 0.2, 1.0]
    list_alpha_beta = [(2.0, 1.0), (2.0, 0.5), (1.0, 1.0), (1.6, 0.8), (0.5, 0.8)]

    fontsize = 9

    def plot_trajectory(ax, particle, T):

        color_alpha = 1.0

        segments = numpy.transpose(numpy.stack((particle.x[:-1, :], particle.x[1:, :])), axes=(1, 0, 2))
        assert segments.shape[0] == len(particle.t) - 1

        norm = plt.Normalize(0.0, T)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', norm=norm, alpha=color_alpha)
        lc.set_array(particle.t)
        lc.set_linewidth(2)
        ax.add_collection(lc)

        ax.plot(particle.x[(0, -1), 0], particle.x[(0, -1), 1], "ro")

        x_min = min(ax.get_xlim()[0], particle.x[:, 0].min())
        x_max = max(ax.get_xlim()[1], particle.x[:, 0].max())
        x_len = 0.5 * (x_max - x_min)
        x_cen = 0.5 * (x_min + x_max)
        y_min = min(ax.get_ylim()[0], particle.x[:, 1].min())
        y_max = max(ax.get_ylim()[1], particle.x[:, 1].max())
        y_len = 0.5 * (y_max - y_min)
        y_cen = 0.5 * (y_min + y_max)
        xy_len = max(x_len, y_len)
        ax.set_xlim(x_cen - xy_len, x_cen + xy_len)
        ax.set_ylim(y_cen - xy_len, y_cen + xy_len)

        ax.locator_params(axis="both", tight=True, nbins=4)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.xaxis.offsetText.set_fontsize(fontsize)
        ax.yaxis.offsetText.set_fontsize(fontsize)
        ax.xaxis.offsetText.set_visible(False)
        ax.set_aspect(1, adjustable="box")

    cols = len(list_delta_t)
    rows = len(list_alpha_beta)

    fig = plt.figure(figsize=(10, 13))
    plt.subplots_adjust(left=0.15, bottom=0.03, right=0.85, top=0.95, wspace=0.2, hspace=0.25)
    axes = fig.subplots(rows, cols, sharey=False, sharex=False)

    for c, delta_t in enumerate(list_delta_t):
        for r, (alpha, beta) in enumerate(list_alpha_beta):
            ax = axes[r][c]
            sp = stochproc.CTRW(alpha=alpha, beta=beta, delta_t=delta_t)
            particle = sp.sample(T=T, size=1)[0]
            del sp
            plot_trajectory(ax, particle, T)

        axes[0][c].set_title("$\Delta t = {:0.02f}$\n".format(delta_t))

    for r, (alpha, beta) in enumerate(list_alpha_beta):
        ax = axes[r][0]
        ax.legend(bbox_to_anchor=(-0.25, 1.0), loc=1, borderaxespad=0.0, ncol=1,
                  title="$\\alpha={:0.02f}$\n$\\beta={:0.02f}$".format(alpha, beta))

    ax = fig.add_axes([0.9, 0.03, 0.02, 0.92])

    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    ColorbarBase(ax, norm=Normalize(vmin=0.0, vmax=T))

    ax.set_title("$t$\n")
    ax.tick_params(labelsize=fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize)

    plt.savefig("results/ctrw_trajectories.pdf")
