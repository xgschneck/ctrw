import numpy

import stochproc
import distributions


def displacement():

    alpha = 1.5
    beta = 0.6
    delta_t = 1.0
    dimensions = 2
    T = 1e4
    S = int(1e3)

    sp = stochproc.FractionalDiffusion(alpha=alpha, beta=beta, delta_t=delta_t, DC=1.0, TC=1.0, dimensions=dimensions)
    particles = sp.sample(T=T, size=S)
    moment = alpha

    t = numpy.logspace(1, numpy.log10(T), 100)
    sad = stochproc.sample_average_displacement(particles, t, moment=moment)
    sad_c, sad_e = distributions.fit_powerlaw(t, sad, None, None, method="opti3")
    sad_C, sad_E = sp.sample_average_displacement_constants(moment=moment)

    Dt = numpy.logspace(1, numpy.log10(T) - 1, 10)
    tad = stochproc.time_average_displacement(particles[:100], Dt, T=T, moment=moment)
    tad_c, tad_e = distributions.fit_powerlaw(Dt, tad, None, None, method="opti3")
    TAD = sp.time_average_displacement(T, Dt, moment=moment)

    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(t, sad, linestyle="-", color="C0", label="sample average")
    ax.plot(t, sad_c * t ** sad_e, linestyle="--", color="C0", label="sample average fitted ${:0.02f} \cdot t^{{{:0.02f}}}$".format(sad_c, sad_e))
    ax.plot(t, sad_C * t ** sad_E, linestyle=":", color="C0", label="sample average expected $\sim t^{{{:0.02f}}}$".format(sad_E))
    ax.plot(Dt, tad, linestyle="-", color="C1", label="time average")
    ax.plot(Dt, tad_c * Dt ** tad_e, linestyle="--", color="C1", label="time average fitted ${:0.02f} \cdot t^{{{:0.02f}}}$".format(tad_c, tad_e))
    ax.plot(Dt, TAD, linestyle=":", color="C1", label="time average expected $\sim \Delta$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(str(sp))
    ax.set_xlabel("$t$ or $\Delta$")
    ax.set_ylabel("$\langle |x|^{{\\alpha}}(t) \\rangle$ or $\langle \overline{{\delta^{{\\alpha}}}} \\rangle$")
    ax.legend()

    fig.savefig("results/ctrw_displacement.pdf")
