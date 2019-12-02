import numpy
from matplotlib import pyplot as plt

import stochproc


def velocity_autocorrelation():

    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca()

    alpha = 1.3
    beta = 1.0

    T0 = 1e-2
    T = 1e6
    S = int(1e1)

    delta_ts = (0.1, 1.0, 10.0)
    linestyles = ("-", "--", ":")

    Dt = numpy.logspace(numpy.log10(T0), numpy.log10(T) - 1, 10)

    for delta_t, linestyle in zip(delta_ts, linestyles):

        vac1 = numpy.zeros(Dt.shape)
        vac2 = numpy.zeros(Dt.shape)
        for s in range(S):
            sp = stochproc.CTRW(alpha=alpha, beta=beta, delta_t=delta_t, DC=1.0, dimensions=2)
            particle = sp.sample(T=T * delta_t, size=1)[0]
            _vac1 = particle.get_velocity_autocorrelation(Dt * delta_t, T=None, integration_steps=1000, version=1)
            _vac2 = particle.get_velocity_autocorrelation(Dt * delta_t, T=None, integration_steps=1000, version=2)
            del sp
            vac1 += _vac1
            vac2 += _vac2
        vac1 /= S
        vac2 /= S

        ax.plot(Dt, vac1, color="C0", linestyle=linestyle, label="VAC$_{{1}}, \Delta t = {:0.02f}$".format(delta_t))
        ax.plot(Dt, vac2, color="C1", linestyle=linestyle, label="VAC$_{{2}}, \Delta t = {:0.02f}$".format(delta_t))

    ax.set_xscale("log")
    ax.legend()

    fig.savefig("results/ctrw_velocity_autocorrelation.pdf")
