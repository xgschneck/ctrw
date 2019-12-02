import numpy
from matplotlib import pyplot as plt

import distributions


def lomax_powerlaw():

    dist = distributions.Lomax(beta=0.746, scale=3.485)

    t = numpy.logspace(-1, 4, 100)
    p = dist.pdf(t)
    a = dist.pdf_asymptotic(t)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca()

    ax.plot(t, p, linestyle="-", label="pdf")
    ax.plot(t, a, linestyle=":", label="$\\beta \\tau^{{\\beta}} t^{{ - (1 + \\beta)}}$")

    ax.set_title(str(dist))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig("results/distributions_lomax_asymptotic_powerlaw.pdf")


def mittagleffler_function_implementation():

    # compares the implementations of the Mittag-Leffler function
    # in [D2] (LIB) and [D3] (IMPL). See also [D1].

    from distributions import MittagLefflerFunction_Quad
    from ext_mittagleffler.mittag_leffler import ml

    x = numpy.linspace(-20.0, 10.0, 100)

    fig = plt.figure(figsize=(10, 4))
    axes = fig.subplots(1, 2)

    cmap = plt.get_cmap("tab10")

    for i, a in enumerate([0.1, 0.5, 1.0, 1.5, 2.0]):
        color = cmap(i)

        eps = numpy.finfo(float).eps
        mlf = MittagLefflerFunction_Quad(a, 1.0, eps)
        y = [mlf(_x) for _x in x]
        axes[0].plot(x, y, color=color, linestyle=":", linewidth=4, label="IMPL (a = {:0.02f})".format(a))
        axes[1].plot(x, y, color=color, linestyle=":", linewidth=4, label="IMPL (a = {:0.02f})".format(a))

        y = [ml(_x, a) for _x in x]
        axes[0].plot(x, y, color=color, linestyle="--", linewidth=2, label="LIB (a = {:0.02f})".format(a))
        axes[1].plot(x, y, color=color, linestyle="--", linewidth=2, label="LIB (a = {:0.02f})".format(a))

    axes[0].legend()
    axes[0].set_xlim(-20.0, 5.0)
    axes[0].set_ylim(-2.0, 5.0)

    axes[1].set_yscale("log")
    axes[1].set_xlim(0.0, 10.0)
    axes[1].set_ylim(1.0, 1e50)

    fig.savefig("results/distributions_mittagleffler_function_implementations.pdf")


def mittagleffler_powerlaw():

    beta = 0.78
    scale = 10

    dist = distributions.MittagLeffler(beta=beta, scale=scale)
    samples = dist.sample(100000)

    t = numpy.logspace(-3, 5, 1000)
    pdf = dist.pdf(t)
    pdf_asymptotic = dist.pdf_asymptotic(t)

    _pdf = pdf[t > 1e4]
    _t = t[t > 1e4]
    c, e = distributions.fit_powerlaw(_t, _pdf)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca()
    plt.hist(samples, bins=t, density=True, alpha=0.5, label="sample histogram")

    ax.plot(t, pdf, linestyle="-", label="pdf")
    ax.plot(t, pdf_asymptotic, linestyle="--", label="pdf asymptotic")

    ax.plot(t, c * t ** e, linestyle=":", label="${:0.02f} \cdot t^{{{:0.02f}}}$".format(c, e))

    ax.set_title(str(dist))
    ax.set(xscale="log", yscale="log")
    ax.set_ylim(1e-8, 5.0)
    ax.legend()

    fig.savefig("results/distributions_mittagleffler_asymptotic_powerlaw.pdf")


def mittagleffler_transition():

    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95, wspace=None, hspace=None)
    ax = fig.gca()

    x = numpy.logspace(-2, 7, 1000)

    rate = 0.4

    scale = 1.0 / rate
    exponential = distributions.Exponential(scale=scale)
    ax.plot(x, exponential.pdf(x), color="k", linestyle=":", label=str(exponential), linewidth=2.5)

    betas = [1.0, 0.95, 0.1]
    colors = ["C3", "C2", "C0"]

    for beta, color in zip(betas, colors):

        # calculate scale via expectation (only exists if beta > 1)
        scale = (beta - 1.0) / rate

        # via q-exponential
        q = (2.0 + beta) / (1.0 + beta)
        _scale = 1.0 / rate / (q - 1.0)
        scale = (1.0 + beta) / rate
        assert numpy.isclose(scale, _scale)

        # via generalized Pareto
        scale = beta / rate

        lomax = distributions.Lomax(beta=beta, scale=scale)
        ax.plot(x, lomax.pdf(x), color=color, linestyle="--", label=str(lomax), alpha=0.5)

        scale = 1.0 / rate
        mlf = distributions.MittagLeffler(beta=beta, scale=scale)
        ax.plot(x, mlf.pdf(x), color=color, linestyle="-", label=str(mlf))

    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e1)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$f(t)$")

    plt.savefig("results/distributions_mittagleffler_transition.pdf")


def mittagleffler_laplace_transform():

    import transforms

    fig = plt.figure(figsize=(10, 4))
    axes = fig.subplots(1, 2)

    t = numpy.linspace(0.0, 100.0, 100)

    beta = 0.78
    scale = 0.746

    dist = distributions.MittagLeffler(beta=beta, scale=scale)

    axes[0].plot(t, dist.pdf(t), label="dist")

    def Lapl_target(s):
        return 1.0 / (1.0 + (s * scale) ** beta)

    Lapl_target_inv = transforms.inverse_laplace_functional(Lapl_target, t[1:])
    axes[0].plot(t[1:], Lapl_target_inv, label="target inverse transf", linestyle="--")

    tt = numpy.linspace(0.001, 10.0, 1000)
    Lapl = transforms.laplace(tt, dist.pdf(tt), t)
    axes[1].plot(t, Lapl, label="dist transformed")

    axes[1].plot(t, Lapl_target(t), label="target", linestyle="--")

    axes[0].legend()
    axes[1].legend()

    plt.savefig("results/distributions_mittagleffler_laplace_transform.pdf")
