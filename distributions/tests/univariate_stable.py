import numpy
from matplotlib import pyplot as plt

import distributions


def stable_discontinuity():

    # the skew univariate stable distribution has a parameter "discontinuity" when alpha -> 1.
    # numerical approximation is not accurate.

    fig = plt.figure(figsize=(8, 7))
    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95, wspace=None, hspace=None)
    ax1, ax2 = fig.subplots(2, 1)

    cmap = plt.cm.jet

    s = numpy.linspace(-5.0, 100.0, 100)

    alphas = [0.95, 0.98, 0.99, 1.0]

    for i, alpha in enumerate(alphas):

        dist = distributions.Stable_scipy(alpha=alpha, beta=1.0)
        color = cmap(i / len(alphas))
        ax1.plot(s, dist.pdf(s), color=color, label=str(dist))
        ax2.plot(s, dist.cf(s), color=color)

    ax1.set_title("density function")
    ax1.legend()
    ax2.set_title("characteristic function")
    ax2.set_yscale("log")
    fig.savefig("results/distributions_stable_parameter_discontinuity.pdf")


def stable_implementations():

    # compares the implementation of stable pdf from [D4] and [D5].

    alpha = 0.5 + numpy.random.random() * 1.5
    beta = numpy.random.random() * 2.0 - 1.0
    scale = numpy.random.random() * 0.6 + 0.7

    alpha = 0.6
    beta = 0.846
    scale = 10.6478

    sc = distributions.Stable_scipy(alpha=alpha, beta=beta, scale=scale)
    ex = distributions.Stable_extern(alpha=alpha, beta=beta, scale=scale)

    x = numpy.linspace(0.0, 200.0, 100)
    sc_pdf = sc.pdf(x)
    ex_pdf = ex.pdf(x)

    fig = plt.figure(figsize=(8, 4))
    ax1, ax2 = fig.subplots(1, 2)

    ax1.plot(x, sc_pdf, label="scipy")
    ax1.plot(x, ex_pdf, linestyle="--", label="external")

    ax2.plot(x, sc_pdf, label="scipy")
    ax2.plot(x, ex_pdf, linestyle="--", label="external")

    ax2.set_ylim(1e-5, 1e-1)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    fig.savefig("results/distributions_stable_implementations.pdf")


def stable_implementations_performance():

    import timeit

    N = 100

    alpha_ = numpy.random.random(N) * 1.5 + 0.5
    beta_ = numpy.random.random(N) * 2.0 - 1.0
    scale_ = numpy.random.random(N) * 0.6 + 0.7

    x = numpy.linspace(0.0, 50.0, 10)

    def test_scipy():
        for alpha, beta, scale in zip(alpha_, beta_, scale_):
            p = distributions.Stable_scipy(alpha=alpha, beta=beta, scale=scale)
            p.pdf(x)
            # p.sample(100)

    def test_extern():
        for alpha, beta, scale in zip(alpha_, beta_, scale_):
            p = distributions.Stable_extern(alpha=alpha, beta=beta, scale=scale)
            p.pdf(x)
            # p.sample(100)

    print("scipy ...")
    tt = timeit.timeit("test_scipy()", globals=locals(), number=1)
    print("scipy", tt)

    print("extern ...")
    tt = timeit.timeit("test_extern()", globals=locals(), number=1)
    print("extern", tt)


def stable_powerlaw():

    alpha = 0.5
    scale = 1.0

    dist = distributions.Stable_scipy(alpha=alpha, beta=0.0, scale=scale)

    R = 1000.0
    x = numpy.logspace(-2, numpy.log10(R), 100)
    x = numpy.concatenate((-x[::-1], x), axis=0)

    pdf = dist.pdf(x)

    _pdf = pdf[x > 1e1]
    _x = x[x > 1e1]
    c, e = distributions.fit_powerlaw(_x, _pdf)

    pdf_asymptotic = dist.pdf_asymptotic(_x)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.gca()
    ax.plot(x, pdf, label=str(dist))
    ax.plot(_x, c * _x ** e, linestyle="--", label="fitted ${:0.02f} \cdot x^{{{:0.02f}}}$".format(c, e))
    ax.plot(_x, pdf_asymptotic, linestyle=":", label="predicted asymptotic")

    ax.set_xlabel("$x$")
    ax.set_yscale("log")
    ax.set_xscale("symlog")
    ax.legend()

    fig.savefig("results/distributions_stable_asymptotic_powerlaw.pdf")
