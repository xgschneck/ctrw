import numpy
from matplotlib import pyplot as plt

import distributions


def geometric_stability(name):

    S = 10000
    p = 0.01

    delta = 3.0
    kappa1 = 1.0
    beta = 0.5

    if name == "laplace":
        increment = distributions.SkewLaplace(scale=delta * numpy.sqrt(p), skewness1=kappa1 * p)
        dist = distributions.SkewLaplace(scale=delta, skewness1=kappa1)
    elif name == "mittagleffler":
        increment = distributions.MittagLeffler(beta=beta, scale=delta * p ** (1.0 / beta))
        dist = distributions.MittagLeffler(beta=beta, scale=delta)
    else:
        raise ValueError

    Ns = distributions.Geometric(p=p).sample(size=S)
    samples = numpy.array([increment.sample(size=N).sum() for N in Ns])

    bins = numpy.linspace(-20, 20, 50)
    density, bins = numpy.histogram(samples, bins=bins, density=True)
    bins = (bins[1:] + bins[:-1]) * 0.5

    fig = plt.figure(figsize=(8, 4))
    ax = fig.gca()

    ax.plot(bins, density, label="geometric sum " + str(increment))
    ax.plot(bins, dist.pdf(bins), label=str(dist))

    plt.legend()

    plt.savefig("results/distributions_{}_geometric_stability.pdf".format(name))
