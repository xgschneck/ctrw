import numpy
from matplotlib import pyplot as plt

import transforms


def test_laplace():

    a = 2.3

    def f(t):
        return numpy.cos(a * t)

    def F(s):
        return s / (a * a + s * s)

    x = numpy.linspace(0.0, 100.0, 10000)
    fx = f(x)

    t = numpy.linspace(0.0, 10.0, 150)
    Ft = F(t)

    Ft_num = transforms.laplace(x, fx, t)

    xx = numpy.linspace(0.01, 10.0, 100)
    fxx = transforms.inverse_laplace_functional(F, xx)

    Ft_num2 = transforms.laplace(xx, fxx, t)

    fig = plt.figure(figsize=(10, 5))
    (ax1, ax2) = fig.subplots(1, 2)

    ax1.plot(x, fx, label="f")
    ax1.plot(xx, fxx, label="invlaplace(F)")
    ax1.legend()
    ax1.set_xlim(0.0, 10.0)

    ax2.plot(t, Ft, label="F")
    ax2.plot(t, Ft_num, label="laplace(f)")
    ax2.plot(t, Ft_num2, label="laplace(invlaplace(F))")
    ax2.legend()

    fig.savefig("results/transforms_laplace_example.pdf")
