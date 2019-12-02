import numpy
from matplotlib import pyplot as plt

import transforms


class Cosine(object):

    a = 0.1

    def f(self, x):
        return numpy.cos(self.a * 2.0 * numpy.pi * x)

    def F(self, k):
        Fk = numpy.zeros(k.shape)
        Fk[numpy.isclose(k, self.a * 2.0 * numpy.pi, atol=0.1)] = numpy.pi
        Fk[numpy.isclose(k, -self.a * 2.0 * numpy.pi, atol=0.1)] = numpy.pi
        return Fk


class Gaussian(object):

    a = 23.484

    def f(self, x):
        return numpy.exp(- self.a * x**2)

    def F(self, k):
        return numpy.sqrt(numpy.pi / self.a) * numpy.exp(- k**2 / 4.0 / self.a)


class Gaussian2(object):

    sigma = 3.484

    def f(self, x):
        return 1.0 / numpy.sqrt(2.0 * numpy.pi) / self.sigma * numpy.exp(- 1.0 / 2.0 / self.sigma**2 * x**2)

    def F(self, k):
        return numpy.exp(- 1.0 / 2.0 * (self.sigma * k)**2)


class Cauchy(object):

    scale = 1.4

    def f(self, x):
        return 1.0 / numpy.pi / self.scale / (1 + (x / self.scale)**2)

    def F(self, k):
        return numpy.exp(- self.scale * numpy.abs(k))


def test_fourier(example):

    # non-unitary, angular frequency

    X = 50.0
    N = 2**12

    x = numpy.linspace(-X, X, N)

    fx = example.f(x)

    k, Fk = transforms.fourier(x, fx)
    assert numpy.allclose(Fk.imag, 0.0, atol=1e-1)
    # Fk = Fk.real

    xi, fxi = transforms.inverse_fourier(k, Fk)
    assert numpy.allclose(fxi.imag, 0.0, atol=1e-6)
    # fxi = fxi.real

    Fka = example.F(k)

    xia, fxia = transforms.inverse_fourier(k, Fka)
    assert numpy.allclose(fxia.imag, 0.0, atol=1e-6)
    # fxia = fxia.real

    fig = plt.figure(figsize=(8, 4))
    ax1, ax2 = fig.subplots(2, 1)

    ax1.plot(xi, fxi, label="ifft(fft(true))")
    ax1.plot(xia, fxia, label="ifft(true)")
    ax1.plot(x, fx, ":", label="true")

    ax2.plot(k, Fk, label="fft(true)")
    ax2.plot(k, Fka, ":", label="true")

    ax1.legend()
    ax2.legend()

    fig.savefig("results/transforms_fourier_example_{}.pdf".format(example.__class__.__name__))


def test_fourier_distributions_univariate():

    import distributions

    # dist = distributions.Rayleigh(scale=3.4); semi = True
    # dist = distributions.Stable(scale=0.12, alpha=1.9, beta=0.0); semi = False
    dist = distributions.Normal(loc=0.0, scale=3.484); semi = False

    X = 100.0
    N = 2**12
    if semi:
        x = numpy.linspace(0.0, X, N)
    else:
        x = numpy.linspace(-X, X, N)

    fx = dist.pdf(x)

    k, Fk = transforms.fourier(x, fx, semi=semi)
    Fk = Fk.real

    xi, fxi = transforms.inverse_fourier(k, Fk, semi=semi)
    fxi = fxi.real

    phi = dist.cf(k)
    phi = phi.real
    phi[~numpy.isfinite(phi)] = 0.0

    xia, fxia = transforms.inverse_fourier(k, phi, semi=semi)

    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1)

    ax1.plot(xi, fxi, "-", label="ifft(fft(PDF))")
    ax1.plot(xia, fxia, "-", label="ifft(CF)")
    ax1.plot(x, fx, ":", label="PDF")

    ax2.plot(k, Fk, "-", label="fft(PDF)")
    ax2.plot(k, phi, ":", label="CF")

    ax1.legend()
    ax2.legend()

    fig.savefig("results/transforms_fourier_probability_distributions_univariate.pdf")


def test_fourier_distributions_bivariate():

    import distributions

    # dist = distributions.Normal2D(scale=0.5); N = 2**10
    dist = distributions.Cauchy2D(scale=0.1); N = 2**10
    # dist = distributions.Stable2D(scale=0.1, alpha=1.5); N = 2**4

    X = 10.0

    x = numpy.linspace(-X, X, N)
    xy = numpy.meshgrid(x, x)

    # PDF

    f = dist.pdf(*xy)

    # FFT of PDF

    uv, F = transforms.fourier_2d(*xy, f)
    F = F.real

    # CF

    Fa = dist.cf(*uv)

    # IFFT of CF

    xy_, f_ = transforms.inverse_fourier_2d(*uv, Fa)

    # Hankel transform of radial PDF

    k = uv[0][0, :]
    _k = numpy.abs(k)
    Fk = transforms.hankel_functional(dist.pdf_distance, _k) * 2.0 * numpy.pi

    # inverse Hankel of radial CF

    r = xy[0][0, :]
    _r = numpy.abs(r)
    fr = transforms.inverse_hankel_functional(dist.cf_distance, _r) / 2.0 / numpy.pi

    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.4)
    ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = fig.subplots(3, 2)

    im = ax1.imshow(f, extent=(-X, X, -X, X))
    ax1.set_title("PDF")
    plt.colorbar(im, ax=ax1)

    im = ax2.imshow(Fa, extent=(uv[0][0, 0], uv[0][0, -1], uv[0][0, 0], uv[0][0, -1]), clim=(0.0, 2.0))
    ax2.set_title("CF")
    plt.colorbar(im, ax=ax2)

    im = ax3.imshow(numpy.absolute(F), extent=(uv[0][0, 0], uv[0][0, -1], uv[0][0, 0], uv[0][0, -1]), clim=(0.0, 2.0))
    ax3.set_title("fft(PDF)")
    plt.colorbar(im, ax=ax3)

    im = ax4.imshow(numpy.absolute(F) - Fa, extent=(uv[0][0, 0], uv[0][0, -1], uv[0][0, 0], uv[0][0, -1]))
    ax4.set_title("fft(PDF) - CF")
    plt.colorbar(im, ax=ax4)

    ax5.plot(uv[0][0, :], Fa[N // 2, :], label="CF")
    ax5.plot(uv[0][0, :], F[N // 2, :], label="fft(PDF)", linestyle="--")
    ax5.set_title("diff (error={:0.05f})".format(numpy.absolute(F - Fa).max()))
    ax5.plot(k, Fk, label="hankel(PDF_distance)", linestyle=":")
    ax5.legend()
    ax5.set_ylim(0.0, 1.5)

    ax6.plot(xy[0][0, :], f[N // 2, :], label="PDF")
    ax6.plot(xy_[0][0, :], f_[N // 2, :], label="ifft(CF)", linestyle="--")
    ax6.set_title("diff (error={:0.05f})".format(numpy.absolute(f - f_).max()))
    ax6.plot(r, fr, label="ihankel(CF_distance)", linestyle=":")
    ax6.legend()
    ax6.set_ylim(-0.1, 2.0)

    fig.savefig("results/transforms_fourier_probability_distributions_bivariate.pdf")


def test_hankel():

    import mpmath
    import scipy.special

    z = 3.836

    def f(r):
        return 1.0 / (z * z + r * r)

    def F(k):

        if type(k) == mpmath.mpc or type(k) == mpmath.mpf:
            return mpmath.besselk(0, k * z)
        else:
            return scipy.special.k0(numpy.abs(k) * z)

    r = numpy.linspace(0.00001, 1.0, 100)
    k = r

    fr1 = [transforms.inverse_hankel_functional(F, _r) for _r in r]
    fr2 = [f(_r) for _r in r]

    Fk1 = [transforms.hankel_functional(f, _k) for _k in k]
    Fk2 = [F(_k) for _k in k]

    fig = plt.figure(figsize=(10, 10))
    (ax1, ax2, ax3) = fig.subplots(3, 1)

    ax1.plot(k, Fk1, label="hankel transform")
    ax1.plot(k, Fk2, label="hankel true")
    ax1.plot(k, numpy.exp(-k), linestyle=":", label="exp")
    ax1.plot(k, -numpy.log(k), linestyle=":", label="log")

    ax2.plot(k, fr1, label="inv hankel transform")
    ax2.plot(k, fr2, label="inv hankel true")

    ax1.legend()
    ax2.legend()

    t = numpy.linspace(0.0001, 10.0, 100)
    LiFk2a = [mpmath.invertlaplace(F, _t, method="stehfest") for _t in t]
    # LiFk2b = [mpmath.invertlaplace(F, _t, method="talbot") for _t in t]
    LiFk2c = [mpmath.invertlaplace(F, _t, method="dehoog") for _t in t]

    ax3.plot(t, LiFk2a, label="stehfest")
    # ax3.plot(t, LiFk2b, label="talbot")
    ax3.plot(t, LiFk2c, label="dehoog")

    ax3.legend()

    fig.savefig("results/transforms_hankel_example.pdf")
