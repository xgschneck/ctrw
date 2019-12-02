import numpy
from matplotlib import pyplot as plt

import stochproc


def green_function_1d():

    alpha = 1.8
    beta = 1.0
    delta_t = 1.0
    DC = 11.4
    TC = 5.0

    sp = stochproc.FractionalDiffusion(alpha=alpha, beta=beta, delta_t=delta_t, DC=DC, TC=TC, dimensions=1)

    T = 100.0
    R = 500.0
    N = 16  # evaluation points
    S = 10000  # trajectories
    method = "dft"
    # method = "lt"
    # method = "ft"
    # method = "direct"

    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()

    print("sampling trajectories ...")

    particles = sp.sample(T=T, size=S)

    print("calculating histogram ...")

    bins = numpy.linspace(-R, R, N)
    bins, hist = stochproc.sample_histogram_1d(particles, T, bins)

    print("evaluating green function ...")

    if beta == 1.0 or beta == 0.0:
        dens_direct = sp.green_function(T, numpy.abs(bins), method="direct")
    else:
        dens_direct = numpy.full(bins.shape, numpy.nan)

    dens_fft = sp.green_function(T, numpy.abs(bins), method="fft")

    if alpha == 2.0:
        dens_lt = sp.green_function(T, numpy.abs(bins), method="lt")
    else:
        dens_lt = numpy.full(bins.shape, numpy.nan)

    print("plotting ...")

    ax.plot(bins, hist, label="hist")
    ax.plot(bins, dens_direct, linestyle="-", label="density (direct)")
    ax.plot(bins, dens_fft, linestyle="--", label="density (fft)")
    ax.plot(bins, dens_lt, linestyle=":", label="density (lt)")
    ax.legend()
    ax.set_title(str(sp))

    plt.savefig("results/ctrw_green_function_1d.pdf")


def green_function_2d():

    alpha = 1.4
    beta = 1.0
    delta_t = 1.0
    DC = 3.7
    TC = 2.4

    T = 100.0   # simulation time
    R = 50.0    # domain radius
    N = 16      # number of evaluation points
    S = 10000   # number of trajectories

    sp = stochproc.FractionalDiffusion(alpha=alpha, beta=beta, delta_t=delta_t, DC=DC, TC=TC, dimensions=2)

    print("sampling trajectories ...")

    particles = sp.sample(T=T, size=S)

    print("calculating histogram ...")

    bins = numpy.linspace(-R, R, N)
    xbins, ybins, hist = stochproc.sample_histogram_2d(particles, T, bins)

    print("evaluating green function (Laplace transform) ...")

    X, Y = numpy.meshgrid(xbins, ybins)
    r = numpy.sqrt(X * X + Y * Y)

    if alpha == 2.0:
        dens_lt = sp.green_function(T, r, method="lt")
    else:
        dens_lt = numpy.full(r.shape, numpy.nan)

    print("evaluating green function (fast Fourier transform) ...")

    dens_fft = sp.green_function(T, r, method="dft")

    print("evaluating green function (direct) ...")

    if beta == 1.0 or beta == 0.0:

        from distributions.tools import evaluate_quadrant
        dens_dir = evaluate_quadrant(lambda r: sp.green_function(T, r, method="direct"), r)

    else:
        dens_dir = numpy.full(r.shape, numpy.nan)

    print("plotting ...")

    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
    (ax1, ax2, ax3, ax4), (axA, axB, axC, axD) = fig.subplots(2, 4)

    vmax = min(hist.max(), dens_lt.max(), dens_fft.max())

    im1 = ax1.imshow(hist, origin="lower", extent=(-R, R, -R, R), vmin=0.0, vmax=vmax)
    plt.colorbar(im1, ax=ax1)
    ax1.set_title("sample histogram")

    im2 = ax2.imshow(dens_lt, origin="lower", extent=(-R, R, -R, R), vmin=0.0, vmax=vmax)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title("Laplace transform")

    im3 = ax3.imshow(dens_fft, origin="lower", extent=(-R, R, -R, R), vmin=0.0, vmax=vmax)
    plt.colorbar(im3, ax=ax3)
    ax3.set_title("fast Fourier transform")

    im4 = ax4.imshow(dens_dir, origin="lower", extent=(-R, R, -R, R), vmin=0.0, vmax=vmax)
    plt.colorbar(im4, ax=ax4)
    ax4.set_title("stable density")

    axA.plot(xbins, hist[:, numpy.isclose(ybins, 0.0)], linestyle="-", label="sample histogram")
    axA.plot(xbins, dens_lt[:, numpy.isclose(ybins, 0.0)], linestyle="--", label="Laplace transform")
    axA.plot(xbins, dens_fft[:, numpy.isclose(ybins, 0.0)], linestyle=":", label="fast Fourier transform")
    axA.plot(xbins, dens_dir[:, numpy.isclose(ybins, 0.0)], linestyle="-", alpha=0.5, label="stable density")
    axA.set_title("radial section")

    axB.plot(xbins, hist[:, numpy.isclose(ybins, 0.0)], linestyle="-", label="sample histogram")
    axB.plot(xbins, dens_lt[:, numpy.isclose(ybins, 0.0)], linestyle="--", label="Laplace transform")
    axB.plot(xbins, dens_fft[:, numpy.isclose(ybins, 0.0)], linestyle=":", label="fast Fourier transform")
    axB.plot(xbins, dens_dir[:, numpy.isclose(ybins, 0.0)], linestyle="-", alpha=0.5, label="stable density")
    axB.set_yscale("log")
    axB.set_title("radial section")
    axB.legend(loc="lower left", bbox_to_anchor=(1.2, 0.0), borderaxespad=0)

    axC.remove()
    axD.remove()

    plt.savefig("results/ctrw_green_function_2d.pdf")
