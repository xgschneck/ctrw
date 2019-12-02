import numpy
from matplotlib import pyplot as plt

import distributions


def bivariate_stable_radial():

    fig = plt.figure(figsize=(8, 4))
    ax = fig.gca()

    cmap = plt.cm.viridis

    scale = 1.5

    numpoints = 50
    numpoints = 5
    r1 = numpy.logspace(-1, 0, 2 * numpoints)
    r2 = numpy.logspace(0, 3, numpoints)
    r = numpy.concatenate([r1, r2[1:]])

    # alphas = [0.1, 0.5, 1.0, 1.5, 1.75, 1.99, 2.0]
    alphas = [0.5, 1.0, 1.5, 2.0]

    for alpha in alphas:

        label = "$\\alpha={{{:0.02f}}}$".format(alpha)
        color = cmap(0.5 * (2.0 - alpha) * 0.8)
        if alpha == 1.99:
            linestyle = "--"
        else:
            linestyle = "-"

        if alpha == 1.0:
            distribution_cauchy = distributions.Cauchy2D(scale=scale)
            ax.plot(r, distribution_cauchy.pdf_radius(r), linewidth=2, label=label, color=color, linestyle=linestyle)

        elif alpha == 2.0:
            distribution_normal = distributions.Normal2D(scale=numpy.sqrt(2) * scale)
            ax.plot(r, distribution_normal.pdf_radius(r), linewidth=2, label=label, color=color, linestyle=linestyle)

        else:
            distribution_stable = distributions.Stable2D(scale=scale, alpha=alpha)
            ax.plot(r, distribution_stable.pdf_radius(r), linewidth=2, label=label, color=color, linestyle=linestyle)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2e0)
    ax.set_xlabel("$r$")
    ax.set_ylabel("$f_{R}(r)$")
    ax.legend(title="$\gamma={{{:0.02f}}}$".format(scale))

    plt.savefig("results/distributions_bivariate_stable_radial.pdf")


def bivariate_stable_distance():

    fig = plt.figure(figsize=(8, 4))
    ax = fig.gca()

    cmap = plt.cm.brg

    scale = 1.3 + numpy.random.random() * 0.5
    # scale = 1.0

    steps = 10
    r1 = numpy.linspace(0.0, 0.5, 2 * steps)
    r2 = numpy.linspace(0.5, 1.0, steps)
    r3 = numpy.linspace(1.0, 2.0, steps)
    r = numpy.concatenate([r1, r2[1:], r3[1:]])

    # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.9, 2.0]
    alphas = [0.5, 1.0, 1.5, 2.0]

    for alpha in alphas:

        if alpha == 1.0:
            distribution_cauchy = distributions.Cauchy2D(scale=scale)
            print(distribution_cauchy)
            ax.plot(r, distribution_cauchy.pdf_distance(r), linewidth=2,
                    label=str(distribution_cauchy), color=cmap(0.5 * 1.0))

        elif alpha == 2.0:
            distribution_normal = distributions.Normal2D(scale=numpy.sqrt(2) * scale)
            print(distribution_normal)
            ax.plot(r, distribution_normal.pdf_distance(r), linewidth=2,
                    label=str(distribution_normal), color=cmap(0.5 * 2.0))

        else:
            distribution_stable = distributions.Stable2D(scale=scale, alpha=alpha)
            print(distribution_stable)
            ax.plot(r, distribution_stable.pdf_distance(r), linewidth=2,
                    label=str(distribution_stable), color=cmap(0.5 * alpha))

    ax.set_ylim(0.0, 100.0)
    ax.set_yscale("symlog", linthreshy=0.1, linscaley=3.0)
    ax.set_xlabel("$|x|$")
    ax.set_ylabel("$f(x)=f(|x|)$")
    ax.legend(loc=1, ncol=2)

    plt.savefig("results/distributions_bivariate_stable_distance.pdf")


def _bivariate_stable_powerlaw_thread(steps, alpha, scale):

    from transforms import hankel_functional, inverse_hankel_functional
    import scipy.special

    if alpha == 1.0:
        dist = distributions.Cauchy2D(scale=scale)
    elif alpha == 2.0:
        dist = distributions.Normal2D(scale=scale * numpy.sqrt(2.0))
    else:
        dist = distributions.Stable2D(alpha=alpha, scale=scale)

    x = numpy.logspace(2, 3, steps)
    y = numpy.zeros(x.shape)
    d = dist.pdf(x, y)
    c, e = distributions.fit_powerlaw(x, d, method="opti4")
    d2 = dist.pdf_distance_asymptotic(x)
    d3 = numpy.array([inverse_hankel_functional(dist.cf_distance, _x) for _x in x]) / 2.0 / numpy.pi
    d4 = numpy.array([inverse_hankel_functional(dist.cf_distance_asymptotic, _x) for _x in x]) / 2.0 / numpy.pi

    k = numpy.logspace(-5, -3, steps)
    char = dist.cf(k, numpy.zeros(k.shape))
    char2 = 1.0 - 1.0 * (scale * k) ** alpha

    X_alpha = - numpy.pi * scipy.special.gamma(- 0.5 * alpha) / scipy.special.gamma(1.0 + 0.5 * alpha) * (2.0 * numpy.pi)**2
    # char3 = [X_alpha * (1.0 - hankel_functional(dist.pdf_distance_asymptotic, _k) * 2.0 * numpy.pi) for _k in k]

    char3 = [(1.0 - hankel_functional(dist.pdf_distance_asymptotic, _k) * 2.0 * numpy.pi / X_alpha) / X_alpha for _k in k]

    return (x, d, c, e, d2, d3, d4, str(dist), k, char, char2, char3)


def bivariate_stable_powerlaw():

    from matplotlib import pyplot as plt

    scales = [0.6, 1.2, 5.0]
    alphas = [1.8, 1.0, 0.6]
    stepss = [10] * 3

    fig = plt.figure(figsize=(10, 7))
    axes = fig.subplots(3, 2)

    results = []
    for steps, alpha, scale in zip(stepss, alphas, scales):
        results.append(_bivariate_stable_powerlaw_thread(steps, alpha, scale))

    for (ax1, ax2), steps, alpha, scale, result in zip(axes, stepss, alphas, scales, results):

        x, d, c, e, d2, d3, d4, name, k, char, char2, char3 = result

        ax1.plot(x, d, "r-", label=name + " [pdf]")
        ax1.plot(x, c * x ** e, "r--", label="${:0.03f} \cdot |\mathbf{{x}}|^{{{:0.03f}}}$ [fitted pdf]".format(c, e))
        ax1.plot(x, d2, "rx:", label="$-\gamma^{{\\alpha}} c(\\alpha)^{-1}  |\mathbf{{x}}|^{{-(2+\\alpha)}}$ [asymptotic pdf]")
        ax1.plot(x, d3, "b--", label="inverse hankel of cf")
        ax1.plot(x, d4, "b:", label="inverse hankel of asymptotic cf")

        ax1.set_xscale("log")
        ax1.set_yscale("log")

        ax1.legend()

        ax2.plot(k, char, "r-", label="$\exp(- \gamma^{{\\alpha}} |\\mathbf{k}|^{{\\alpha}})$ [cf]")
        ax2.plot(k, char2, "r:", label="$1-\gamma^{{\\alpha}} |\\mathbf{{k}}|^{{\\alpha}}$ [asymptotic cf]")
        ax2.plot(k, char3, "b--", label="hankel of asymptotic pdf")

        ax2.set_xscale("log")
        ax2.set_yscale("log")

        ax2.legend()

    plt.savefig("results/distributions_bivariate_stable_asymptotic_powerlaw.pdf")


def bivariate_stable_density_implementations_performance():

    import timeit

    r = numpy.linspace(0.0, 100.0, 107)

    dist = distributions.Stable2D(alpha=1.423, scale=0.6)

    def test_nolan1():
        numpy.vectorize(dist._pdf_radius_nolan1)(r)

    def test_nolan2():
        numpy.vectorize(dist._pdf_radius_nolan2)(r)

    def test_zolotarev():
        numpy.vectorize(dist._pdf_radius_zolotarev)(r)

    print("test_nolan1 ...")
    tt = timeit.timeit("test_nolan1()", globals=locals(), number=1)
    print("test_nolan1", tt)

    print("test_nolan2 ...")
    tt = timeit.timeit("test_nolan2()", globals=locals(), number=1)
    print("test_nolan2", tt)

    print("test_zolotarev ...")
    tt = timeit.timeit("test_zolotarev()", globals=locals(), number=1)
    print("test_zolotarev", tt)


def bivariate_stable_density_implementations():

    dist = distributions.Stable2D(alpha=1.0, scale=5.6)

    from matplotlib import pyplot as plt
    plt.figure()

    r = numpy.linspace(0.0, 30.0, 20)

    p1 = numpy.vectorize(dist._pdf_radius_nolan1)(r)
    p2 = numpy.vectorize(dist._pdf_radius_nolan2)(r)
    p3 = numpy.vectorize(dist._pdf_radius_zolotarev)(r)

    plt.plot(r, p1, label="nolan1")
    plt.plot(r, p2, label="nolan2")
    plt.plot(r, p3, label="zolotarev")

    distributions.Stable_scipy(
        alpha=dist.univariate_alpha,
        beta=1.0,
        scale=dist.univariate_scale,
        loc=0.0)

    # plt.plot(r, ...)

    # plt.yscale("log")
    plt.legend()

    plt.savefig("results/distributions_bivariate_stable_implementations.pdf")


def bivariate_stable_fourier_transform_density():

    import transforms

    from matplotlib import pyplot as plt
    from matplotlib import colors as cl
    import matplotlib

    fig = plt.figure(figsize=(10, 11))
    oleft = matplotlib.gridspec.GridSpec(3, 1, figure=fig, left=0.01, bottom=0.1, right=0.1, top=0.95, wspace=0.05, hspace=0.0)
    left = matplotlib.gridspec.GridSpec(3, 2, figure=fig, left=0.14, bottom=0.1, right=0.7, top=0.95, wspace=0.05, hspace=0.0)
    right = matplotlib.gridspec.GridSpec(3, 1, figure=fig, left=0.75, bottom=0.1, right=0.97, top=0.95, wspace=0.05, hspace=0.0)
    bleft = matplotlib.gridspec.GridSpec(1, 2, figure=fig, left=0.15, bottom=0.04, right=0.7, top=0.06, wspace=0.05, hspace=0.0)

    ax10 = plt.subplot(oleft[0])
    ax20 = plt.subplot(oleft[1])
    ax30 = plt.subplot(oleft[2])

    ax11 = plt.subplot(left[0, 0])
    ax12 = plt.subplot(left[0, 1], sharex=ax11, sharey=ax11)
    ax21 = plt.subplot(left[1, 0], sharex=ax11, sharey=ax11)
    ax22 = plt.subplot(left[1, 1], sharex=ax11, sharey=ax11)
    ax31 = plt.subplot(left[2, 0], sharex=ax11, sharey=ax11)
    ax32 = plt.subplot(left[2, 1], sharex=ax11, sharey=ax11)

    ax13 = plt.subplot(right[0])
    ax23 = plt.subplot(right[1], sharex=ax13, sharey=ax13)
    ax33 = plt.subplot(right[2], sharex=ax13, sharey=ax13)

    ax41 = plt.subplot(bleft[0])
    ax42 = plt.subplot(bleft[1])

    ax10.set_axis_off()
    ax20.set_axis_off()
    ax30.set_axis_off()

    plt.setp(ax11.get_xticklabels(), visible=False)
    plt.setp(ax12.get_xticklabels(), visible=False)
    plt.setp(ax13.get_xticklabels(), visible=False)
    plt.setp(ax21.get_xticklabels(), visible=False)
    plt.setp(ax22.get_xticklabels(), visible=False)
    plt.setp(ax23.get_xticklabels(), visible=False)

    plt.setp(ax12.get_yticklabels(), visible=False)
    plt.setp(ax22.get_yticklabels(), visible=False)
    plt.setp(ax32.get_yticklabels(), visible=False)

    # ax22.yaxis.get_offset_text().set_visible(False)

    vmin = 1e-4
    norm = cl.LogNorm(vmin=vmin, vmax=1.0)
    cmap0 = plt.cm.GnBu
    cmap1 = plt.cm.OrRd
    cmap0 = plt.cm.gist_yarg
    cmap1 = plt.cm.gist_yarg

    def plot(ax0, ax1, ax2, ax3, dist):

        sample = dist.sample(100000)

        dx = 0.1
        edges = numpy.arange(-10.0, 10.0 + dx, dx)
        h, xe, ye = numpy.histogram2d(*sample.T, bins=[edges, edges], normed=True, range=[[-10.0, 10.0], [-10.0, 10.0]])

        x = (xe[:-1] + xe[1:]) * 0.5
        xy = numpy.meshgrid(x, x)

        uv, F = transforms.fourier_2d(*xy, h)

        F = F.real
        F[F < vmin] = vmin

        Fa = dist.cf(*uv)

        ax1.imshow(Fa, extent=(uv[0][0, 0], uv[0][0, -1], uv[0][0, 0], uv[0][0, -1]), norm=norm, cmap=cmap1)
        ax2.imshow(F, extent=(uv[0][0, 0], uv[0][0, -1], uv[0][0, 0], uv[0][0, -1]), norm=norm, cmap=cmap0)
        ax3.plot(uv[0][0, :], Fa[Fa.shape[0] // 2, :], color=cmap1(1.0), linewidth=2, linestyle="--",
                 label="$\\varphi(\mathbf{{k}})|_{{k_{{2}}=0}}$")
        ax3.plot(uv[0][0, :], F[F.shape[0] // 2, :], color=cmap0(0.5), linewidth=2,
                 label="$\mathcal{{F}}_{{\mathbf{{x}}}}\{p(\mathbf{{x}})\}(\mathbf{{k}})|_{{k_{{2}}=0}}$")
        ax3.set_xlim(0.0, uv[0][0, -1])
        ax3.set_yscale("symlog", linthreshy=0.01, linscaley=3.0)

        name = str(dist)
        error = numpy.absolute(F - Fa).max()

        ax0.text(0.5, 0.25, "{}\nabserr={:0.05f}".format(name, error), rotation=90, rotation_mode="anchor",
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

        pos_good = ax2.get_position(False).get_points()
        box = ax3.get_position(False)
        pos_bad = box.get_points()
        pos_bad[0][1] = pos_good[0][1]
        pos_bad[1][1] = pos_good[1][1]
        box.set_points(pos_bad)
        ax3.set_position(box)

    plot(ax10, ax11, ax12, ax13, distributions.Normal2D(scale=0.34847))
    plot(ax20, ax21, ax22, ax23, distributions.Stable2D(alpha=1.34, scale=0.4283))
    plot(ax30, ax31, ax32, ax33, distributions.Cauchy2D(scale=0.47847))

    ax12.set_title("FFT of sample histogram\n$\mathcal{{F}}_{{\mathbf{{x}}}}\{p(\mathbf{{x}})\}(\mathbf{{k}})$\n")
    ax11.set_title("characteristic function\n$\\varphi(\mathbf{{k}})= \exp(-\gamma^{{\\alpha}} |\mathbf{{k}}|^{{\\alpha}})$\n")
    ax13.set_title("radial section\n$(k_{{2}}=0)$\n")

    ax11.set_ylabel("$k_{{2}}$")
    ax21.set_ylabel("$k_{{2}}$")
    ax31.set_ylabel("$k_{{2}}$")
    ax31.set_xlabel("$k_{{1}}$")
    ax32.set_xlabel("$k_{{1}}$")
    ax33.set_xlabel("$k_{{1}}$")

    #ax13.set_position(pos_bad)

    matplotlib.colorbar.ColorbarBase(ax41, cmap=cmap1, norm=norm, orientation="horizontal")
    matplotlib.colorbar.ColorbarBase(ax42, cmap=cmap0, norm=norm, orientation="horizontal")

    for ax in [ax41, ax42]:
        box = ax.get_position(False)
        pos = box.get_points()
        pos[0][0] += 0.05
        pos[1][0] -= 0.05
        box.set_points(pos)
        ax.set_position(box)

    ax33.legend(bbox_to_anchor=(0.5, -0.4), loc=8, borderaxespad=0.0, ncol=1)

    plt.savefig("results/distributions_bivariate_stable_fourier_transform_density.pdf")
