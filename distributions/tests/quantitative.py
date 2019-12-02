import numpy

import distributions


def bivariate_normal():

    scale = 0.1 + numpy.random.random() * 10.0

    N_rad = distributions.Normal2D_radial(scale)
    N_mul = distributions.Normal2D_mv(scale)

    print(repr(N_rad))
    print(repr(N_mul))

    R = 100.0
    N = 1000
    x = numpy.linspace(-R, R, N)
    dx = 2.0 * R / N
    xx, yy = numpy.meshgrid(x, x)

    density_rad = N_rad.pdf(xx, yy)
    density_mul = N_mul.pdf(xx, yy)

    total_density_rad = numpy.sum(density_rad) * dx * dx
    total_density_mul = numpy.sum(density_mul) * dx * dx

    assert numpy.isclose(total_density_rad, 1.0, 1e-2)
    assert numpy.isclose(total_density_mul, 1.0, 1e-2)

    density_diff = density_rad - density_mul

    assert numpy.allclose(density_diff, 0.0)

    K = 1000000
    samples_rad = N_rad.sample(K)
    samples_mul = N_mul.sample(K)

    radial_var_rad = numpy.linalg.norm(samples_rad, axis=1).var()
    radial_var_mul = numpy.linalg.norm(samples_mul, axis=1).var()

    assert numpy.isclose(radial_var_rad, radial_var_mul, scale * 0.1)

    # test zero eval

    peak_rad = N_rad.pdf(0.0, 0.0)
    peak_mul = N_mul.pdf(0.0, 0.0)

    print(peak_rad, peak_mul)


def bivariate_cauchy():

    scale = 0.8 + numpy.random.random() * 5.0

    cauchy_rad = distributions.Cauchy2D_radial(scale)
    cauchy_mul = distributions.Cauchy2D_mv(scale)

    print(repr(cauchy_rad))
    print(repr(cauchy_mul))

    N = 5000
    D = 300.0
    dx = 2.0 * D / N
    x = numpy.linspace(-D, D, N)
    xx, yy = numpy.meshgrid(x, x)

    density_rad = cauchy_rad.pdf(xx, yy)
    density_mul = cauchy_mul.pdf(xx, yy)

    total_density_rad = numpy.sum(density_rad) * dx * dx
    total_density_mul = numpy.sum(density_mul) * dx * dx

    assert numpy.isclose(total_density_rad, 1.0, 1e-1)
    assert numpy.isclose(total_density_mul, 1.0, 1e-1)
    assert numpy.allclose(density_rad, density_mul)

    K = 1000000
    samples_rad = cauchy_rad.sample(K)
    samples_mul = cauchy_mul.sample(K)

    # var test not applicable for infinite var distribution!
    # radial_var_rad = numpy.linalg.norm(samples_rad, axis=1).var()
    # radial_var_mul = numpy.linalg.norm(samples_mul, axis=1).var()
    # assert numpy.isclose(radial_var_rad, radial_var_mul, scale * 0.1)

    x_bins = numpy.linspace(- D - 0.5 * dx, D + 0.5 * dx, N + 1)
    hist_rad, _, _ = numpy.histogram2d(*samples_rad.T, [x_bins, x_bins], density=True)
    hist_mul, _, _ = numpy.histogram2d(*samples_mul.T, [x_bins, x_bins], density=True)

    # print(numpy.abs(hist - density).max())
    assert numpy.allclose(hist_rad, density_rad, atol=1e-2)
    assert numpy.allclose(hist_mul, density_mul, atol=1e-2)
    assert numpy.allclose(hist_mul, hist_rad, atol=1e-1)

    # test zero eval

    peak_rad = cauchy_rad.pdf(0.0, 0.0)
    peak_mul = cauchy_mul.pdf(0.0, 0.0)

    print(peak_rad, peak_mul)


def bivariate_scaling():

    scale = numpy.random.random() * 2.0
    alpha = numpy.random.random() * 1.5 + 0.5

    dist_stable = distributions.Stable2D(scale=scale, alpha=alpha)
    dist_normal = distributions.Normal2D(scale=scale * numpy.sqrt(2.0))
    dist_cauchy = distributions.Cauchy2D(scale=scale)

    x = numpy.random.randn(5) * 4.0
    y = numpy.random.randn(5) * 4.0

    scale2 = numpy.random.random() * 10.0

    dist_stable_rescaled = distributions.Stable2D(scale=scale * scale2, alpha=alpha)
    dist_normal_rescaled = distributions.Normal2D(scale=scale * scale2 * numpy.sqrt(2.0))
    dist_cauchy_rescaled = distributions.Cauchy2D(scale=scale * scale2)

    f = dist_cauchy.pdf(x / scale2, y / scale2) / scale2 / scale2
    g = dist_cauchy_rescaled.pdf(x, y)

    assert numpy.allclose(f, g)

    f = dist_normal.pdf(x / scale2, y / scale2) / scale2 / scale2
    g = dist_normal_rescaled.pdf(x, y)

    assert numpy.allclose(f, g)

    f = dist_stable.pdf(x / scale2, y / scale2) / scale2 / scale2
    g = dist_stable_rescaled.pdf(x, y)

    assert numpy.allclose(f, g)
