# CTRW

A proof-of-concept library for simulating continuous time random walks (CTRW) in one and two dimensions.
The library contains a set of probability distributions, Laplace and Fourier transform routines and a framework for simulating and analyzing CTRWs.

Ensembles of random walk trajectories can be compared with the fundamental solution of the fractional diffusion equation.
Trajectories can be characterized using implemented displacement measures, local time and velocity autocorrelation.
Application examples can be found in the testing routines.

The program code in this repository is made available under the [GPLv3 license](COPYING) (GPL-3.0-or-later).


## Usage

* After cloning this repository, run `git submodule update --init` to install external libraries.
* In `ext_mittagleffler/mittag_leffler.py` change the import line to `from .ml_internal import LTInversion`!
* The test scripts generate pdf figures in the `results` directory.


## Literature and references

See the `REFERENCES` files in the subdirectories:
* [distributions/REFERENCES](distributions/REFERENCES)
* [stochproc/REFERENCES](stochproc/REFERENCES)
* [transforms/REFERENCES](transforms/REFERENCES)

## Required libraries

 * numpy/scipy
 * [mpmath](http://mpmath.org) [T2]
 * [hankel](https://hankel.readthedocs.io/en/latest) [T1]
 * multiprocessing (partial)
 * matplotlib (for testing)
 * mittag-leffler:
 A Mittag-Leffler function implementation.
 The mittag-leffler source code is provided [here](https://github.com/khinsen/mittag-leffler).
 See also references [D2].
 The files must be copied to the `ext_mittagleffler` directory (see Usage).
 * PyLevy:
 Univariate Lévy-stable distributions. 
 The source code of PyLevy can be found [here](https://github.com/josemiotto/pylevy).
 See also references [D5].
 A copy must be placed in the `ext_levy` directory (see Usage).

## Bivariate distributions

For isotropic bivariate distributions, `pdf_radius` returns the density of the amplitude distribution (the distribution of |X|).
`pdf_distance` returns the density of bivariate distributions (i.e. of X), but takes the absolute value |x| as an argument.
This function is not a density on R but on R^2.


## CTRWs

Standard spatial increments are configured with a stability parameter `0 < alpha <= 2` and a scale `0 < DC`.
Standard time increments are configured with a memory parameter `0 <= beta <= 1` and a scale `0 < TC`.
By convention `beta == 0` results in a degenerate increment distribution and constant time steps.
The parameter `delta_t` scales both distributions simultaneously, retaining analytical properties.

Trajectory or particle objects have the data members `dx, dt, x, t`.
Trajectories always start at `t_{0} = 0, x_{0} = 0` and for `i > 0` the trajectory is defined by 
`t_{i} = dt_{0} + ... + dt_{i-1}`, `x_{i} = dx_{0} + ... + dx_{i-1}`.
