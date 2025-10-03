import numpy.testing as npt
import numpy as np
import scipy.stats as ss
import pytest
import pyest.gm as gm
import pyest.metrics as metrics


def test_l2_dist():
    """ test computation of L2 distance between Gaussian mixtures """

    # test identical mixtures
    gm1 = gm.defaults.default_gm()
    gm2 = gm.defaults.default_gm()

    result = gm.l2_dist(gm1, gm2)
    assert abs(result) < 1e-15

    # test different mixtures
    gm3 = gm.defaults.default_gm(mean_shift=np.array([-1, 2, -3, 4]))
    des_l2 = 0.261873360193419
    npt.assert_approx_equal(gm.l2_dist(gm1, gm3), des_l2, significant=9)


def test_integral_squared_error_2d():
    """ test computation of integral squared error between 2D densities """

    # test identical mixtures
    p1 = gm.defaults.default_gm().marginal_2d([0, 1])
    p2 = gm.defaults.default_gm().marginal_2d([0, 1])
    lb, ub = gm.bounds(p1.m, p1.P, sigma_mult=3)

    ise, int_err = metrics.integral_squared_error_2d(p1, p2, lb[0], ub[0], lb[1], ub[1])
    assert abs(ise) < 1e-10

    # now, create two mixtures with disjoint supports. The ISE in this case
    # should trend toward the sum of the individual integrals of the squared densities.
    shift = ub - lb
    p2 = gm.defaults.default_gm(mean_shift=np.array([shift[0], shift[1], 0, 0])).marginal_2d([0, 1])
    ise, int_err = metrics.integral_squared_error_2d(p1, p2, lb[0], ub[0] + shift[0], lb[1], ub[1] + shift[1])
    int_p1_sq = gm.integral_squared_gm(p1)
    int_p2_sq = gm.integral_squared_gm(p2)
    des_ise = int_p1_sq + int_p2_sq
    assert abs(ise - des_ise) < 1e-5


def test_normalized_integral_squared_error_2d():
    """ test computation of NISE between 2D densities """

    # test identical mixtures
    p1 = gm.defaults.default_gm().marginal_2d([0, 1])
    p2 = gm.defaults.default_gm().marginal_2d([0, 1])
    lb, ub = gm.bounds(p1.m, p1.P, sigma_mult=3)

    nise, ise, int_err = metrics.normalized_integral_squared_error_2d(p1, p2, lb[0], ub[0], lb[1], ub[1])
    assert abs(nise) < 1e-10

    # now, create two mixtures with disjoint supports. The NISE in this case
    # should trend toward 1
    shift = ub - lb
    p2 = gm.defaults.default_gm(mean_shift=np.array([shift[0], shift[1], 0, 0])).marginal_2d([0, 1])
    nise, ise, int_err = metrics.normalized_integral_squared_error_2d(p1, p2, lb[0], ub[0] + shift[0], lb[1], ub[1] + shift[1])
    des_nise = 1
    assert abs(nise - des_nise) < 1e-4



if __name__ == '__main__':
    pytest.main([__file__])
