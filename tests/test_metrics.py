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



if __name__ == '__main__':
    pytest.main([__file__])
