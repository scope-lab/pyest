import numpy as np
import numpy.testing as npt
import pyest.gm as gm
from pyest.particle import DiracMixture
import pytest


def test_dirac_mixture():
    n_samp = int(1e6)
    p = gm.defaults.default_gm()

    # create DiracMixture from random samples
    pdm = DiracMixture(np.full(n_samp, 1/n_samp), p.rvs(n_samp))
    mean = pdm.mean()
    cov = pdm.cov()

    npt.assert_allclose(mean, p.mean(), rtol=1e-1)
    npt.assert_array_almost_equal(cov, p.cov(), decimal=2)

#import pyest.distributions as pydist
#from pyest.gm.defaults import default_gm
#from pyest.utils import fail
#
#def test_gm_distr():
#    gmdist = pydist.defaults.default_gm_distr()
#    assert(gmdist.gm == default_gm())
#    assert(gmdist.state_dim == default_gm().msize)
#
#def test_MvUniform():
#    lb = [0, 50]
#    rb = [10, 60]
#
#    udist = pydist.MvUniform(lb, rb)
#
#    pdf_val = 1/np.prod(np.array(rb) - np.array(lb))
#
#    # check in-value
#    npt.assert_allclose(udist.pdf([5, 51]), pdf_val, rtol=1e-14)
#    # check out-value
#    assert(udist.pdf([5, 48.3]) == 0)
#
#    # test failure cases
#    lb_bad_size = [0, 50, 50]
#    fail(pydist.MvUniform, ValueError, lb_bad_size, rb)
#    lb_bad_val = [11, 50]
#    fail(pydist.MvUniform, ValueError, lb_bad_val, rb)
#
#


if __name__ == "__main__":
    pytest.main([__file__])