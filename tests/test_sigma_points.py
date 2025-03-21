import numpy as np
import numpy.testing as npt
import pyest.filters.sigma_points as pysp
import pytest
from pyest.utils import has_unique_columns, fail, BadCholeskyFactor
from scipy.linalg import cholesky


def test_sigma_points():
    m = np.array([6, 7])
    n = m.shape[0]
    S = np.array([[0.6, 0.5], [0.1, 1.0]])

    alpha = 1e-3
    kappa = 0
    sig_opts = pysp.SigmaPointOptions(alpha=alpha, kappa=kappa)
    sigmas = pysp.SigmaPoints(m, S, sig_opts)

    # check that sigma points are unique
    assert(has_unique_columns(sigmas.X))
    # first sigma point should be the original mean
    npt.assert_array_equal(m,sigmas.X[:,0].flatten())
    # mean of sigma points should be equal to original mean
    sp_mean = np.sum(np.tile(sigmas.wm, (n,1))*sigmas.X, axis=1).flatten()
    npt.assert_almost_equal(m, sp_mean)
    # covariance of sigma points should be equal to original covariance
    sp_cov = np.zeros_like(S)
    for i in range(len(sigmas.wc)):
        sp_cov += sigmas.wc[i]*np.outer(sigmas.X[:,i] - m, (sigmas.X[:,i]-m).T)
    npt.assert_almost_equal(S@S.T, sp_cov)


def test_unscented_transform():
    m = np.array([6, 7])
    n = m.shape[0]
    Stemp = np.array([[0.6, 0.5], [0.1, 1.0]])
    P = Stemp @ Stemp.T
    Schol = cholesky(P,lower=True)

    alpha = 1e-3
    kappa = 0
    beta = 2
    opts = pysp.SigmaPointOptions(alpha,beta,kappa)
    f = np.sin

    mt, Pt, Dt, SP, y = pysp.unscented_transform(
        m, P, f, opts, cov_type='full')

    mt_des = np.array([-0.194193779898342, 0.325208394293441])  # from MATLAB
    Pt_des = np.array([[0.576900718520819, 0.348820174683678],
                       [0.348820174684809, 0.794205655799182]])
    Dt_des = np.array([[-0.085221718300584, -0.084161005327373, -0.085221718300584, -0.086282090386922, -0.085221718300584],
                       [0.331778204425349, 0.332542323692703, 0.332528684633987, 0.331013409646586, 0.331027072615301]])
    SP_des = np.array([[6.000000000000000, 6.001104536101734, 6.000000000000000, 5.998895463898266, 6.000000000000000],
                       [7.000000000000000, 7.001014000355691, 7.000995893206482, 6.998985999644309, 6.999004106793518]])
    wc_des = 1e5*np.array([-9.999959999722444, 2.499999999928111,
                           2.499999999928111, 2.499999999928111, 2.499999999928111])

    npt.assert_almost_equal(mt_des, mt, decimal=15)
    npt.assert_almost_equal(Pt_des, Pt, decimal=11)
    npt.assert_almost_equal(Dt_des, Dt, decimal=15)
    npt.assert_almost_equal(SP_des, SP.X, decimal=15)
    npt.assert_almost_equal(wc_des, SP.wc, decimal=15)

    # repeat using square root factor
    mt, St, Dt, SP, y = pysp.unscented_transform(
        m, Schol, f, opts, cov_type='cholesky')
    npt.assert_almost_equal(mt_des, mt, decimal=15)
    npt.assert_almost_equal(Pt_des, St@St.T, decimal=10)
    npt.assert_almost_equal(Dt_des, Dt, decimal=15)
    npt.assert_almost_equal(SP_des, SP.X, decimal=15)
    npt.assert_almost_equal(wc_des, SP.wc, decimal=15)

    # test checking of lower triangular cholesky factor
    fail(pysp.unscented_transform, BadCholeskyFactor, m, Schol.T, f, opts, cov_type='cholesky')

def test_angular_residuals():
    y = np.deg2rad([1, 179, 181, 359, -1, -179, -181, -359])
    m = np.pi
    des_res = np.deg2rad([-179.,   -1.,    1.,  179.,  179.,    1.,   -1., -179.])
    npt.assert_almost_equal(des_res, pysp.angular_residual(y, m), decimal=15)

if __name__ == "__main__":
    pytest.main([__file__])