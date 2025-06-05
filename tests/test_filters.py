import pyest.filters as filters
import pyest.gm as pygm
import numpy as np
import numpy.testing as npt
from pyest.gm.defaults import default_gm
import pytest


def fail(f, error, *args, **kwargs):
    try:
        f(*args, **kwargs)
        raise AssertionError('should throw exception')
    except error:
        pass
    except Exception as e:
        raise AssertionError(
            'received {} instead of {} exception'.format(type(e), error))


def test_kalmandiscretepredict():
    """ test discrete Kalman filter prediction """
    def F(tkm1, tk): return np.array([[1, tk-tkm1], [0, 1]])
    Q = 1e-4*np.eye(2)
    kdp = filters.KfdPredict(F, Q=Q)

    mkm1 = np.array([1, 4])
    Pkm1 = np.eye(2)
    m_prior, P_prior = kdp.predict(tv=(0, 1), m_post=mkm1, P_post=Pkm1)

    npt.assert_almost_equal(m_prior, np.array([5, 4]), decimal=15)
    npt.assert_almost_equal(
        P_prior,
        np.array([
            [2.000100000000000,   1.000000000000000],
            [1.000000000000000, 1.000100000000000]
        ]),
        decimal=15
    )

    # test usage of process noise mapping matrix
    covariance_rotation = np.pi/8
    cos_a = np.cos(covariance_rotation)
    sin_a = np.sin(covariance_rotation)
    M = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a],
    ])

    m_prior, P_prior = filters.KfdPredict(F, Q=Q, M=M).predict(
        tv=(0, 1), m_post=mkm1, P_post=Pkm1)
    m_des, P_des = filters.KfdPredict(F, Q=M@Q@M.T).predict(
        tv=(0, 1), m_post=mkm1, P_post=Pkm1)
    npt.assert_array_equal(m_prior, m_des)
    npt.assert_array_equal(P_prior, P_des)


def test_extendedkalmandiscretepredict():
    """ test discrete time Extended Kalman filter prediction """
    def F(tkm1, tk, xkm1): return np.array([[0, np.cos(xkm1[1])], [1, 0]])
    def f(tkm1, tk, xkm1): return np.array([np.sin(xkm1[1]), xkm1[0]])
    Q = 1e-4*np.eye(2)
    kdp = filters.EkfdPredict(f, F, Q=Q)

    mkm1 = np.array([30., 0.])
    Pkm1 = np.array([[1., 0.5], [0.5, 3.4]])
    m_prior, P_prior = kdp.predict(tv=(0, 1), m_post=mkm1, P_post=Pkm1)
    G_exact = np.array([[0, 1],[1,0]])
    P_des = G_exact@Pkm1@G_exact.T + Q

    npt.assert_almost_equal(m_prior, np.array([0., 30.]), decimal=15)
    npt.assert_almost_equal(
        P_prior,
        P_des,
        decimal=15
    )

    # test usage of process noise mapping matrix
    covariance_rotation = np.pi/8
    cos_a = np.cos(covariance_rotation)
    sin_a = np.sin(covariance_rotation)
    M = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a],
    ])

    m_prior, P_prior = filters.EkfdPredict(f, F, Q=Q, M=M).predict(
        tv=(0, 1), m_post=mkm1, P_post=Pkm1)
    m_des, P_des = filters.EkfdPredict(f, F, Q=M@Q@M.T).predict(
        tv=(0, 1), m_post=mkm1, P_post=Pkm1)
    npt.assert_array_equal(m_prior, m_des)
    npt.assert_array_equal(P_prior, P_des)

def test_badunderweighting():
    R = np.eye(2)
    H = np.eye(2)
    fail(filters.KfdUpdate, ValueError, R, H, p=-1)


def test_ukfpredict():
    """ test unscented Kalman filter prediction """
    def F(tkm1, tk): return np.array([[1, tk-tkm1], [0, 1]])
    def f(x, tkm1, tk): return F(tkm1, tk)@x
    Q = 1e-4*np.eye(2)
    ukdp = filters.UkfPredict(f, Q=Q)

    mkm1 = np.array([1, 4])
    Pkm1 = np.eye(2)
    m_prior, P_prior = ukdp.predict(tv=(0, 1), m_post=mkm1, cov_post=Pkm1)

    npt.assert_allclose(m_prior, np.array([5, 4]), rtol=1e-10)

    npt.assert_allclose(
        P_prior,
        np.array([
            [2.000100000000000,   1.000000000000000],
            [1.000000000000000, 1.000100000000000]
        ]),
        rtol=1e-10
    )


def test_ukfupdate():
    # measurement function
    def h(m): return np.array([np.linalg.norm(m), np.arctan2(m[1], m[0])])
    # measurement covariance sqrt factor
    S = np.diag(np.array([1.2, 1e-3]))
    # measurement covariance
    R = S @ S.T
    # prior mean
    m = np.array([5, 4])
    # prior covariance
    P = np.diag([1.3, 1.2])**2
    # measurement
    z = np.array([
        7.048324804888169,
        0.676574827238148
    ])

    uup = filters.UkfUpdate(h, R)
    mkp, Pkp = uup.update(m, P, z)

    mkp_des = np.array([
        5.217969797170512,
        4.164008680583979
    ])

    Pkp_des = np.array([
        [0.462433790929766, 0.371681364603986],
        [0.371681364604055, 0.299977505340776]
    ])

    npt.assert_almost_equal(mkp, mkp_des, decimal=6)
    npt.assert_almost_equal(Pkp, Pkp_des, decimal=7)

    # test usage of process noise mapping matrix
    covariance_rotation = -np.pi/8
    cos_a = np.cos(covariance_rotation)
    sin_a = np.sin(covariance_rotation)
    L = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a],
    ])
    mkp, Pkp = filters.UkfUpdate(h, R, L=L).update(m, P, z)
    m_des, P_des = filters.UkfUpdate(h, L@R@L.T).update(m, P, z)
    npt.assert_array_equal(mkp, m_des)
    npt.assert_array_equal(Pkp, P_des)


def test_GmkfPredict():
    """ test discrete Gaussian Mixture filter prediction """
    def F(tkm1, tk): return np.array([[1, tk-tkm1], [0, 1]])
    Q = 1e-4*np.eye(2)
    gmp = filters.GmkfPredict(F, Q=Q)

    # form the posterior at k-1
    wkm1 = np.array([0.5, 0.5])
    mkm1 = np.array([[1, 4], [1.1, 3.9]])
    Pkm1 = np.tile(np.eye(2)[np.newaxis, :, :], (2, 1, 1))
    pkm1 = pygm.GaussianMixture(wkm1, mkm1, Pkm1)

    # propagate to k
    pkm = gmp.predict(tv=(0, 1), pkm1=pkm1)

    # check weights
    npt.assert_array_equal(pkm.w, wkm1)
    # check means
    npt.assert_almost_equal(pkm.m[0], np.array([5, 4]), decimal=15)
    npt.assert_almost_equal(pkm.m[1], np.array([5, 3.9]), decimal=15)
    # check covariances
    for Pkm in pkm.P:
        npt.assert_almost_equal(
            Pkm,
            np.array([
                [2.000100000000000,   1.000000000000000],
                [1.000000000000000, 1.000100000000000]
            ]),
            decimal=15
        )


def test_Gmekfpredict():
    """ test discrete time Extended Kalman filter prediction """
    def F(tkm1, tk, xkm1): return np.array([[0, np.cos(xkm1[1])], [1, 0]])
    def f(tkm1, tk, xkm1): return np.array([np.sin(xkm1[1]), xkm1[0]])
    Q = 1e-4*np.eye(2)
    gmekfpred = filters.GmekfPredict(f, F, Q=Q)

    wkm1 = np.array([0.4, 0.6])
    mkm1 = np.array([[30., 0.], [10., np.pi/2]])
    Pkm1 = np.array([[[1., 0.5], [0.5, 3.4]],
                     [[1., 0.2], [0.2, 1.5]]])
    pkm1 = pygm.GaussianMixture(wkm1, mkm1, Pkm1)
    p_prior = gmekfpred.predict(tv=(0, 1), pkm1=pkm1)
    G1_exact = np.array([[0, 1], [1, 0]])
    G2_exact = np.array([[0, 0], [1, 0]])
    w_des = wkm1.copy()
    m_des = np.array([[0., 30.], [1., 10.]])
    P_des = np.array([G@P@G.T + Q for P, G in zip(Pkm1, [G1_exact, G2_exact])])

    npt.assert_almost_equal(p_prior.w, w_des, decimal=15)
    npt.assert_almost_equal(p_prior.m, m_des, decimal=15)
    npt.assert_almost_equal(
        p_prior.P,
        P_des,
        decimal=15
    )


def test_GmukfPredict():
    """ test Gaussian mixture unscented Kalman filter prediction """
    def F(tkm1, tk): return np.array([[1, tk-tkm1], [0, 1]])
    def f(x, tkm1, tk): return F(tkm1, tk) @ x
    Q = 1e-4*np.eye(2)
    gmp = filters.GmukfPredict(f, Q=Q)

    # form the posterior at k-1
    wkm1 = np.array([0.5, 0.5])
    mkm1 = np.array([[1, 4], [1.1, 3.9]])
    Pkm1 = np.tile(np.eye(2)[np.newaxis, :, :], (2, 1, 1))
    pkm1 = pygm.GaussianMixture(wkm1, mkm1, Pkm1)

    # propagate to k
    pkm = gmp.predict(tv=(0, 1), pkm1=pkm1)

    # check weights
    npt.assert_array_equal(pkm.w, wkm1)
    # check means
    npt.assert_allclose(pkm.m[0], np.array([5, 4]), rtol=1e-10)
    npt.assert_allclose(pkm.m[1], np.array([5, 3.9]), rtol=1e-10)
    # check covariances
    for Pkm in pkm.P:
        npt.assert_allclose(
            Pkm,
            np.array([
                [2.000100000000000,   1.000000000000000],
                [1.000000000000000, 1.000100000000000]
            ]),
            rtol=1e-12
        )


def test_GmkfUpdate():
    """ test discrete Gaussian Mixture filter update """
    np.random.seed(46)
    pkm = default_gm()
    mshape = pkm.m[0].shape

    # generate a random truth using the first component of the GM
    Sk = np.linalg.cholesky(pkm.P[0])
    xtrue = pkm.m[0] + Sk@np.random.randn(*mshape)

    # define measurment matrix and covariance
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = 0.5*np.eye(2)
    # generate random measurement
    zk = H@xtrue + np.linalg.cholesky(R) @ np.random.randn(2)

    gmu = filters.GmkfUpdate(R, H)

    pkp = gmu.update(pkm, zk, unnormalized=False)
    npt.assert_array_almost_equal(
        pkp.w,
        np.array([0.9999957548898593, 4.24511014068597e-06]),
        decimal=15
    )
    npt.assert_allclose(
        pkp.P,
        np.array([[
            [0.33158338,  0.0357516, -0.00218292, -0.00189099],
            [0.0357516,  0.05381413,  0.02765637,  0.04426107],
            [-0.00218292,  0.02765637,  0.08838574,  0.00225619],
            [-0.00189099,  0.04426107,  0.00225619,  0.2455928]],
            [[0.16645219,  0.04357641, -0.00263503, -0.00102216],
             [0.04357641,  0.09108115,  0.02534425,  0.04045612],
             [-0.00263503,  0.02534425,  0.08852918,  0.00249192],
             [-0.00102216,  0.04045612,  0.00249192,  0.24596461]]]
        ),
        rtol=1e-5
    )
    npt.assert_allclose(
        pkp.m,
        np.array([
            [10.599916950203614,  5.088256210845965,
                0.109210904900068, 0.017558918150984],
            [13.718446710002267,  4.02956486549151,
                0.174226062360085, 0.192242755088443]
        ]),
        rtol=1e-8
    )


def test_GmukfUpdate():
    """ test Gaussian mixture unscented Kalman filter update """
    np.random.seed(46)
    pkm = default_gm()
    mshape = pkm.m[0].shape

    # generate a random truth using the first component of the GM
    Sk = np.linalg.cholesky(pkm.P[0])
    xtrue = pkm.m[0] + Sk@np.random.randn(*mshape)

    # define measurment matrix and covariance
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    def h(x): return H@x
    R = 0.5*np.eye(2)
    # generate random measurement
    zk = H@xtrue + np.linalg.cholesky(R) @ np.random.randn(2)

    gmu = filters.GmukfUpdate(h, R)

    pkp = gmu.update(pkm, zk, unnormalized=False)
    npt.assert_allclose(
        pkp.w,
        np.array([0.9999957548898593, 4.24511014068597e-06]),
        rtol=1e-7
    )
    npt.assert_allclose(
        pkp.P,
        np.array([[
            [0.33158338,  0.0357516, -0.00218292, -0.00189099],
            [0.0357516,  0.05381413,  0.02765637,  0.04426107],
            [-0.00218292,  0.02765637,  0.08838574,  0.00225619],
            [-0.00189099,  0.04426107,  0.00225619,  0.2455928]],
            [[0.16645219,  0.04357641, -0.00263503, -0.00102216],
             [0.04357641,  0.09108115,  0.02534425,  0.04045612],
             [-0.00263503,  0.02534425,  0.08852918,  0.00249192],
             [-0.00102216,  0.04045612,  0.00249192,  0.24596461]]]
        ),
        rtol=1e-5
    )
    npt.assert_allclose(
        pkp.m,
        np.array([
            [10.599916950203614,  5.088256210845965,
                0.109210904900068, 0.017558918150984],
            [13.718446710002267,  4.02956486549151,
                0.174226062360085, 0.192242755088443]
        ]),
        rtol=1e-8
    )


if __name__ == "__main__":
    pytest.main([__file__])
