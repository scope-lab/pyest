
import numpy.testing as npt
import numpy as np
import scipy.stats as ss
import pytest
import pyest.gm as gm
import pyest.metrics as metrics
from scipy.special import erfcx


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


def test_standardize_rv():
    P = np.array([[2, 0.5], [0.5, 1]])
    e = np.array([1, 2])
    e_std, corr, Dinv = metrics.standardize_rv(e, P)

    # test that correllation matrix has unit diagonal
    assert np.allclose(np.diag(corr), np.ones(corr.shape[0]))
    # test that inverse standardization givens original error vector
    e_reconstructed = np.linalg.solve(Dinv, e_std)
    assert np.allclose(e, e_reconstructed)


def test_weev1_standard_normal():
    """ test computation of WEEV-1 measure for standard normal samples """
    # generate samples from a standard normal distribution
    n_samples = 100
    n = 10
    samples = np.random.randn(n_samples, n)

    # compute the WEEV-1 of each error vector sample
    weev1s = metrics._weev1_standard_normal(samples)

    des_weev1s = np.zeros(n_samples)
    for i in range(n_samples):
        des_weev1s[i] = np.linalg.norm(samples[i], ord=1)

    npt.assert_allclose(weev1s, des_weev1s, rtol=1e-10)


def test_weev2_standard_normal():
    """ test computation of WEEV-2 measure for standard normal samples """

    # generate samples from a standard normal distribution
    n_samples = 100000
    n = 100
    # set random seed
    rng = np.random.default_rng(seed=0)
    samples = rng.normal(size=(n_samples, n))

    # compute the WEEV-2 of each error vector sample
    weev2s = metrics._weev2_standard_normal(samples)

    # the WEEV-2 should be chi-distributed with n degrees of freedom
    expected_mean = ss.chi.mean(df=n)
    expected_var = ss.chi.var(df=n)

    assert abs(np.mean(weev2s) - expected_mean) < 1e-2
    assert abs(np.var(weev2s) - expected_var) < 1e-2


def test_weev1_log_mgf():
    # test the log MGF using two properties of the MGF:
    n = 15  # state dimension
    # property 1: Jensen's inequality, M(t)>=exp(mu*t)
    n_samples = int(1e6)
    weev1_cdf, weev1_samps = metrics._weev1_ecdf(n, n_samples=n_samples)
    weev1_mean = np.mean(weev1_samps)
    mc_std_err = np.std(weev1_samps) / np.sqrt(n_samples)
    for t in np.logspace(-12, 2):
        # assert metrics.weev1_log_mgf(t, n) >= t * weev1_mean
        assert metrics.weev1_log_mgf(t, n) >= t * (weev1_mean - 3*mc_std_err)
        # assert metrics.weev1_log_mgf(-t, n) >= -t * weev1_mean
        assert metrics.weev1_log_mgf(-t, n) >= -t * (weev1_mean + 3*mc_std_err)

    # property 2: E[X] = M'(0)
    h = 1e-8
    mgf_based_mean = (metrics.weev1_log_mgf(h, n) - metrics.weev1_log_mgf(-h, n))/(2*h)
    npt.assert_allclose(weev1_mean, mgf_based_mean, rtol=1e-3)


def test_confidence_bounds_from_samples():
    # generate samples from a standard normal distribution
    n_samples = 10000
    n = 10
    rng = np.random.default_rng(seed=0)
    samples = rng.normal(size=(n_samples, n))

    # compute the WEEV-1 of each error vector sample
    weev1s = metrics._weev1_standard_normal(samples)

    # compute confidence bounds
    alpha = 0.05
    r1_weev1, r2_weev1 = metrics.confidence_bounds_from_samples(weev1s, alpha)

    # the confidence bounds should contain the true mean
    true_mean = np.mean(weev1s)
    assert r1_weev1 <= true_mean <= r2_weev1

    # compare to chernoff bounds
    r1_weev1_chernoff, r2_weev1_chernoff = metrics.weev1_chernoff_confidence_interval(n, alpha)
    assert r1_weev1_chernoff <= r1_weev1
    assert r2_weev1_chernoff >= r2_weev1


def test_weev1_chernoff_cdf_left_bound():
    r"""

    Notes
    -----
    One of the checks performed in this is related to the stationarity condition for the Chernoff bound,
    which states that the optimal value of "a" should equal the derivative of the cumulant generating
    function at the optimal "t" value. For WEEV-1, the derivative of the cumulant generating function is given by:
    .. math::
        \begin{align*}
        \Lambda'(t) &= n\left(t+\frac{\mathcal{N}(t)}{\Phi(t)}\right)\\
        \end{align*}

    The stable computation of the ratio of the normal PDF and CDF is given by the scaled complementary error function, erfcx, as follows:
    .. math::
        \begin{align*}
        \frac{\mathcal{N}(t)}{\Phi(t)} &= \frac{
        \frac{1}{\sqrt{2\pi}}e^{-t^2/2}}{\frac{1}{2}\text{erfc}\left(-\frac{t}{\sqrt{2}}\right)}\\
        &=
        \frac{\frac{2}{\sqrt{2\pi}}}{\text{erfcx}\left(-\frac{t}{\sqrt{2}}\right)}
        \end{align*}
    """
    for n in (1, 3, 5, 10, 15, 30):
        # compute samples for testing chernoff against
        ecdf, samps = metrics._weev1_ecdf(n, n_samples=int(1e7))
        for a in (1e-3, 1e-2, 0.05, 0.4):
            r1_chernoff_cdf, t_chernoff = metrics.weev1_chernoff_cdf_lower_bound(a, n)
            # check stationarity - the bound "a" should equal the derivative of the cumulant generating function at tstar
            a_test = (t_chernoff + np.sqrt(2/np.pi)/erfcx(-t_chernoff/np.sqrt(2)))*n
            npt.assert_allclose(a, a_test, rtol=1e-4)
            # Pr(X<=a) <= inf_{t<0}(M(t)*e^{-ta}) = r1_chernoff_cdf; (left tail)
            assert ecdf.evaluate(a) <= r1_chernoff_cdf


def test_weev1_chernoff_cdf_right_bound():
    r"""

    Notes
    -----
    One of the checks performed in this is related to the stationarity condition for the Chernoff bound,
    which states that the optimal value of "a" should equal the derivative of the cumulant generating
    function at the optimal "t" value. For WEEV-1, the derivative of the cumulant generating function is given by:
    .. math::
        \begin{align*}
        \Lambda'(t) &= n\left(t+\frac{\mathcal{N}(t)}{\Phi(t)}\right)\\
        \end{align*}

    """
    for n in (1, 3, 5, 10, 15, 30):
        # compute samples for testing chernoff against
        ecdf, samps = metrics._weev1_ecdf(n, n_samples=int(1e7))
        for a in n*np.array([1e-3, 1e-2, 0.05, 0.1, 1., 2., 3.]):
            C_ub, t_chernoff = metrics.weev1_chernoff_ccdf_upper_bound(a, n)
            if t_chernoff == 0:
                # check that the bound is trivial because the optimal t value is 0,
                # which means that the Chernoff bound is not providing any improvement over Markov's inequality
                assert metrics.weev1_dlog_mgf(0, n) > a
                continue  # skip the rest of this case, since the bound is trivial and the stationarity condition is not well-defined
            # check stationarity - the bound "a" should equal the derivative of the cumulant generating function at tstar
            a_test = metrics.weev1_dlog_mgf(t_chernoff, n)
            npt.assert_allclose(a, a_test, rtol=1e-4)
            # Pr(X>=a) <= inf_{t<0}(M(t)*e^{-ta}) = r2_chernoff_cdf; (right tail)
            assert 1 - ecdf.evaluate(a) <= C_ub


def test_weev1_confidence_interval():
    n = 10
    alpha = 0.05
    r1, r2 = metrics.weev1_confidence_interval(n, alpha, n_samples=int(1e6), rng=np.random.default_rng(seed=0), clear_cache=True)
    # the confidence interval should contain the true mean
    weev1_cdf, weev1_samps = metrics._weev1_ecdf(n, n_samples=int(1e6), rng=np.random.default_rng(seed=0))
    true_mean = np.mean(weev1_samps)
    assert r1 <= true_mean <= r2

    # test that cache works properly
    r1_cached, r2_cached = metrics.weev1_confidence_interval(n, alpha, n_samples=int(1e6), rng=np.random.default_rng(seed=0))
    assert r1 == r1_cached
    assert r2 == r2_cached

    # test that specifying a different number of samples or random seed results in different bounds
    r1_diff_samples, r2_diff_samples = metrics.weev1_confidence_interval(n, alpha, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1_diff_samples != r1
    assert r2_diff_samples != r2
    r1_diff_rng, r2_diff_rng = metrics.weev1_confidence_interval(n, alpha, n_samples=int(1e6), rng=np.random.default_rng(seed=1))
    assert r1_diff_rng != r1
    assert r2_diff_rng != r2

    # test that default options give expected results
    r1_default, r2_default = metrics.weev1_confidence_interval(n, alpha, clear_cache=True)
    r1_explicit, r2_explicit = metrics.weev1_confidence_interval(n, alpha, n_samples=int(1e7), rng=None, clear_cache=True)
    assert r1_default == r1_explicit
    assert r2_default == r2_explicit

    # test that higher false alarm rates result in narrower confidence intervals
    r1_high_alpha, r2_high_alpha = metrics.weev1_confidence_interval(n, alpha=0.1, n_samples=int(1e6))
    assert r1_high_alpha > r1
    assert r2_high_alpha < r2

    # test that chernoff bounds are looser than empirical bounds
    r1_chernoff, r2_chernoff = metrics.weev1_chernoff_confidence_interval(n, alpha)
    assert r1_chernoff <= r1
    assert r2_chernoff >= r2



def test_zca_whiten_standardized_rv():
    P = np.array([[2, 0.5], [0.5, 1]])
    e = np.array([1, 2])
    e_std, corr, Dinv = metrics.standardize_rv(e, P)
    e_zca, W = metrics.zca_whiten_standardized_rv(e_std, corr, Dinv)

    # test that W @ P @ W.T is the identity matrix
    P_zca = W @ P @ W.T
    assert np.allclose(P_zca, np.eye(P_zca.shape[0]), atol=1e-10)

    # test that inverse whitening transform gives back original vector
    e_reconstructed = np.linalg.solve(W, e_zca)
    assert np.allclose(e, e_reconstructed, atol=1e-10)


def test_zca_cor_whiten_rv():
    P = np.array([[2, 0.5], [0.5, 1]])
    e = np.array([1, 2])
    e_zca_cor, W = metrics.zca_cor_whiten_rv(e, P)

    # test that W @ P @ W.T is the identity matrix
    P_zca_cor = W @ P @ W.T
    assert np.allclose(P_zca_cor, np.eye(P_zca_cor.shape[0]), atol=1e-10)

    # test that inverse whitening transform gives back original vector
    e_reconstructed = np.linalg.solve(W, e_zca_cor)
    assert np.allclose(e, e_reconstructed, atol=1e-10)


def test_weev1():
    # create an error vector with known WEEV-1 value
    e_whitened = np.array([1, 2, 3])
    P_whitened = np.eye(3)
    weev1_des = np.linalg.norm(e_whitened, ord=1)
    weev1_computed = metrics.weev1(e_whitened, P_whitened)
    assert np.allclose(weev1_des, weev1_computed, rtol=1e-10)

    # apply a random linear transformation to the error vector
    rng = np.random.default_rng(seed=0)
    A = rng.normal(size=(3, 3))
    e_new = A @ e_whitened
    P_new = A @ P_whitened @ A.T
    e_whitened_new, W = metrics.zca_cor_whiten_rv(e_new, P_new)
    weev1_des_new = np.linalg.norm(e_whitened_new, ord=1)
    weev1_computed_new = metrics.weev1(e_new, P_new)
    assert np.allclose(weev1_des_new, weev1_computed_new, rtol=1e-10)


def test_weev2_confidence_interval():
    n = 30
    alpha = 0.05
    r1, r2 = metrics.weev2_confidence_interval(n, alpha)
    # confirm matches sample-based confidence interval
    n_samples = int(1e6)
    weev2s = metrics._weev2_standard_normal(np.random.randn(n_samples, n))
    r1_samp = np.quantile(weev2s, alpha/2)
    r2_samp = np.quantile(weev2s, 1 - alpha/2)
    assert abs(r1 - r1_samp) < 1e-2
    assert abs(r2 - r2_samp) < 1e-2


def test_weevinf_standard_normal():
    """ test computation of WEEV-∞ measure for standard normal samples """
    # generate samples from a standard normal distribution
    n_samples = 100
    n = 10
    samples = np.random.randn(n_samples, n)

    # compute the WEEV-∞ of each error vector sample
    weevinfs = metrics._weevinf_standard_normal(samples)

    des_weevinfs = np.zeros(n_samples)
    for i in range(n_samples):
        des_weevinfs[i] = np.linalg.norm(samples[i], ord=np.inf)

    npt.assert_allclose(weevinfs, des_weevinfs, rtol=1e-10)


def test_weevinf_confidence_interval():
    n = 10
    alpha = 0.05
    r1, r2 = metrics._weevinf_confidence_interval_empirical(n, alpha, n_samples=int(1e6), rng=np.random.default_rng(seed=0), clear_cache=True)
    # the confidence interval should contain the true mean
    weevinf_cdf, weevinf_samps = metrics._weevinf_ecdf(n, n_samples=int(1e6), rng=np.random.default_rng(seed=0))
    true_mean = np.mean(weevinf_samps)
    assert r1 <= true_mean <= r2

    # test that cache works properly
    r1_cached, r2_cached = metrics._weevinf_confidence_interval_empirical(n, alpha, n_samples=int(1e6), rng=np.random.default_rng(seed=0))
    assert r1 == r1_cached
    assert r2 == r2_cached

    # test that specifying a different number of samples or random seed results in different bounds
    r1_diff_samples, r2_diff_samples = metrics._weevinf_confidence_interval_empirical(n, alpha, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1_diff_samples != r1
    assert r2_diff_samples != r2
    r1_diff_rng, r2_diff_rng = metrics._weevinf_confidence_interval_empirical(n, alpha, n_samples=int(1e6), rng=np.random.default_rng(seed=1))
    assert r1_diff_rng != r1
    assert r2_diff_rng != r2

    # test that default options give expected results
    r1_default, r2_default = metrics._weevinf_confidence_interval_empirical(n, alpha, clear_cache=True)
    r1_explicit, r2_explicit = metrics._weevinf_confidence_interval_empirical(n, alpha, n_samples=int(1e7), rng=None, clear_cache=True)
    assert r1_default == r1_explicit
    assert r2_default == r2_explicit

    # test that higher false alarm rates result in narrower confidence intervals
    r1_high_alpha, r2_high_alpha = metrics._weevinf_confidence_interval_empirical(n, alpha=0.1, n_samples=int(1e6))
    assert r1_high_alpha > r1
    assert r2_high_alpha < r2

    # test agreement between the semi-analytical and empirical methods
    r1_semi, r2_semi = metrics.weevinf_confidence_interval(n, alpha)
    r1_emp, r2_emp = metrics._weevinf_confidence_interval_empirical(n, alpha, n_samples=int(1e6), rng=np.random.default_rng(seed=0))
    assert np.allclose(r1_semi, r1_emp, rtol=1e-2)
    assert np.allclose(r2_semi, r2_emp, rtol=1e-2)


def test_snees():
    n = 10
    # create an error vector with known SNEES value
    e = np.ones(n)
    P = np.eye(n)

    snees_des = 1
    snees_computed = metrics.snees(e, P)
    assert np.allclose(snees_des, snees_computed, rtol=1e-10)

    # apply a random linear transformation to the error vector
    rng = np.random.default_rng(seed=0)
    A = rng.normal(size=(n, n))
    e_new = A @ e
    P_new = A @ P @ A.T
    snees_des_new = 1
    snees_computed_new = metrics.snees(e_new, P_new)
    assert np.allclose(snees_des_new, snees_computed_new, rtol=1e-10)


def test_asnees():
    n = 10
    num_trials = 100
    num_time_steps = 5

    # create random error vectors and covariance matrices
    rng = np.random.default_rng(seed=0)
    e = rng.normal(size=(num_trials, num_time_steps, n))
    S = np.random.normal(size=(num_trials, num_time_steps, n, n))
    P = np.zeros_like(S)
    for i in range(num_trials):
        for j in range(num_time_steps):
            P[i, j] = S[i, j] @ S[i, j].T  # make sure covariance matrices are positive definite

    asnees_computed = metrics.asnees(e, P)
    # test that the output has the correct shape
    assert asnees_computed.shape == (num_time_steps,)

    # compute the ASNEES values manually for an arbitrary time step
    k = 3
    snees_vals = np.zeros(num_trials)
    for i in range(num_trials):
        snees_vals[i] = metrics.snees(e[i, k], P[i, k])
    asnees_des = np.mean(snees_vals)
    assert np.allclose(asnees_des, asnees_computed[k], rtol=1e-10)

    # also confirm that all ASNEES values are positive
    assert np.all(asnees_computed > 0)


def test_asnees_confidence_interval():
    n = 2
    num_trials = 50
    alpha = 0.05

    # compare to published textbook ANEES value (r1=1.5, r2=2.6 for n=2, num_trials=50, alpha=0.05)
    r1, r2 = metrics.asnees_confidence_interval(n, alpha, num_trials)
    assert abs(n*r1 - 1.5) < 0.05
    assert abs(n*r2 - 2.6) < 0.05

    # test that region narrows when number of trials increases
    r1_more_trials, r2_more_trials = metrics.asnees_confidence_interval(n, alpha, num_trials*2)
    assert r1_more_trials > r1
    assert r2_more_trials < r2


def test_mnees_confidence_interval():
    n = 2
    num_trials = 50
    alpha = 0.05

    r1, r2 = metrics.mnees_confidence_interval(n, alpha, num_trials)
    assert r2 > r1
    # test that region bounds monotonically increase with number of trials
    r1_last = 0
    r2_last = 0
    for num_trials in range(1, 100):
        r1, r2 = metrics.mnees_confidence_interval(n, alpha, num_trials)
        assert r1 >= r1_last
        assert r2 >= r2_last
        r1_last = r1
        r2_last = r2

    # the MNEES bounds should be greater than the ANEES bounds for num_trials > 1
    r1, r2 = metrics.mnees_confidence_interval(n, alpha, num_trials)
    r1_anees, r2_anees = metrics.anees_confidence_interval(n, alpha, num_trials)
    assert r1_anees < r1
    assert r2_anees < r2

    # ... and they should match for num_trials = 1
    r1_anees_1trial, r2_anees_1trial = metrics.anees_confidence_interval(n, alpha, 1)
    r1_mnees_1trial, r2_mnees_1trial = metrics.mnees_confidence_interval(n, alpha, 1)
    assert r1_anees_1trial == r1_mnees_1trial
    assert r2_anees_1trial == r2_mnees_1trial

def test_aweev1_confidence_interval():
    n = 13
    num_trials = 100
    alpha = 0.05

    r1, r2 = metrics.aweev1_confidence_interval(n, alpha, num_trials, n_samples=int(1e5), rng=np.random.default_rng(seed=0), clear_cache=True)
    # the confidence interval should contain the true mean
    weev1_samps = metrics._weev1_ecdf(n, n_samples=int(1e6), rng=np.random.default_rng(seed=0))[1]
    aweev1_samps = metrics._aweev1_ecdf(num_trials, weev1_samps, n_samples=int(1e5), rng=np.random.default_rng(seed=0))[1]
    true_mean = np.mean(aweev1_samps)
    assert r1 <= true_mean <= r2

    # test that cache works properly
    r1_cached, r2_cached = metrics.aweev1_confidence_interval(n, alpha, num_trials, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1 == r1_cached
    assert r2 == r2_cached

     # test that specifying a larger number of trials results in a narrower confidence interval
    r1_more_trials, r2_more_trials = metrics.aweev1_confidence_interval(n, alpha, num_trials*2, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1_more_trials > r1
    assert r2_more_trials < r2


def test_mweev1_confidence_interval():
    n = 13
    num_trials = 100
    alpha = 0.05

    r1, r2 = metrics.mweev1_confidence_interval(n, alpha, num_trials, n_samples=int(1e5), rng=np.random.default_rng(seed=0), clear_cache=True)
    # the confidence interval should contain the true mean
    weev1_samps = metrics._weev1_ecdf(n, n_samples=int(1e6), rng=np.random.default_rng(seed=0))[1]
    mweev1_samps = metrics._mweev1_ecdf(num_trials, weev1_samps, n_samples=int(1e5), rng=np.random.default_rng(seed=0))[1]
    true_mean = np.mean(mweev1_samps)
    assert r1 <= true_mean <= r2

    # test that cache works properly
    r1_cached, r2_cached = metrics.mweev1_confidence_interval(n, alpha, num_trials, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1 == r1_cached
    assert r2 == r2_cached

     # test that specifying a larger number of trials results in a narrower and shifted confidence interval
    r1_more_trials, r2_more_trials = metrics.mweev1_confidence_interval(n, alpha, num_trials*2, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1_more_trials > r1
    assert r2_more_trials > r2
    assert (r2_more_trials - r1_more_trials) < (r2 - r1)


def test_aweevinf_confidence_interval():
    n = 13
    num_trials = 100
    alpha = 0.05

    r1, r2 = metrics.aweevinf_confidence_interval(n, alpha, num_trials, n_samples=int(1e5), rng=np.random.default_rng(seed=0), clear_cache=True)
    # the confidence interval should contain the true mean
    weevinf_samps = metrics._weevinf_ecdf(n, n_samples=int(1e6), rng=np.random.default_rng(seed=0))[1]
    aweevinf_samps = metrics._aweevinf_ecdf(num_trials, weevinf_samps, n_samples=int(1e5), rng=np.random.default_rng(seed=0))[1]
    true_mean = np.mean(aweevinf_samps)
    assert r1 <= true_mean <= r2

    # test that cache works properly
    r1_cached, r2_cached = metrics.aweevinf_confidence_interval(n, alpha, num_trials, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1 == r1_cached
    assert r2 == r2_cached

     # test that specifying a larger number of trials results in a narrower confidence interval
    r1_more_trials, r2_more_trials = metrics.aweevinf_confidence_interval(n, alpha, num_trials*2, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1_more_trials > r1
    assert r2_more_trials < r2


def test_mweevinf_confidence_interval():
    n = 13
    num_trials = 100
    alpha = 0.05

    r1, r2 = metrics._mweevinf_confidence_interval_empirical(n, alpha, num_trials, n_samples=int(1e5), rng=np.random.default_rng(seed=0), clear_cache=True)
    # the confidence interval should contain the true mean
    weevinf_samps = metrics._weevinf_ecdf(n, n_samples=int(1e6), rng=np.random.default_rng(seed=0))[1]
    mweevinf_samps = metrics._mweevinf_ecdf(num_trials, weevinf_samps, n_samples=int(1e5), rng=np.random.default_rng(seed=0))[1]
    true_mean = np.mean(mweevinf_samps)
    assert r1 <= true_mean <= r2

    # test that cache works properly
    r1_cached, r2_cached = metrics._mweevinf_confidence_interval_empirical(n, alpha, num_trials, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1 == r1_cached
    assert r2 == r2_cached

     # test that specifying a larger number of trials results in a narrower and shifted confidence interval
    r1_more_trials, r2_more_trials = metrics._mweevinf_confidence_interval_empirical(n, alpha, num_trials*2, n_samples=int(1e5), rng=np.random.default_rng(seed=0))
    assert r1_more_trials > r1
    assert r2_more_trials > r2
    assert (r2_more_trials - r1_more_trials) < (r2 - r1)

    # test that semi-analytical solution matches empirical one
    r1_semi, r2_semi = metrics.mweevinf_confidence_interval(n, alpha, num_trials)
    assert np.isclose(r1, r1_semi, rtol=0.1)
    assert np.isclose(r2, r2_semi, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])

