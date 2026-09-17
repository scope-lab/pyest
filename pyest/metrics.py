import numpy as np
from diskcache import Cache
from scipy.integrate import dblquad
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar, root_scalar
from scipy.special import erf, erfcx, erfinv, log_ndtr
from scipy.stats import chi, chi2, ecdf

import pyest.gm as pygm


def l2_dist(p1: pygm.GaussianMixture, p2: pygm.GaussianMixture) -> float:
    """Compute L2 distance between GMs p1 and p2.

    Parameters
    ----------
    p1 : GaussianMixture
        first Gaussian mixture
    p2 : GaussianMixture
        second Gaussian mixture

    Returns
    -------
    float
        L2 distance between the two input GMs

    """
    # first term is product of p1 and p1
    t1 = np.sum(
        [
            wi * wj * pygm.eval_mvnpdf(mi, mj, Pi + Pj)
            for (wi, mi, Pi) in p1
            for (wj, mj, Pj) in p1
        ]
    )
    t2 = np.sum(
        [
            wi * wj * pygm.eval_mvnpdf(mi, mj, Pi + Pj)
            for (wi, mi, Pi) in p1
            for (wj, mj, Pj) in p2
        ]
    )
    t3 = 0.0
    n = len(p2)
    for i in range(n):
        wi, mi, Pi = p2[i]
        # Diagonal term
        t3 += wi * wi * pygm.eval_mvnpdf(mi, mi, Pi + Pi)
        for j in range(i + 1, n):
            wj, mj, Pj = p2[j]
            val = wi * wj * pygm.eval_mvnpdf(mi, mj, Pi + Pj)
            t3 += 2 * val
    # t3 = np.sum([
    #     wi*wj*pygm.eval_mvnpdf(mi, mj, Pi + Pj) for (wi, mi, Pi) in p2 for (wj, mj, Pj) in p2
    # ])

    l2 = t1 - 2 * t2 + t3
    return l2


def max_covariance_ratio(S: np.ndarray, S_ref: np.ndarray) -> float:
    """Compute the maximum covariance ratio between two distributions.

    Required:
    ---------
    S : np.ndarray
        covariance matrix lower-triangular Cholesky square-root factor of
        the test distribution
    S_ref : np.ndarray
        covariance matrix lower-triangular Cholesky square-root factor of
        the reference distribution

    Returns
    -------
    float
        maximum covariance ratio

    """
    mat = solve_triangular(S, S_ref, lower=True)
    s_vals = np.linalg.svd(mat, compute_uv=False, hermitian=False)
    return max(s_vals[0], 1.0 / s_vals[-1])


def madem(m, S, m_ref):
    """Mahalanobis distance of the error of the mean.

    Required:
    ---------
    m : np.ndarray
        mean of the test distribution
    S : np.ndarray
        covariance matrix lower-triangular Cholesky square-root factor of
        the test distribution
    m_ref : np.ndarray
        mean of the reference distribution

    Returns
    -------
    float
        Mahalanobis distance of the error of the mean (MaDEM)

    """
    return np.linalg.norm(
        solve_triangular(S, m - m_ref, lower=True),
    )


def integral_squared_error_2d(p1, p2, a, b, c, d, epsabs=1.49e-2, epsrel=1.49e-2):
    """Compute integral squared error between two 2D densities.

    Parameters
    ----------
    p1: callable
        first density, p1([x,y])
    p2: callable
        second density, p2([x,y])
    a, b : float
        The limits of integration in x: a<b
    c, d : float
        The limits of integration in y: c<d
    epsabs : float, optional
        absolute error tolerance for numerical integration
    epsrel : float, optional
        relative error tolerance for numerical integration


    Returns
    -------
    ise: foat
        kld
    int_error:
        numerical integration estimated error

    See Also
    --------
    normalized_integral_squared_error_2d : compute normalized integral squared
        error between two 2D densities
    l2_dist : compute L2 distance (ISE) between two Gaussian mixtures

    Notes
    -----
    This function is intended for use with generic callable densities and
    makes no assumptions about the form of the densities. If both p1 and p2
    are Gaussian mixtures, use l2_dist instead, which is exact and more
    efficient.

    """

    def integrand_fun(y, x):
        return (p1([x, y]) - p2([x, y])) ** 2

    return dblquad(integrand_fun, a, b, c, d, epsabs=epsabs, epsrel=epsrel)


def normalized_integral_squared_error_2d(
    p1, p2, a, b, c, d, epsabs=1.49e-2, epsrel=1.49e-2
):
    """Compute normalized integral squared error between two 2D, densities.

    Parameters
    ----------
    p1: callable
        first density
    p2: callable
        second density
    a, b : float
        The limits of integration in x: a<b
    c, d : float
        The limits of integration in y: c<d
    epsabs : float, optional
        absolute error tolerance for numerical integration
    epsrel : float, optional
        relative error tolerance for numerical integration

    Returns
    -------
    nise: float
        normalized integral squared error
    ise:
        integral squared error
    err:
        numerical integration estimated error in ISE computation

    See Also
    --------
    integral_squared_error_2d : compute integral squared error between two
        2D densities
    l2_dist : compute L2 distance (ISE) between two Gaussian mixtures

    Notes
    -----
    This function is intended for use with generic callable densities and
    makes no assumptions about the form of the densities. If both p1 and p2
    are Gaussian mixtures, use metrics.l2_dist and gm.integral_squared_gm
    instead for the numerator and denominator terms separately, which is
    exact and more efficient.

    """
    ise, err = integral_squared_error_2d(
        p1, p2, a, b, c, d, epsabs=epsabs, epsrel=epsrel
    )
    # if p1 is a GaussianMixtureRv, use l2_dist
    if isinstance(p1, pygm.GaussianMixture):
        int_p1_sq = pygm.integral_squared_gm(p1)
    else:

        def p1_sq_integrand_fun(y, x):
            return p1([x, y]) ** 2

        int_p1_sq = dblquad(
            p1_sq_integrand_fun, a, b, c, d, epsabs=epsabs, epsrel=epsrel
        )[0]
    if isinstance(p2, pygm.GaussianMixture):
        int_p2_sq = pygm.integral_squared_gm(p2)
    else:

        def p2_sq_integrand_fun(y, x):
            return p2([x, y]) ** 2

        int_p2_sq = dblquad(
            p2_sq_integrand_fun, a, b, c, d, epsabs=epsabs, epsrel=epsrel
        )[0]
    nise = ise / (int_p1_sq + int_p2_sq)

    return nise, ise, err


def _weev1_standard_normal(x):
    """Compute WEEV-1 measure of error samples generated from standard normal distribution.

    Parameters
    ----------
    x: ndarray
        An array of shape (n_samples, n) containing the samples.

    Returns
    -------
    ndarray
        An array of shape (n_samples,) containing the WEEV-1 measure for each sample.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    return np.sum(np.abs(x), axis=-1)


def _weev2_standard_normal(x):
    """Compute WEEV-2 measure of samples generated from standard normal distribution.

    Parameters
    ----------
    x : ndarray
        An array of shape (n_samples, n) containing the samples.

    Returns
    -------
    ndarray
        An array of shape (n_samples,) containing the WEEV-2 measure for each sample.

    """
    return np.sqrt(np.sum(x**2, axis=-1))


def _weevinf_standard_normal(x):
    """Compute WEEV-inf measure of error samples generated from standard normal distribution.

    Parameters
    ----------
    x: ndarray
        An array of shape (n_samples, n) containing the samples.

    Returns
    -------
    ndarray
        An array of shape (n_samples,) containing the WEEV-inf measure for each sample.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    return np.max(np.abs(x), axis=-1)


def weev1_log_mgf(t, n):
    """Compute the log of the moment generating function for WEEV-1.

    Parameters
    ----------
    t : float
        The point at which to evaluate the log of the moment generating function.
    n : int
        The dimension of the standard normal distribution.

    Returns
    -------
    float
        The log of the moment generating function at t.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # For small |t|, log(2) + log_ndtr(t) suffers catastrophic cancellation.
    # Instead, use the identity : log(2·Phi(t)) = log1p(erf(t/sqrt(2))), where erf(small) is
    # computed accurately and log1p handles it without cancellation.
    erf_saturation_threshold = 4.0
    if abs(t) < erf_saturation_threshold:
        return n * (np.log1p(erf(t / np.sqrt(2))) + t**2 / 2)
    return n * (np.log(2) + t**2 / 2 + log_ndtr(t))


def _weev1_ecdf(n, n_samples=int(1e7), rng=None):
    """Compute the empirical CDF of the WEEV-1 measure for a given dimension.

    Parameters
    ----------
    n : int
        The dimension of the standard normal distribution.
    n_samples : int, optional
        The number of samples to generate (default is 1e7).
    rng : numpy.random.Generator, optional
        The random number generator (default is None).

    Returns
    -------
    weev1_cdf : scipy.stats.ecdf.cdf
        The empirical CDF
    weev1s : np.array
        The (n_samples,) array of WEEV-1 samples

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    # generate samples from a standard normal distribution
    samples = rng.normal(size=(n_samples, n))
    # compute the WEEV-1 of each error vector sample
    weev1s = _weev1_standard_normal(samples)
    weev1_cdf = ecdf(weev1s).cdf
    return weev1_cdf, weev1s


def confidence_bounds_from_samples(samps, alpha):
    """Compute confidence bounds from a set of samples.

    Parameters
    ----------
    samps : np.array
        The (n_samples,) array of samples.
    alpha : float
        The significance level for the confidence interval.

    Returns
    -------
    r1 : float
        The lower confidence bound.
    r2 : float
        The upper confidence bound.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    r1 = np.quantile(samps, alpha / 2)
    r2 = np.quantile(samps, 1 - alpha / 2)
    return r1, r2


def weev1_chernoff_cdf_lower_bound(a, n):
    """Compute the Chernoff lower bound for the WEEV-1 CDF at variate a.

    Parameters
    ----------
    a : float
        The variate at which to evaluate the CDF.
    n : int
        The dimension of the standard normal distribution.

    Returns
    -------
    C : float
        The Chernoff lower bound of the WEEV-1 CDF. Pr(X<=a) <= C
    t_chernoff : float
        The optimal value of t for the Chernoff bound.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # if a is greater than the mean of the distribution, the left tail bound is trivial (i.e., 1), so we can skip the optimization
    if a >= weev1_dlog_mgf(0, n):
        return 1.0, 0.0

    # find the chernoff bound for a given cdf value a
    # Pr(X<=a) <= inf_{t<0}(M(t)*e^{-ta}); (left tail)
    def obj_fun(t):
        return weev1_log_mgf(t, n) - t * a

    # the bound -200n/rval was determined through experimentation and works well for n=50/rval=0.001 as well as n=2/rval=0.001
    opt = minimize_scalar(
        obj_fun, bounds=(-200 * n / a, 0), method="bounded", options={"xatol": 1e-16}
    )
    assert opt.success, "Optimization for left bound did not succeed"
    return np.exp(opt.fun), opt.x


def weev1_chernoff_ccdf_upper_bound(a, n):
    """Compute the Chernoff upper bound for the WEEV-1 CCDF at variate a.

    Parameters
    ----------
    a : float
        The variate at which to evaluate the CCDF.
    n : int
        The dimension of the standard normal distribution.

    Returns
    -------
    C : float
        The Chernoff upper bound of the WEEV-1 CCDF. Pr(X>=a) <= C
    t_chernoff : float
        The optimal value of t for the Chernoff bound.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # find the chernoff bound for the weev1 cdf evaluated at variate a
    # Pr(X<=a) <= inf_{t>0}(M(t)*e^{-tr}); (right tail)
    # if a is less than the mean of the distribution, the right tail bound is trivial (i.e., 1), so we can skip the optimization
    if a <= weev1_dlog_mgf(0, n):
        return 1.0, 0.0

    def obj_fun(t):
        return weev1_log_mgf(t, n) - t * a

    # TODO: come up with a more principled way to determine the optimization bounds here.
    opt = minimize_scalar(
        obj_fun, bounds=(0, 200 * n / a), method="bounded", options={"xatol": 1e-16}
    )
    assert opt.success, "Optimization for right bound did not succeed"
    return np.exp(opt.fun), opt.x


def weev1_chernoff_confidence_interval(n, alpha):
    """Compute the Chernoff confidence interval for the WEEV-1 measure.

    Parameters
    ----------
    n : int
        The dimension of the standard normal distribution.
    alpha : float
        The significance level for the confidence interval.

    Returns
    -------
    r1 : float
        The lower confidence bound for WEEV-1.
    r2 : float
        The upper confidence bound for WEEV-1.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # a reasonable choice for an upper bound on the right bound is to take the associated
    # normal distribution bound (e.g., for 3-sigma), then multiply it by the dimension, which
    # would be the equivalent weev-1 norm associated with all elements being at that limit.
    ub_r2 = chi2.ppf(1 - alpha / 2, 1) * n
    lb_r1 = chi2.ppf(
        alpha / 2, 1
    )  # this is a very loose lower bound, but it works well in practice for root finding initialization
    r1_sol = root_scalar(
        lambda r: weev1_chernoff_cdf_lower_bound(r, n)[0] - alpha / 2,
        bracket=(lb_r1, 10 * n),
    )
    assert r1_sol.converged, "Root finding for left bound did not converge"
    r1 = r1_sol.root
    r2_sol = root_scalar(
        lambda r: weev1_chernoff_ccdf_upper_bound(r, n)[0] - alpha / 2,
        bracket=(r1, ub_r2),
    )
    assert r2_sol.converged, "Root finding for right bound did not converge"
    r2 = r2_sol.root
    return r1, r2


def weev1_dlog_mgf(t, n):
    r"""Compute the derivative of the log of the moment generating function for WEEV-1 using a numerically stable approach.

    Parameters
    ----------
    t : float
        The point at which to evaluate the log of the moment generating function.
    n : int
        The dimension of the state.

    Returns
    -------
    dlog_mgf : float
        The derivative of the log of the moment generating function for WEEV-1.

    Notes
    -----
    For WEEV-1, the derivative of the cumulant generating function is given by:
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
    return n * (t + np.sqrt(2 / np.pi) / erfcx(-t / np.sqrt(2)))


def weev1_confidence_interval(
    n, alpha, *, n_samples=int(1e7), rng=None, clear_cache=False
):
    r"""Compute confidence bounds for the WEEV-1 measure using an empirical CDF approach.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals
    n_samples : int, optional
        Number of samples to generate for the empirical CDF (default is 1e6).
    rng : numpy.random.Generator, optional
        Random number generator (default is None).
    clear_cache : bool, optional
        Whether to clear the cache before generating new samples (default is False).

    Returns
    -------
    r1 : float
        The lower confidence bound for WEEV-1.
    r2 : float
        The upper confidence bound for WEEV-1.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # open cache if it exists, otherwise create it
    weev1_cache = Cache(__file__[:-3] + "aweev1_cache")
    weev1_cache.clear() if clear_cache else None

    # first check cache of bounds
    bound_key = (alpha, n, n_samples, rng)
    if weev1_cache.get(bound_key) is not None:
        r1, r2 = weev1_cache.get(bound_key)
        return r1, r2

    # bounds not in cache. Check to see if samples exist for this state dimension
    weev1_sample_cache = Cache(__file__[:-3] + "aweev1_sample_cache")
    weev1_sample_cache.clear() if clear_cache else None
    sample_key = (n, n_samples, rng)
    if weev1_sample_cache.get(sample_key) is not None:
        weev1_samps = weev1_sample_cache.get(sample_key)
        r1, r2 = confidence_bounds_from_samples(weev1_samps, alpha)
        weev1_cache.set(bound_key, (r1, r2))
        return r1, r2

    # sample cache does not exist, so generate it
    weev1_samps = _weev1_ecdf(n, n_samples=n_samples, rng=rng)[1]
    weev1_sample_cache.set(sample_key, weev1_samps)
    r1, r2 = confidence_bounds_from_samples(weev1_samps, alpha)
    weev1_cache.set(bound_key, (r1, r2))
    return r1, r2


def _aweev1_ecdf(num_trials, weev1_samples, n_samples=int(1e6), rng=None):
    """Compute the empirical CDF of the AWEEV-1 measure for a given number of trials.

    Parameters
    ----------
    num_trials : int
        the number of trials used in the estimator credibility analysis.
    weev1_samples : np.array
        the (n_samples,) array of weev-1 samples.
    n_samples : int, optional
        the number of samples to generate per trial (default is 1e7).
    rng : numpy.random.generator, optional
        the random number generator (default is none).

    Returns
    -------
    aweev1_cdf : scipy.stats.ecdf.cdf
        the empirical cdf of aweev-1.
    aweev1_samps : np.array
        the (num_trials,) array of aweev-1 samples.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # Compute the mean over the num_trials axis in place using
    # Welford's online update:
    #
    #     mean_k = mean_{k-1} + (x_k - mean_{k-1}) / k
    #
    # This keeps memory usage low while avoiding overflows.
    weev1_samples = np.asarray(weev1_samples)

    rng = np.random.default_rng(seed=0) if rng is None else rng
    aweev1_samps = np.zeros(n_samples, dtype=np.float64)
    row = np.empty(n_samples, dtype=np.float64)
    for k in range(1, num_trials + 1):
        # draw one trial's worth of samples into a single (n_samples,) buffer
        row[:] = rng.choice(weev1_samples, size=n_samples, replace=True)
        row -= aweev1_samps  # x_k - mean_{k-1}        (in place)
        row /= k  # (x_k - mean_{k-1}) / k  (in place)
        aweev1_samps += row  # mean_k                  (in place)

    aweev1_cdf = ecdf(aweev1_samps).cdf
    return aweev1_cdf, aweev1_samps


def aweev1_confidence_interval(
    n, alpha, num_trials, *, n_samples=int(1e7), rng=None, clear_cache=False
):
    """Compute confidence bounds for the AWEEV-1 measure using an empirical CDF approach.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals
    num_trials : int
        The number of trials used in the estimator credibility analysis.
    n_samples : int, optional
        Number of samples to generate for the empirical CDF (default is 1e7).
    rng : numpy.random.Generator, optional
        Random number generator (default is None).
    clear_cache : bool, optional
        Whether to clear the cache before generating new samples (default is False).

    Returns
    -------
    r1 : float
        The lower confidence bound for AWEEV-1.
    r2 : float
        The upper confidence bound for AWEEV-1.

    Notes
    -----
    AWEEV statistics rely on empirical CDFs, which require a one-time cost
    to generate samples and compute the associated confidence bounds. Thus,
    this function may take a long time to run the first time for a given state
    dimension and number of trials, but subsequent calls with the same parameters
    will be fast due to caching of results.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # open cache if it exists, otherwise create it
    aweev1_cache = Cache(__file__[:-3] + "aweev1_cache")
    aweev1_cache.clear() if clear_cache else None

    # first check cache of bounds
    bound_key = (alpha, n, num_trials, n_samples, rng)
    if aweev1_cache.get(bound_key) is not None:
        r1, r2 = aweev1_cache.get(bound_key)
        return r1, r2

    # bounds not in cache. Check to see if weev1 samples exist for this state dimension
    aweev1_sample_cache = Cache(__file__[:-3] + "aweev1_sample_cache")
    aweev1_sample_cache.clear() if clear_cache else None
    sample_key = (n, n_samples, num_trials, rng)
    if aweev1_sample_cache.get(sample_key) is not None:
        aweev1_samps = aweev1_sample_cache.get(sample_key)
        r1, r2 = confidence_bounds_from_samples(aweev1_samps, alpha)
        aweev1_cache.set(bound_key, (r1, r2))
        return r1, r2

    # no sample cache exists, so generate
    weev1_samps = _weev1_ecdf(n, n_samples=n_samples)[1]
    aweev1_samps = _aweev1_ecdf(num_trials, weev1_samps, n_samples=n_samples, rng=rng)[1]
    aweev1_sample_cache.set(sample_key, aweev1_samps)
    r1, r2 = confidence_bounds_from_samples(aweev1_samps, alpha)
    aweev1_cache.set(bound_key, (r1, r2))
    return r1, r2

def _mweev1_ecdf(num_trials, weev1_samples, n_samples=int(1e7), rng=None):
    """Compute the empirical CDF of the MWEEV-1 measure for a given number of trials.

    Parameters
    ----------
    num_trials : int
        the number of trials used in the estimator credibility analysis.
    weev1_samples : np.array
        the (n_samples,) array of weev-1 samples.
    n_samples : int, optional
        the number of samples to generate per trial (default is 1e7).
    rng : numpy.random.generator, optional
        the random number generator (default is none).

    Returns
    -------
    mweev1_cdf : scipy.stats.ecdf.cdf
        the empirical cdf of mweev-1.
    mweev1_samps : np.array
        the (num_trials,) array of mweev-1 samples.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    weev1_trial_samps = np.random.choice(
        weev1_samples, size=(num_trials, n_samples), replace=True
    )
    mweev1_samps = np.max(weev1_trial_samps, axis=0)
    mweev1_cdf = ecdf(mweev1_samps).cdf
    return mweev1_cdf, mweev1_samps


def mweev1_confidence_interval(
    n: int,
    alpha: float,
    num_trials: int,
    *,
    n_samples=int(1e7),
    rng=None,
    clear_cache=False,
) -> tuple[float, float]:
    """Compute confidence bounds for the MWEEV-1 measure using an empirical CDF approach.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals
    num_trials : int
        The number of trials used in the estimator credibility analysis.
    n_samples : int, optional
        Number of samples to generate for the empirical CDF (default is 1e7).
    rng : numpy.random.Generator, optional
        Random number generator (default is None).
    clear_cache : bool, optional
        Whether to clear the cache before generating new samples (default is False).

    Returns
    -------
    r1 : float
        The lower confidence bound for MWEEV-1.
    r2 : float
        The upper confidence bound for MWEEV-1.

    Notes
    -----
    MWEEV statistics rely on empirical CDFs, which require a one-time cost
    to generate samples and compute the associated confidence bounds. Thus,
    this function may take a long time to run the first time for a given state
    dimension and number of trials, but subsequent calls with the same parameters
    will be fast due to caching of results.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # open cache if it exists, otherwise create it
    mweev1_cache = Cache(__file__[:-3] + "mweev1_cache")
    mweev1_cache.clear() if clear_cache else None

    # first check cache of bounds
    bound_key = (alpha, n, num_trials, n_samples, rng)
    if mweev1_cache.get(bound_key) is not None:
        r1, r2 = mweev1_cache.get(bound_key)
        return r1, r2

    # bounds not in cache. Check to see if weev1 samples exist for this state dimension
    mweev1_sample_cache = Cache(__file__[:-3] + "mweev1_sample_cache")
    mweev1_sample_cache.clear() if clear_cache else None
    sample_key = (n, n_samples, num_trials, rng)
    if mweev1_sample_cache.get(sample_key) is not None:
        mweev1_samps = mweev1_sample_cache.get(sample_key)
        r1, r2 = confidence_bounds_from_samples(mweev1_samps, alpha)
        mweev1_cache.set(bound_key, (r1, r2))
        return r1, r2

    # no sample cache exists, so generate
    weev1_samps = _weev1_ecdf(n, n_samples=n_samples)[1]
    mweev1_samps = _mweev1_ecdf(
        num_trials, weev1_samps, n_samples=n_samples, rng=rng
    )[1]
    mweev1_sample_cache.set(sample_key, mweev1_samps)
    r1, r2 = confidence_bounds_from_samples(mweev1_samps, alpha)
    mweev1_cache.set(bound_key, (r1, r2))
    return r1, r2


def weev1(e, P, whitening_transform="zca-cor"):
    """Compute whitened estimation error 1-norm (WEEV-1) measure of error.

    Parameters
    ----------
    e : np.ndarray
        (n,) or (N, n) estimation error vector, where n is the state dimension and N is the number
        of samples (if e is a set of samples)
    P : np.ndarray
        (n, n) or (N, n, n) covariance matrix reported by estimator

    Returns
    -------
    float or np.ndarray
        WEEV-1 measure of error. If e and P are both (n,) and (n, n), respectively, this returns a
        single float. If e and P are both (N, n) and (N, n, n), respectively, this returns an array
        of shape (N,) containing the WEEV-1 measure for each sample.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    if whitening_transform != "zca-cor":
        raise ValueError(
            "Invalid whitening transform specified. Only 'zca-cor' is currently supported."
        )

    # handle different dimensions of e and P
    if e.ndim == 1 and P.ndim == 2:
        e_whitened, W = zca_cor_whiten_rv(e, P)
        return np.sum(np.abs(e_whitened))
    if e.ndim == 2 and P.ndim == 3:
        N, n = e.shape
        e_whitened_samples = np.zeros_like(e)
        for i in range(N):
            e_whitened_samples[i], W = zca_cor_whiten_rv(e[i], P[i])
        return np.sum(np.abs(e_whitened_samples), axis=-1)
    raise ValueError(
        "Invalid dimensions for e and P. e should be (n,) or (N, n) and P should be (n, n) or (N, n, n)."
    )


def weev2_confidence_interval(n, alpha):
    r"""Compute confidence bounds for the WEEV-2 measure using the chi distribution.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals

    Returns
    -------
    r1 : float
        The lower confidence bound for WEEV-2.
    r2 : float
        The upper confidence bound for WEEV-2.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    r1 = chi.ppf(alpha / 2, n)
    r2 = chi.ppf(1 - alpha / 2, n)

    return r1, r2


def weev2(e, P, whitening_transform="zca-cor"):
    """Compute whitened estimation error 2-norm (WEEV-2) measure of error.

    Parameters
    ----------
    e : np.ndarray
        (n,) or (N, n) estimation error vector, where n is the state dimension and N is the number of samples (if e is a set of samples)
    P : np.ndarray
        (n, n) or (N, n, n) covariance matrix reported by estimator

    Returns
    -------
    float or np.ndarray
        WEEV-2 measure of error. If e and P are both (n,) and (n, n), respectively, this returns a single float.
        If e and P are both (N, n) and (N, n, n), respectively, this returns an array of shape (N,) containing
        the WEEV-2 measure for each sample.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    if whitening_transform != "zca-cor":
        raise ValueError(
            "Invalid whitening transform specified. Only 'zca-cor' is currently supported."
        )

    # handle different dimensions of e and P
    if e.ndim == 1 and P.ndim == 2:
        e_whitened, W = zca_cor_whiten_rv(e, P)
        return np.sqrt(np.sum(e_whitened**2))
    if e.ndim == 2 and P.ndim == 3:
        N, n = e.shape
        e_whitened_samples = np.zeros_like(e)
        for i in range(N):
            e_whitened_samples[i], W = zca_cor_whiten_rv(e[i], P[i])
        return np.sqrt(np.sum(e_whitened_samples**2, axis=-1))
    raise ValueError(
        "Invalid dimensions for e and P. e should be (n,) or (N, n) and P should be (n, n) or (N, n, n)."
    )


def _weevinf_ecdf(n, n_samples=int(1e7), rng=None):
    """Compute the empirical CDF of the WEEV-inf measure for a given dimension.

    Parameters
    ----------
    n : int
        The dimension of the standard normal distribution.
    n_samples : int, optional
        The number of samples to generate (default is 1e7).
    rng : numpy.random.Generator, optional
        The random number generator (default is None).

    Returns
    -------
    weevinf_cdf : scipy.stats.ecdf.cdf
        The empirical CDF
    weevinfs : np.array
        The (n_samples,) array of WEEV-inf samples

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    # generate samples from a standard normal distribution
    samples = rng.normal(size=(n_samples, n))
    # compute the WEEV-inf of each error vector sample
    weevinfs = _weevinf_standard_normal(samples)
    weevinf_cdf = ecdf(weevinfs).cdf
    return weevinf_cdf, weevinfs

def weevinf_confidence_interval(n, alpha):
    """Compute the confidence interval for the WEEV-inf measure using the semi-analytical approach.

    Parameters
    ----------
    n : int
        The dimension of the standard normal distribution.
    alpha : float
        The significance level for the confidence interval.

    Returns
    -------
    r1 : float
        The lower confidence bound for WEEV-inf.
    r2 : float
        The upper confidence bound for WEEV-inf.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    r1 = np.sqrt(2)*erfinv((alpha/2)**(1/n))
    r2 = np.sqrt(2)*erfinv((1-alpha/2)**(1/n))

    return r1, r2

def _weevinf_confidence_interval_empirical(
    n, alpha, *, n_samples=int(1e7), rng=None, clear_cache=False
):
    r"""Compute confidence bounds for the WEEV-inf measure using an empirical CDF approach.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals
    n_samples : int, optional
        Number of samples to generate for the empirical CDF (default is 1e7).
    rng : numpy.random.Generator, optional
        Random number generator (default is None).
    clear_cache : bool, optional
        Whether to clear the cache before generating new samples (default is False).

    Returns
    -------
    r1 : float
        The lower confidence bound for WEEV-inf.
    r2 : float
        The upper confidence bound for WEEV-inf.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # open cache if it exists, otherwise create it
    weevinf_cache = Cache(__file__[:-3] + "weevinf_cache")
    weevinf_cache.clear() if clear_cache else None

    # first check cache of bounds
    bound_key = (alpha, n, n_samples, rng)
    if weevinf_cache.get(bound_key) is not None:
        r1, r2 = weevinf_cache.get(bound_key)
        return r1, r2

    # bounds not in cache. Check to see if samples exist for this state dimension
    weevinf_sample_cache = Cache(__file__[:-3] + "weevinf_sample_cache")
    weevinf_sample_cache.clear() if clear_cache else None
    sample_key = (n, n_samples, rng)
    if weevinf_sample_cache.get(sample_key) is not None:
        weevinf_samps = weevinf_sample_cache.get(sample_key)
        r1, r2 = confidence_bounds_from_samples(weevinf_samps, alpha)
        weevinf_cache.set(bound_key, (r1, r2))
        return r1, r2

    # sample cache does not exist, so generate it
    _, weevinf_samps = _weevinf_ecdf(n, n_samples=n_samples, rng=rng)
    weevinf_sample_cache.set(sample_key, weevinf_samps)
    r1, r2 = confidence_bounds_from_samples(weevinf_samps, alpha)
    weevinf_cache.set(bound_key, (r1, r2))
    return r1, r2


def weevinf(e, P, whitening_transform="zca-cor"):
    """Compute whitened estimation error inf-norm (WEEV-inf) measure of error.

    Parameters
    ----------
    e : np.ndarray
        (n,) or (N, n) estimation error vector, where n is the state dimension and N is the number of samples (if e is a set of samples)
    P : np.ndarray
        (n, n) or (N, n, n) covariance matrix reported by estimator

    Returns
    -------
    float or np.ndarray
        The WEEV-inf measure of error. If e and P are both (n,) and (n, n), respectively, this returns a single float.
        If e and P are both (N, n) and (N, n, n), respectively, this returns an array of shape (N,) containing
        the WEEV-inf measure for each sample.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    if whitening_transform != "zca-cor":
        raise ValueError(
            "Invalid whitening transform specified. Only 'zca-cor' is currently supported."
        )

    # handle different dimensions of e and P
    if e.ndim == 1 and P.ndim == 2:
        e_whitened, W = zca_cor_whiten_rv(e, P)
        return np.linalg.norm(e_whitened, ord=np.inf)
    if e.ndim == 2 and P.ndim == 3:
        N, n = e.shape
        e_whitened_samples = np.zeros_like(e)
        for i in range(N):
            e_whitened_samples[i], W = zca_cor_whiten_rv(e[i], P[i])
        return np.linalg.norm(e_whitened_samples, ord=np.inf, axis=-1)
    raise ValueError(
        "Invalid dimensions for e and P. e should be (n,) or (N, n) and P should be (n, n) or (N, n, n)."
    )


def _aweevinf_ecdf(num_trials, weevinf_samples, n_samples=int(1e7), rng=None):
    """Compute the empirical CDF of the AWEEV-inf measure for a given number of trials.

    Parameters
    ----------
    num_trials : int
        the number of trials used in the estimator credibility analysis.
    weevinf_samples : np.array
        the (n_samples,) array of weev-inf samples.
    n_samples : int, optional
        the number of samples to generate per trial (default is 1e7).
    rng : numpy.random.generator, optional
        the random number generator (default is none).

    Returns
    -------
    aweevinf_cdf : scipy.stats.ecdf.cdf
        the empirical cdf of aweev-inf.
    aweevinf_samps : np.array
        the (num_trials,) array of aweev-inf samples.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    weevinf_trial_samps = np.random.choice(
        weevinf_samples, size=(num_trials, n_samples), replace=True
    )
    aweevinf_samps = np.mean(weevinf_trial_samps, axis=0)  # arithmetic mean
    aweevinf_cdf = ecdf(aweevinf_samps).cdf
    return aweevinf_cdf, aweevinf_samps


def aweevinf_confidence_interval(
    n: int,
    alpha: float,
    num_trials: int,
    *,
    n_samples=int(1e7),
    rng=None,
    clear_cache=False,
) -> tuple[float, float]:
    """Compute confidence bounds for the AWEEV-inf measure using an empirical CDF approach.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals
    num_trials : int
        The number of trials used in the estimator credibility analysis.
    n_samples : int, optional
        Number of samples to generate for the empirical CDF (default is 1e7).
    rng : numpy.random.Generator, optional
        Random number generator (default is None).
    clear_cache : bool, optional
        Whether to clear the cache before generating new samples (default is False).

    Returns
    -------
    r1 : float
        The lower confidence bound for AWEEV-inf.
    r2 : float
        The upper confidence bound for AWEEV-inf.

    Notes
    -----
    AWEEV statistics rely on empirical CDFs, which require a one-time cost
    to generate samples and compute the associated confidence bounds. Thus,
    this function may take a long time to run the first time for a given state
    dimension and number of trials, but subsequent calls with the same parameters
    will be fast due to caching of results.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # open cache if it exists, otherwise create it
    aweevinf_cache = Cache(__file__[:-3] + "aweevinf_cache")
    aweevinf_cache.clear() if clear_cache else None

    # first check cache of bounds
    bound_key = (alpha, n, num_trials, n_samples, rng)
    if aweevinf_cache.get(bound_key) is not None:
        r1, r2 = aweevinf_cache.get(bound_key)
        return r1, r2

    # bounds not in cache. Check to see if weev1 samples exist for this state dimension
    aweevinf_sample_cache = Cache(__file__[:-3] + "aweevinf_sample_cache")
    aweevinf_sample_cache.clear() if clear_cache else None
    sample_key = (n, n_samples, num_trials, rng)
    if aweevinf_sample_cache.get(sample_key) is not None:
        aweevinf_samps = aweevinf_sample_cache.get(sample_key)
        r1, r2 = confidence_bounds_from_samples(aweevinf_samps, alpha)
        aweevinf_cache.set(bound_key, (r1, r2))
        return r1, r2

    # no sample cache exists, so generate
    weevinf_samps = _weevinf_ecdf(n, n_samples=n_samples)[1]
    aweevinf_samps = _aweevinf_ecdf(
        num_trials, weevinf_samps, n_samples=n_samples, rng=rng
    )[1]
    aweevinf_sample_cache.set(sample_key, aweevinf_samps)
    r1, r2 = confidence_bounds_from_samples(aweevinf_samps, alpha)
    aweevinf_cache.set(bound_key, (r1, r2))
    return r1, r2


def _mweevinf_ecdf(num_trials, weevinf_samples, n_samples=int(1e7), rng=None):
    """Compute the empirical CDF of the MWEEV-inf measure for a given number of trials.

    Parameters
    ----------
    num_trials : int
        the number of trials used in the estimator credibility analysis.
    weevinf_samples : np.array
        the (n_samples,) array of weev-inf samples.
    n_samples : int, optional
        the number of samples to generate per trial (default is 1e7).
    rng : numpy.random.generator, optional
        the random number generator (default is none).

    Returns
    -------
    mweevinf_cdf : scipy.stats.ecdf.cdf
        the empirical cdf of aweev-inf.
    mweevinf_samps : np.array
        the (num_trials,) array of aweev-inf samples.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    weevinf_trial_samps = np.random.choice(
        weevinf_samples, size=(num_trials, n_samples), replace=True
    )
    mweevinf_samps = np.max(weevinf_trial_samps, axis=0)
    mweevinf_cdf = ecdf(mweevinf_samps).cdf
    return mweevinf_cdf, mweevinf_samps


def mweevinf_confidence_interval(
        n: int,
        alpha: float,
        num_trials: int
) -> tuple[float, float]:
    """Compute confidence bounds for the MWEEV-inf measure using a semi-analytical approach.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for the confidence interval
    num_trials : int
        The number of trials used in the estimator credibility analysis.

    Returns
    -------
    r1 : float
        The lower confidence bound for MWEEV-inf.
    r2 : float
        The upper confidence bound for MWEEV-inf.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    r1 = np.sqrt(2)*erfinv((alpha/2)**(1/n/num_trials))
    r2 = np.sqrt(2)*erfinv((1-alpha/2)**(1/n/num_trials))

    return r1, r2

def _mweevinf_confidence_interval_empirical(
    n: int,
    alpha: float,
    num_trials: int,
    *,
    n_samples=int(1e7),
    rng=None,
    clear_cache=False,
) -> tuple[float, float]:
    """Compute confidence bounds for the MWEEV-inf measure using an empirical CDF approach.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals
    num_trials : int
        The number of trials used in the estimator credibility analysis.
    n_samples : int, optional
        Number of samples to generate for the empirical CDF (default is 1e7).
    rng : numpy.random.Generator, optional
        Random number generator (default is None).
    clear_cache : bool, optional
        Whether to clear the cache before generating new samples (default is False).

    Returns
    -------
    r1 : float
        The lower confidence bound for MWEEV-inf.
    r2 : float
        The upper confidence bound for MWEEV-inf.

    Notes
    -----
    MWEEV statistics rely on empirical CDFs, which require a one-time cost
    to generate samples and compute the associated confidence bounds. Thus,
    this function may take a long time to run the first time for a given state
    dimension and number of trials, but subsequent calls with the same parameters
    will be fast due to caching of results.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    # open cache if it exists, otherwise create it
    mweevinf_cache = Cache(__file__[:-3] + "mweevinf_cache")
    mweevinf_cache.clear() if clear_cache else None

    # first check cache of bounds
    bound_key = (alpha, n, num_trials, n_samples, rng)
    if mweevinf_cache.get(bound_key) is not None:
        r1, r2 = mweevinf_cache.get(bound_key)
        return r1, r2

    # bounds not in cache. Check to see if weev1 samples exist for this state dimension
    mweevinf_sample_cache = Cache(__file__[:-3] + "mweevinf_sample_cache")
    mweevinf_sample_cache.clear() if clear_cache else None
    sample_key = (n, n_samples, num_trials, rng)
    if mweevinf_sample_cache.get(sample_key) is not None:
        mweevinf_samps = mweevinf_sample_cache.get(sample_key)
        r1, r2 = confidence_bounds_from_samples(mweevinf_samps, alpha)
        mweevinf_cache.set(bound_key, (r1, r2))
        return r1, r2

    # no sample cache exists, so generate
    weevinf_samps = _weevinf_ecdf(n, n_samples=n_samples)[1]
    mweevinf_samps = _mweevinf_ecdf(
        num_trials, weevinf_samps, n_samples=n_samples, rng=rng
    )[1]
    mweevinf_sample_cache.set(sample_key, mweevinf_samps)
    r1, r2 = confidence_bounds_from_samples(mweevinf_samps, alpha)
    mweevinf_cache.set(bound_key, (r1, r2))
    return r1, r2


def nees(e, P):
    """Compute normalized estimation error squared (NEES) measure of error.

    Parameters
    ----------
    e : np.ndarray
        (n,) or (N, n) estimation error vector, where n is the state dimension and N is
        the number of samples (if e is a set of samples)
    P : np.ndarray
        (n, n) or (N, n, n) covariance matrix reported by estimator

    Returns
    -------
    float or np.ndarray
        NEES measure of error

    """
    if e.ndim == 1 and P.ndim == 2:
        return e.T @ np.linalg.solve(P, e)
    if e.ndim == 2 and P.ndim == 3:
        N, n = e.shape
        nees_vals = np.zeros(N)
        for i in range(N):
            nees_vals[i] = e[i].T @ np.linalg.solve(P[i], e[i])
        return nees_vals
    raise ValueError(
        "Invalid dimensions for e and P. e should be (n,) or (N, n) and P should be (n, n) or (N, n, n)."
    )

def anees(e, P):
    """Compute average normalized estimation error squared (ANEES) measure of error.

    Parameters
    ----------
    e : np.ndarray
        (M, N, n) estimation error vector, where n is the state dimension and N is the
        number of time steps and M is the number of Monte Carlo trials
    P : np.ndarray
        (M, N, n, n) covariance matrix reported by estimator

    Returns
    -------
    ndarray
        (N,) array containing the ANEES measure of error for each time step

    """
    M, N, n = e.shape
    nees_vals = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            nees_vals[i, j] = nees(e[i, j], P[i, j])
    return np.mean(nees_vals, axis=0)


def snees(e, P):
    """Compute scaled normalized estimation error squared (SNEES) measure of error.

    Parameters
    ----------
    e : np.ndarray
        (n,) or (N, n) estimation error vector, where n is the state dimension and N is the number of samples (if e is a set of samples)
    P : np.ndarray
        (n, n) or (N, n, n) covariance matrix reported by estimator

    Returns
    -------
    float or np.ndarray
        SNEES measure of error

    """
    n = P.shape[-1]
    return nees(e, P) / n


def asnees(e, P):
    """Compute average scaled normalized estimation error squared (ASNEES) measure of error.

    Parameters
    ----------
    e : np.ndarray
        (M, N, n) estimation error vector, where n is the state dimension and N is the
        number of time steps and M is the number of Monte Carlo trials
    P : np.ndarray
        (M, N, n, n) covariance matrix reported by estimator

    Returns
    -------
    ndarray
        (N,) array containing the ASNEES measure of error for each time step

    """
    M, N, n = e.shape
    snees_vals = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            snees_vals[i, j] = snees(e[i, j], P[i, j])
    return np.mean(snees_vals, axis=0)


def nees_confidence_interval(n, alpha):
    """Compute confidence bounds for NEES measure of error.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals

    Returns
    -------
    r1 : float
        The lower confidence bound for NEES.
    r2 : float
        The upper confidence bound for NEES.

    """
    r1 = chi2.ppf(alpha / 2, n)
    r2 = chi2.ppf(1 - alpha / 2, n)

    return r1, r2


def snees_confidence_interval(n, alpha):
    """Compute confidence bounds for SNEES measure of error.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals

    Returns
    -------
    r1 : float
        The lower confidence bound for SNEES.
    r2 : float
        The upper confidence bound for SNEES.

    """
    r1_nees, r2_nees = nees_confidence_interval(n, alpha)
    r1 = r1_nees / n
    r2 = r2_nees / n

    return r1, r2

def anees_confidence_interval(n, alpha, num_trials):
    """Compute confidence bounds for ANEES measure of error.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals
    num_trials : int
        Number of Monte Carlo trials


    Returns
    -------
    r1 : float
        The lower confidence bound for ANEES.
    r2 : float
        The upper confidence bound for ANEES.

    """
    r1 = chi2.ppf(alpha / 2, num_trials * n) / num_trials
    r2 = chi2.ppf(1 - alpha / 2, num_trials * n) / num_trials

    return r1, r2


def asnees_confidence_interval(n, alpha, num_trials):
    """Compute confidence bounds for ASNEES measure of error.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals
    num_trials : int
        Number of Monte Carlo trials


    Returns
    -------
    r1 : float
        The lower confidence bound for ASNEES.
    r2 : float
        The upper confidence bound for ASNEES.

    """
    r1 = chi2.ppf(alpha / 2, num_trials * n) / (num_trials * n)
    r2 = chi2.ppf(1 - alpha / 2, num_trials * n) / (num_trials * n)

    return r1, r2

def mnees_confidence_interval(
    n: int,
    alpha: float,
    num_trials: int,
) -> tuple[float, float]:
    """Compute confidence bounds for the MNEES measure.

    Parameters
    ----------
    n : int
        State dimension
    alpha : float
        False alarm rate for confidence intervals
    num_trials : int
        The number of trials used in the estimator credibility analysis.

    Returns
    -------
    r1 : float
        The lower confidence bound for MNEES.
    r2 : float
        The upper confidence bound for MNEES.

    References
    ----------
    .. [1] Tim Goulet, Keith A. LeGrand, and Jackson Kulik, "State Estimator
           Credibility Analysis with Whitened Error Vector Norms," 2026

    """
    r1 = chi2.ppf((alpha / 2)**(1/num_trials), n)
    r2 = chi2.ppf((1 - alpha / 2)**(1/num_trials), n)
    return r1, r2


def standardize_rv(e, P):
    """Standardize the zero-mean random variable by transforming it to have unit variances.

    Parameters
    ----------
    e : np.ndarray
        (n,) estimation error vector
    P : np.ndarray
        (n, n) covariance matrix

    Returns
    -------
    e_std : np.ndarray
        (n,) standardized estimation error vector (e_std = Dinv @ e, where Dinv is the standardization matrix)
    corr : np.ndarray
        (n, n) correlation matrix
    Dinv : np.ndarray
        (n, n) diagonal matrix with elements equal to the inverse of the standard deviations of
        the original random variable (i.e., Dinv @ P @ Dinv has 1s on the diagonal)

    """
    # standardize the random variable, such that that variances are equal to 1
    Dinv = np.diag(
        1 / np.sqrt(np.diag(P))
    )  # standardization matrix (Dinv @ P @ Dinv has 1s on the diagonal)
    e_std = Dinv @ e
    corr = Dinv @ P @ Dinv  # correlation matrix (diagonal elements are 1)
    return e_std, corr, Dinv


def zca_whiten_standardized_rv(e_std, corr, Dinv):
    """Perform ZCA whitening on the standardized random variable using the correlation matrix.

    Parameters
    ----------
    e_std : np.ndarray
        (n,) standardized estimation error vector
    corr : np.ndarray
        (n, n) correlation matrix
    Dinv : np.ndarray
        (n, n) diagonal matrix with elements equal to the inverse of the standard deviations of
        the original random variable (i.e., Dinv @ P @ Dinv has 1s on the diagonal). Used only
        for computing the whitening matrix W, which can be applied to the original (non-standardized)
        error vector e to obtain the whitened error vector.

    Returns
    -------
    e_whitened : np.ndarray
        (n,) whitened estimation error vector
    W : np.ndarray
        (n, n) whitening matrix such that e_whitened = W @ e

    """
    # whiten the standardized random variable using ZCA whitening based on the correlation rather than covariance matrix
    # W = corr^{-1/2}D^{-1}, first standardizing the random variable by multiplication with D^{-1} and subsequently employing
    # ZCA whitening based on the correlation rather than covariance matrix
    # cormat is the correlation matrix
    corr_sqrt = np.linalg.cholesky(corr)
    e_whitened = solve_triangular(corr_sqrt, e_std, lower=True)
    W = solve_triangular(corr_sqrt, Dinv, lower=True)
    return e_whitened, W


def zca_cor_whiten_rv(e, P):
    """Perform ZCA-cor whitening based on the covariance matrix.

    Parameters
    ----------
    e : np.ndarray
        (n,) estimation error vector
    P : np.ndarray
        (n, n) covariance matrix

    Returns
    -------
    e_whitened : np.ndarray
        (n,) whitened estimation error vector
    W : np.ndarray
        (n, n) whitening matrix such that e_whitened = W @ e

    """
    e_std, corr, Dinv = standardize_rv(e, P)
    return zca_whiten_standardized_rv(e_std, corr, Dinv)
