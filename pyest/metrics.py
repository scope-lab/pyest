import numpy as np
import pyest.gm as pygm
from scipy.linalg import solve_triangular
from scipy.integrate import dblquad


def l2_dist(p1, p2):
    """ compute L2 distance between GMs p1 and p2

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
    t1 = np.sum([
        wi*wj*pygm.eval_mvnpdf(mi, mj, Pi + Pj) for (wi, mi, Pi) in p1 for (wj, mj, Pj) in p1
    ])
    t2 = np.sum([
        wi*wj*pygm.eval_mvnpdf(mi, mj, Pi + Pj) for (wi, mi, Pi) in p1 for (wj, mj, Pj) in p2
    ])
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
    #t3 = np.sum([
    #    wi*wj*pygm.eval_mvnpdf(mi, mj, Pi + Pj) for (wi, mi, Pi) in p2 for (wj, mj, Pj) in p2
    #])

    l2 = t1 - 2*t2 + t3
    return l2


def max_covariance_ratio(S, S_ref):
    """ compute the maximum covariance ratio between two distributions

    Required:
    ---------
    S : np.ndarray
        covariance matrix lower-triangular Cholesky square-root factor of
        the test distribution
    S_ref : np.ndarray
        covariance matrix lower-triangular Cholesky square-root factor of
        the reference distribution

    Returns:
    --------
    float
        maximum covariance ratio
    """
    mat = solve_triangular(S, S_ref, lower=True)
    s_vals = np.linalg.svd(mat, compute_uv=False, hermitian=False)
    return max(s_vals[0], 1.0 / s_vals[-1])


def madem(m, S, m_ref):
    """ Mahalanobis distance of the error of the mean

    Required:
    ---------
    m : np.ndarray
        mean of the test distribution
    S : np.ndarray
        covariance matrix lower-triangular Cholesky square-root factor of
        the test distribution
    m_ref : np.ndarray
        mean of the reference distribution

    Returns:
    --------
    float
        Mahalanobis distance of the error of the mean (MaDEM)
    """
    return np.linalg.norm(
        solve_triangular(S, m - m_ref, lower=True)
    )


def integral_squared_error_2d(p1, p2, a, b, c, d, epsabs=1.49e-2, epsrel=1.49e-2):
    ''' compute integral squared error between two 2D densities

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
    '''
    integrand_fun = lambda y,x: (p1([x,y])-p2([x,y]))**2
    return dblquad(integrand_fun, a, b, c, d, epsabs=epsabs, epsrel=epsrel)


def normalized_integral_squared_error_2d(p1, p2, a, b, c, d, epsabs=1.49e-2, epsrel=1.49e-2):
    ''' compute normalized integral squared error between two 2D, densities

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

    '''
    ise, err = integral_squared_error_2d(p1, p2, a, b, c, d, epsabs=epsabs, epsrel=epsrel)
    # if p1 is a GaussianMixtureRv, use l2_dist
    if isinstance(p1, pygm.GaussianMixture):
        int_p1_sq = pygm.integral_squared_gm(p1)
    else:
        p1_sq_integrand_fun = lambda y,x: p1([x,y])**2
        int_p1_sq = dblquad(p1_sq_integrand_fun, a, b, c, d, epsabs=epsabs, epsrel=epsrel)[0]
    if isinstance(p2, pygm.GaussianMixture):
        int_p2_sq = pygm.integral_squared_gm(p2)
    else:
        p2_sq_integrand_fun = lambda y,x: p2([x,y])**2
        int_p2_sq = dblquad(p2_sq_integrand_fun, a, b, c, d, epsabs=epsabs, epsrel=epsrel)[0]
    nise = ise/(int_p1_sq + int_p2_sq)

    return nise, ise, err
