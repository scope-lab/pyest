import numpy as np
import pyest.gm as pygm
from scipy.linalg import solve_triangular


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