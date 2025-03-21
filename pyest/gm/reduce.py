import numpy as np
from numba import jit
from scipy.spatial import KDTree

from .gm import GaussianMixture


@jit
def merge_components(w, m, P):
    """ merge Gaussian mixture components into single component

    Parameters
    ----------
    w: ndarray
      (nC,) component weights
    m: ndarray
      (nC,nX) component means
    P: ndarray
      (nC,nX,nX) component covariances

    Returns
    -------
    float:
      merged weight
    ndarray:
      (nX,) merged mean
    ndarray:
      (nX,nX) merged covariance
    """

    # merge weights
    w_merged = np.sum(w)
    # merge means
    m_merged = 1/w_merged*np.sum(
        np.expand_dims(w, axis=0).T*m,
        axis=0
    )

    # merge covariances
    tempsum = np.zeros(P[0].shape)
    for wi, mi, Pi in zip(w, m, P):
        delm = m_merged - mi
        tempsum += wi*(Pi + np.outer(delm, delm))
    P_merged = 1/w_merged*tempsum
    return w_merged, m_merged, P_merged


def merge(p, md, gate=None):
    """ merge gm components

    Parameters
    ----------
    p : GaussianMixture
        gm to be reduced through component merging
    md : float
        mahalanobis distance threshold. components that within this distance
        of one another are merged
    gate : float, optional
        a course gate value. If the Euclidean distance between two components
        exceeds this gate value, they will not be considered for merging.
        Default is None.

    Returns
    -------
    GaussianMixture

    """

    if gate is not None:
        tree = KDTree(p.m)
    else:
        tree = None

    idxs = np.arange(len(p))
    valid_mask = p.w > 0

    # precompute covariance inverses
    Pinv = np.array([np.linalg.inv(P) for P in p.P])

    # allocate memory
    wmerged = np.full(len(p), np.nan)
    mmerged = np.full((len(p), p.dim), np.nan)
    Pmerged = np.full((len(p), p.dim, p.dim), np.nan)
    new_comp_count = 0

    while any(valid_mask):
        valid_idxs = idxs[valid_mask]

        # find highest weight component
        max_valid_idx_idx = np.argmax(p.w[valid_idxs])
        comp_idx = idxs[valid_idxs[max_valid_idx_idx]]
        # mean with highest weight
        mmax = p.m[comp_idx]

        merge_mask = np.array([False]*len(p))

        if tree is not None:
            in_gate_idxs = tree.query_ball_point(mmax, r=gate)
            in_gate_mask = np.full(idxs.shape, False)
            in_gate_mask[in_gate_idxs] = True
            valid_idxs = idxs[np.logical_and(valid_mask, in_gate_mask)]

        for i in valid_idxs:
            delm = p.m[i] - mmax
            # mahalanobis distance for this component pair
            mahdist = delm@Pinv[i]@delm
            merge_mask[i] = mahdist <= md

        w, m, P = merge_components(*p[merge_mask])

        # store merged components
        wmerged[new_comp_count] = w
        mmerged[new_comp_count] = m
        Pmerged[new_comp_count] = P

        # mark the components we just merged as used
        valid_mask[merge_mask] = False

        new_comp_count += 1

    return GaussianMixture(
        wmerged[:new_comp_count],
        mmerged[:new_comp_count],
        Pmerged[:new_comp_count]
    )


def merge_runnals(p, K):
    """ KL-Divergence based mixture reduction

    Paramters
    ---------
    p : GaussianMixture
        GM that is to be reduced
    K : int
        desired number of components in final mixture

    Returns
    -------
    GaussianMixture
        a reduced GM with K components

    Notes
    -----
    This algorithm is based on the work by Runnals [1].

    This function was ported to python from David F. Crouse's MATLAB tracking
    library.

    References
    ----------
    [1] A. R. Runnalls, "Kullback-Leibler approach to Gaussian mixture
    reduction," IEEE Trans. Aerosp. Electron. Syst., vol. 43, no. 3, pp.
    989-999, Jul. 2007.
"""
    w, m, P = _merge_runnals(p.w, p.m, p.P, K)
    return GaussianMixture(w, m, P)


def _merge_runnals(w, mu, P, K):
    """ KL-Divergence based mixture reduction

    Paramters
    ---------
    w: ndarray
        (nC,) component weights
    mu: ndarray
        (nC,nX) component means
    P: ndarray
        (nC,nX,nX) component covariances
    K : int
        desired number of components in final mixture

    Returns
    -------
    w: ndarray
        (nC,) component weights
    mu: ndarray
        (nC,nX) component means
    P: ndarray
        (nC,nX,nX) component covariances

    Notes
    -----
    This algorithm is based on the work by Runnals [1].

    This function was ported to python from David F. Crouse's MATLAB tracking
    library.

    References
    ----------
    [1] A. R. Runnalls, "Kullback-Leibler approach to Gaussian mixture
    reduction," IEEE Trans. Aerosp. Electron. Syst., vol. 43, no. 3, pp.
    989-999, Jul. 2007.
"""
    N = len(w)

    # If no reduction is necessary.
    if N <= K:
        return w, mu, P

    # We will only be using one triangular of this matrix.
    M = np.inf*np.ones((N, N))  # This is the cost matrix.

    # We shall fill the cost matrix with the cost of all pairs.
    for cur1 in range(N-1):
        for cur2 in range(cur1+1, N):
            M[cur1, cur2] = runnals_b_dist(w[cur1], w[cur2], mu[cur1],
                                           mu[cur2], P[cur1], P[cur2])

    Nr = N
    for mergeRound in range(N-K):
        # find the minimum cost pair
        minRow, minCol = np.unravel_index(np.argmin(M), M.shape)
        # Now we know which two hypotheses to merge.
        curClust = np.array([minRow, minCol])
        wmerged, mumerged, Pmerged = merge_components(
            w[curClust], mu[curClust], P[curClust])
        w = np.hstack((np.delete(w, curClust, axis=0), wmerged))
        mu = np.vstack((np.delete(mu, curClust, axis=0),
                       np.expand_dims(mumerged, axis=0)))
        P = np.vstack((np.delete(P, curClust, axis=0),
                      np.expand_dims(Pmerged, axis=0)))

        # Now we must remove the two hypotheses from the cost matrix and add
        # the merged hypothesis to the end of the matrix.
        M = np.delete(M, curClust, axis=0)
        M = np.delete(M, curClust, axis=1)

        Nr -= 1

        # Now we must add the distances for the new stuff to the end of the
        # matrix.
        M = np.vstack((
            np.hstack((M, np.full((Nr-1, 1), np.inf))),
            np.full(Nr, np.inf)
        ))

        for curRow in range(Nr-1):
            M[curRow, Nr-1] = runnals_b_dist(w[curRow], w[Nr-1],
                                             mu[curRow], mu[Nr-1], P[curRow], P[Nr-1])

    return w, mu, P


@jit(nopython=True)
def runnals_b_dist(w1, w2, m1, m2, P1, P2):
    """ dissimilarity measure between two GM components

    Parameters
    ----------
    w1 : float
        weight of first component
    w2 : float
        weight of second component
    m1 : ndarray
       (nx,) mean of first component
    m2 : ndarray
       (nx,) mean of second component
    P1 : ndarray
       (nx,nx) covariance of first component
    P2 : ndarray
       (nx,nx) covariance of second component

    Returns
    -------
    float
      dissimilarity between components

    Notes
    -----
    As described in [1], this dissimilarity measure is an upper bound of the KL
    divergence between the original mixture and merged mixture.

    This function was ported to python from David F. Crouse's MATLAB tracking
    library.

    References
    ----------
    [1] A. R. Runnalls, "Kullback-Leibler approach to Gaussian mixture
    reduction," IEEE Trans. Aerosp. Electron. Syst., vol. 43, no. 3, pp.
    989-999, Jul. 2007.
    """

    diff = m1-m2

    w1m = w1/(w1+w2)
    w2m = w2/(w1+w2)
    P12 = w1m*P1 + w2m*P2 + w1m*w2m*(np.outer(diff, diff))

    val = 0.5*((w1+w2)*np.log(np.linalg.det(P12))-w1 *
               np.log(np.linalg.det(P1))-w2*np.log(np.linalg.det(P2)))

    # deal with the case where w1 and w2 are both essentially zero
    if not np.isfinite(val):
        val = 0
    return val
