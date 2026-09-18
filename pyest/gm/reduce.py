import numpy as np
import warnings
from numba import jit
from scipy.spatial import KDTree
from numpy.linalg import qr
from scipy.linalg import solve_triangular

from .gm import GaussianMixture
from pyest.linalg import make_chol_diag_positive


def prune(p, prune_threshold=0, warn_tol=0.01):
    """ Prune Gaussian mixture mixands with negligible weight

    Parameters
    ----------
    p : GaussianMixture
        gm to be pruned
    prune_threshold : float
        minimum weight relative to the maximum weight in the distribution,
        below which mixands will be pruned. Default is 0, so only 0 weight
        mixands will be pruned
    warn_tol : float
        proportion of total distribution mass that will throw a warning if
        discarded. Default is 0.01. Setting to 1 disables this warning.

    Returns
    -------
    p_pruned : GaussianMixture
        pruned gm

    """
    w = p.w
    tol = prune_threshold*np.max(w)

    idx_keep = w > tol

    if np.sum(w[~idx_keep]) >= warn_tol*np.sum(w):
        warnings.warn(f"prune removed {np.sum(w[~idx_keep]) / np.sum(w)}"
                      " proportion of total probability mass")

    wsum = np.sum(w)
    w = w[idx_keep]
    w = w*wsum/np.sum(w)
    m = p.m[idx_keep, :]
    Schol = p.Schol[idx_keep, :, :]

    return GaussianMixture(w, m, Schol, cov_type='cholesky')


def truncate(p, K=10_000, warn_tol=0.01):
    """ Truncate Gaussian mixture to a specified size

    Parameters
    ----------
    p : GaussianMixture
        gm to be truncated
    K : int
        number of components to truncate to
    warn_tol : float
        proportion of total distribution mass that will throw a warning if
        discarded. Default is 0.01. Setting to 1 disables this warning.

    Returns
    -------
    p_truncated : GaussianMixture
        truncated gm

    """
    K = int(K)
    if p.size <= K:
        return p

    w = p.w

    idx_keep = np.argpartition(w, -K)[-K:]
    idx_keep = idx_keep[np.argsort(w[idx_keep])[::-1]]

    wsum = np.sum(w)
    w = w[idx_keep]
    trunc_norm = np.sum(w)
    w = w * wsum / trunc_norm

    if wsum - trunc_norm >= warn_tol*wsum:
        warnings.warn(f"truncate removed {(wsum - trunc_norm)/wsum}"
                      " proportion of total probability mass")

    m = p.m[idx_keep]
    Schol = p.Schol[idx_keep]

    return GaussianMixture(w, m, Schol, cov_type='cholesky')


@jit(nopython=True)
def merge_components(w, m, cov, cov_type='full'):
    """ merge Gaussian mixture components into single component

    Parameters
    ----------
    w: ndarray
      (nC,) component weights
    m: ndarray
      (nC,nX) component means
    cov: ndarray
      (nC,nX,nX) component covariances or lower-triangular covariance square
      root factors, dictated by cov_type parameter
    cov_type: string
      form of covariance provided by cov, options of 'full' and 'cholesky',
      default is 'full'

    Returns
    -------
    float:
      merged weight
    ndarray:
      (nX,) merged mean
    ndarray:
      (nX,nX) merged covariance or lower-triangular covariance square root
      factor, dictated by cov_type parameter
    """
    if w.ndim != 1:
        raise ValueError('weight array w must be 1-dimensional, '
                         f'it is currently {w.ndim}-dimensional')
    if m.ndim != 2:
        raise ValueError('mean array m must be 2-dimensional, '
                         f'it is currently {m.ndim}-dimensional')
    if cov.ndim != 3:
        raise ValueError('covariance array cov must be 3-dimensional, '
                         f'it is currently {cov.ndim}-dimensional')
    if not (w.shape[0] == m.shape[0] == cov.shape[0]):
        raise ValueError('number of components in input arrays do not align')
    if not (m.shape[1] == cov.shape[1]):
        raise ValueError('state dimension of input arrays do not align')

    nc = len(w)
    if nc == 1:
        # Output must be converted to float64 for jitting, and this seems to
        # be the most efficient way to do it, where copies of the arrays are
        # only made if dtypes are actually changing
        return (
            np.float64(w[0]),
            np.asarray(m[0], np.float64),
            np.asarray(cov[0], np.float64)
        )

    nx = len(m[0])

    # merge weights
    w_merged = np.sum(w)
    # merge means
    m_merged = 1/w_merged*np.sum(
        np.expand_dims(w, axis=0).T*m,
        axis=0
    )

    if cov_type == 'full':
        # merge covariances
        delm = m_merged - m
        outer = delm[:, :, None] * delm[:, None, :]
        cov_merged = (
            np.sum(w[:, None, None] * (cov + outer), axis=0)
            / w_merged
        )
    elif cov_type == 'cholesky':
        # merge covariance square root factors
        # Goal is to turn the mixture covariance formula into SS^T, where S
        # is a short fat matrix concatenating all the mixture covariances and
        # mean deviations, then perform LQ
        sqrt_w = np.sqrt(w)
        S = sqrt_w[:, None, None] * cov
        d = sqrt_w[:, None] * (m_merged - m)
        blocks = np.concatenate((S, d[:, :, None]), axis=2)
        concat_matrix = np.ascontiguousarray(
            blocks.transpose(1, 0, 2)).reshape(nx, nc * (nx + 1))
        _, R = qr(concat_matrix.T)
        cov_merged = R.T / np.sqrt(w_merged)
        # Making the diagonal positive
        cov_merged = cov_merged @ np.diag(np.sign(np.diag(cov_merged)))
    else:
        raise ValueError('cov_type must be one of "full" or "cholesky"')

    return w_merged, m_merged, cov_merged


def merge(p, md, gate=None, cov_type='cholesky'):
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
    cov_type: string
      form of covariance provided by cov, options of 'full' and 'cholesky',
      default is 'cholesky'

    Returns
    -------
    p_merged : GaussianMixture
        merged gm
    """

    if gate is not None:
        tree = KDTree(p.m)
    else:
        tree = None

    idxs = np.arange(len(p))
    valid_mask = p.w > 0

    # allocate memory
    w_merged = np.full(len(p), np.nan)
    m_merged = np.full((len(p), p.dim), np.nan)
    cov_merged = np.full((len(p), p.dim, p.dim), np.nan)
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
            # Computing mahalanobis distance
            mahdist = np.sqrt(
                np.sum(solve_triangular(p.Schol[i], delm, lower=True)**2))
            merge_mask[i] = mahdist <= md

        if cov_type == 'full':
            cov_merge_mask = p.P[merge_mask]
        elif cov_type == 'cholesky':
            cov_merge_mask = p.Schol[merge_mask]
        else:
            raise ValueError('cov_type must be one of "full" or "cholesky"')

        w, m, cov = merge_components(p.w[merge_mask],
                                     p.m[merge_mask],
                                     cov_merge_mask,
                                     cov_type=cov_type)

        # store merged components
        w_merged[new_comp_count] = w
        m_merged[new_comp_count] = m
        cov_merged[new_comp_count] = cov

        # mark the components we just merged as used
        valid_mask[merge_mask] = False

        new_comp_count += 1

    return GaussianMixture(
        w_merged[:new_comp_count],
        m_merged[:new_comp_count],
        cov_merged[:new_comp_count],
        cov_type=cov_type
    )


def merge_identical(p):
    """ Combine identical mixands of a Gaussian Mixture

    Parameters
    ----------
    p : GaussianMixture
        GM that is to have identical mixands combined

    Returns
    -------
    p_merged : GaussianMixture
        GM with identical components combined
    """
    nc = p.size
    nx = p.dim

    w = p.w
    m = p.m
    S = p.Schol

    tril_idx = np.tril_indices(nx)
    ntril = int(nx*(nx+1)/2)

    mS = np.zeros((nc, nx + ntril))
    for i in range(nc):
        Si = S[i]
        mS[i] = np.concatenate((m[i], Si[tril_idx].flatten()))

    # inverse_idx says which element of mS_unique each elemnt of mS
    # corresponds to
    mS_unique, inverse_idx = np.unique(mS, axis=0, return_inverse=True)

    if len(mS_unique) == nc:
        w_merged = w
        m_merged = m
        S_merged = S
    else:
        nc_unique = len(mS_unique)

        # Preallocating zeros for w_merged
        w_merged = np.zeros((nc_unique,))

        # Making m_merged and S_merged their unique values
        m_merged = np.array([mSi[:nx] for mSi in mS_unique])
        S_merged = np.zeros((nc_unique, nx, nx))
        S_merged[:, tril_idx[0], tril_idx[1]] = mS_unique[:, nx:]

        # Finding which unique mixands each of the original mixands belonged
        # to, and adding their weight accordingly
        for i in range(nc):
            w_merged[inverse_idx[i]] += w[i]

    return GaussianMixture(w_merged, m_merged, S_merged, cov_type='cholesky')


def merge_runnalls(p, K, b_max=np.inf, K_max=np.inf, cov_type='cholesky'):
    """ KL-Divergence based mixture reduction

    Parameters
    ---------
    p : GaussianMixture
        GM that is to be reduced
    K : int
        desired number of components in final mixture
    b_max : float, optional
        If the number of components is less than or equal to K_max, then
        reduction will continue until the lowest cost between two components
        is greater than b_max or there are K components left. The default if
        this parameter is omitted is inf, which means that reduction will
        continue until K components are reached.
    K_max : int, optional
        Regardless of the b_max setting, it won't be allowed that more than
        K_max components are present. The default if is inf.
    cov_type: string
        form of covariance provided by cov, options of 'full' and 'cholesky',
        default is 'cholesky'

    Returns
    -------
    p_merged : GaussianMixture
        reduced GM

    Notes
    -----
    This algorithm is based on the work by Runnalls [1].

    This function was ported to python from David F. Crouse's MATLAB tracking
    library.

    References
    ----------
    [1] A. R. Runnalls, "Kullback-Leibler approach to Gaussian mixture
    reduction," IEEE Trans. Aerosp. Electron. Syst., vol. 43, no. 3, pp.
    989-999, Jul. 2007.
    """
    if cov_type == 'full':
        cov = p.P
    elif cov_type == 'cholesky':
        cov = p.Schol
    else:
        raise ValueError('cov_type must be one of "full" or "cholesky"')
    w, m, cov = _merge_runnalls(p.w, p.m, cov, K, b_max=b_max, K_max=K_max,
                                cov_type=cov_type)
    return GaussianMixture(w, m, cov, cov_type=cov_type)


@jit(nopython=True)
def _merge_runnalls(w, m, cov, K, b_max=np.inf, K_max=np.inf,
                    cov_type='cholesky'):
    """ KL-Divergence based mixture reduction

    Parameters
    ---------
    w: ndarray
        (nC,) component weights
    mu: ndarray
        (nC,nX) component means
    cov: ndarray
        (nC,nX,nX) component covariances or lower-triangular covariance square
        root factors, dictated by cov_type parameter
    K : int
        desired number of components in final mixture
    b_max : float, optional
        If the number of components is less than or equal to K_max, then
        reduction will continue until the lowest cost between two components
        is greater than b_max or there are K components left. The default if
        this parameter is omitted is inf, which means that reduction will
        continue until K components are reached.
    K_max : int, optional
        Regardless of the b_max setting, it won't be allowed that more than
        K_max components are present. The default if is inf.
    cov_type: string
        form of covariance provided by cov, options of 'full' and 'cholesky',
        default is 'cholesky'

    Returns
    -------
    w: ndarray
        (nC,) component weights
    mu: ndarray
        (nC,nX) component means
    cov: ndarray
        (nC,nX,nX) component covariances or lower-triangular covariance square
        root factors, dictated by cov_type parameter

    Notes
    -----
    This algorithm is based on the work by Runnalls [1].

    This function was ported to python from David F. Crouse's MATLAB tracking
    library.

    References
    ----------
    [1] A. R. Runnalls, "Kullback-Leibler approach to Gaussian mixture
    reduction," IEEE Trans. Aerosp. Electron. Syst., vol. 43, no. 3, pp.
    989-999, Jul. 2007.
    """
    N = len(w)

    # Copying arrays so nothing is modified in place
    w = w.copy()
    m = m.copy()
    cov = cov.copy()

    # If no reduction is necessary.
    if N <= K:
        return w, m, cov

    # We will only be using the upper triangular portion of this matrix (not
    # including the diagonal)
    M = np.inf*np.ones((N, N))  # This is the cost matrix.

    # precomputing the values of w*log(det(P)) for each component
    w_log_det_P = np.inf*np.ones((N,))
    if cov_type == 'full':
        for cur1 in range(N):
            w_log_det_P[cur1] = w[cur1]*np.linalg.slogdet(cov[cur1])[1]
    elif cov_type == 'cholesky':
        for cur1 in range(N):
            w_log_det_P[cur1] = w[cur1]*2*np.log(
                np.prod(np.diag(cov[cur1])))

    # We shall fill the cost matrix with the cost of all pairs.
    for cur1 in range(N-1):
        for cur2 in range(cur1+1, N):
            M[cur1, cur2] = runnalls_b_dist(w[cur1], w[cur2],
                                            m[cur1], m[cur2],
                                            cov[cur1], cov[cur2],
                                            w_log_det_P[cur1],
                                            w_log_det_P[cur2],
                                            cov_type=cov_type)

    Nr = N
    merged_idxs = np.ones((N,), dtype=np.bool_)
    for mergeRound in range(N-K):
        # find the minimum cost pair
        # This was the original implementation for finding the min value, but
        # jit can't deal with np.unravel_index
        # minRow, minCol = np.unravel_index(np.argmin(M), M.shape)

        # This is the jit-compatible version
        flat_idx = np.argmin(M)
        minRow = flat_idx // M.shape[1]
        minCol = flat_idx % M.shape[1]

        # If we are at less than K_max components and the minimum cost is
        # higher than our b_max bound, then we stop merging
        if M[minRow, minCol] >= b_max and Nr <= K_max:
            break

        # Now we know which two hypotheses to merge.
        curClust = np.array([minRow, minCol])
        w_merged, m_merged, cov_merged = merge_components(
            w[curClust], m[curClust], cov[curClust], cov_type=cov_type)

        # Assigning new merged component to minRow index
        w[minRow] = w_merged
        m[minRow] = m_merged
        cov[minRow] = cov_merged
        if cov_type == 'full':
            w_log_det_P[minRow] = w[minRow]*np.linalg.slogdet(cov[minRow])[1]
        elif cov_type == 'cholesky':
            w_log_det_P[minRow] = w[minRow]*2*np.log(
                np.prod(np.diag(cov[minRow])))

        # "Removing" component at minCol (just noting that it shouldn't be
        # used, not explicitly removing it to save on memory reallocation)
        merged_idxs[minCol] = False

        # Now we must make the costs for the minCol mixand inf so that it
        # won't be considered for merging later
        M[:, minCol] = np.inf
        M[minCol, :] = np.inf

        # We must now fill in the costs for the merged estimate, which is in
        # minRow.

        # This fills in costs for the mixands with indices less than minRow,
        # going down the minRow column of the upper triangular cost matrix
        for cur1 in range(minRow):
            if merged_idxs[cur1]:
                M[cur1, minRow] = runnalls_b_dist(w[cur1], w[minRow],
                                                  m[cur1], m[minRow],
                                                  cov[cur1], cov[minRow],
                                                  w_log_det_P[cur1],
                                                  w_log_det_P[minRow],
                                                  cov_type=cov_type)
        # Then, to fill in the indices greater than minRow, we bounce off the
        # diagonal and head down the minRow row
        # (minRow, minRow) is left unfilled, since that would be the cost of
        # merging the new mixand with itself
        for cur2 in range(minRow+1, N):
            if merged_idxs[cur2]:
                M[minRow, cur2] = runnalls_b_dist(w[minRow], w[cur2],
                                                  m[minRow], m[cur2],
                                                  cov[minRow], cov[cur2],
                                                  w_log_det_P[minRow],
                                                  w_log_det_P[cur2],
                                                  cov_type=cov_type)

        # Adjusting the current number of components to reflect that a pair
        # was merged
        Nr -= 1

    w = w[merged_idxs]
    m = m[merged_idxs]
    cov = cov[merged_idxs]

    return w, m, cov


@jit(nopython=True)
def runnalls_b_dist(w1, w2, m1, m2, cov1, cov2, w1_log_det_P1, w2_log_det_P2,
                    cov_type='cholesky'):
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
    cov1 : ndarray
        (nx,nx) covariance or lower-triangular covariance square root factor of
        first component
    cov2 : ndarray
        (nx,nx) covariance or lower-triangular covariance square root factor of
        second component
    w1_log_det_P1 : float
        precomputed w1*log(det(P1))
    w2_log_det_P2 : float
        precomputed w2*log(det(P2))
    cov_type : string
        form of covariance provided by cov, options of 'full' and 'cholesky',
        default is 'full'

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
    w_sum = w1+w2
    w1m = w1/w_sum
    w2m = w2/w_sum

    if cov_type == 'full':
        P1 = cov1
        P2 = cov2
        diff = m1-m2
        P12 = w1m*P1 + w2m*P2 + w1m*w2m*(np.outer(diff, diff))
        val = 0.5*(w_sum*np.linalg.slogdet(P12)[1]
                   - w1_log_det_P1 - w2_log_det_P2)
    elif cov_type == 'cholesky':
        S1 = cov1
        S2 = cov2
        sqrt_w1m = np.sqrt(w1m)
        sqrt_w2m = np.sqrt(w2m)
        m_merged = w1m*m1 + w2m*m2

        diff1 = m1 - m_merged
        diff2 = m2 - m_merged

        # A full qr is used rather than an economy qr, since only the first
        # argument of numpy's qr is able to be used with numba's jit
        _, S12T = qr(np.concatenate((sqrt_w1m*S1, sqrt_w1m*diff1[:, None],
                                     sqrt_w2m*S2, sqrt_w2m*diff2[:, None]),
                                    axis=1).T)
        S12 = S12T.T

        # Note that the use of np.abs() is only required here because we aren't
        # enforcing a positive diagonal after the qr
        w12_log_det_P12 = w_sum*2*np.log(np.prod(np.abs(np.diag(S12))))
        val = 0.5*(w12_log_det_P12 - w1_log_det_P1 - w2_log_det_P2)

    # deal with the case where w1 and w2 are both essentially zero
    if not np.isfinite(val):
        val = 0
    return val


# Deprecated functions with old spelling (single 'l')
def merge_runnals(p, K):
    """
    .. deprecated::
        Use :func:`merge_runnalls` instead. The spelling 'runnals' was incorrect.
    """
    warnings.warn(
        "merge_runnals is deprecated and will be removed in a future version. "
        "Use merge_runnalls instead (correct spelling with double 'l').",
        DeprecationWarning,
        stacklevel=2
    )
    return merge_runnalls(p, K)


def _merge_runnals(w, mu, P, K):
    """
    .. deprecated::
        Use :func:`_merge_runnalls` instead. The spelling 'runnals' was incorrect.
    """
    warnings.warn(
        "_merge_runnals is deprecated and will be removed in a future version. "
        "Use _merge_runnalls instead (correct spelling with double 'l').",
        DeprecationWarning,
        stacklevel=2
    )
    return _merge_runnalls(w, mu, P, K, cov_type='full')


def runnals_b_dist(w1, w2, m1, m2, P1, P2):
    """
    .. deprecated::
        Use :func:`runnalls_b_dist` instead. The spelling 'runnals' was incorrect.
    """
    warnings.warn(
        "runnals_b_dist is deprecated and will be removed in a future version. "
        "Use runnalls_b_dist instead (correct spelling with double 'l').",
        DeprecationWarning,
        stacklevel=2
    )
    return runnalls_b_dist(w1, w2, m1, m2, P1, P2,
                           w1*np.linalg.slogdet(P1)[1],
                           w2*np.linalg.slogdet(P2)[1],
                           cov_type='full')


class GaussianMixtureReductionOptions():
    """ Gaussian Mixture reduction options

    Parameters
    ----------
    K_min : int
        Desired number of mixands to merge mixture down to. Also, the minimum
        number of mixands the mixture will ever be reduced to, short of
        pruning extra zero weight components. Default is 100.
    K_max : int
        Maximum number of mixands allowed following reduction. Default is 100.
    K_max_runnalls : int
        Maximum number of mixands allowed prior to Runnalls' merging. Default
        is 1_000.
    K_max_merge : int
        Maximum number of mixands allowed prior to Mahalanobis distance
        merging. Default is 10_000.
    prune_threshold : float
        minimum weight relative to the maximum weight in the distribution,
        below which mixands will be pruned. Default is 0, so only 0 weight
        mixands will be pruned
    merge_md_threshold : float
        mahalanobis distance threshold. components that within this distance
        of one another are merged
    runnalls_b_max : float
        In runnalls merging, if the number of components is less than or equal
        to K_max, then reduction will continue until the lowest cost between
        two components is greater than runnalls_b_max or there are K_min
        components left. The default if this parameter is omitted is inf, which
        means that reduction will continue until K components is reached.
    max_iter_cluster : int
        Maximum iterations used for cluster assignments in reduction via
        clustering. Default is 100.
    cov_type : string
        Form of covariance used throughout merging. Options of 'full' and
        'cholesky'. Default is 'cholesky'.
    """
    def __init__(self, K_min=100, K_max=100, K_max_runnalls=1_000,
                 K_max_merge=10_000, prune_threshold=0,
                 merge_md_threshold=0.1, runnalls_b_max=np.inf,
                 max_iter_cluster=100, cov_type='cholesky'):
        self.K_min = K_min
        self.K_max = K_max
        self.K_max_runnalls = K_max_runnalls
        self.K_max_merge = K_max_merge
        self.prune_threshold = prune_threshold
        self.merge_md_threshold = merge_md_threshold
        self.runnalls_b_max = runnalls_b_max
        self.max_iter_cluster = max_iter_cluster
        self.cov_type = cov_type

        self.validate_parameters()

    def validate_parameters(self):
        """ Method for ensuring each option takes on a reasonable value
        """
        # Checking that everything that should be an int is an int
        must_be_int = ["K_min", "K_max", "K_max_runnalls", "K_max_merge",
                       "max_iter_cluster"]
        for parameter_type in must_be_int:
            parameter = getattr(self, parameter_type)
            if not isinstance(parameter, int):
                raise ValueError(parameter_type + " must be an int")

        # Make sure K_min <= K_max
        if self.K_min > self.K_max:
            raise ValueError('K_max must be greater than or equal to K_min')

        # Make sure prune_threshold is positive and less than 1
        if self.prune_threshold < 0 or self.prune_threshold >= 1:
            raise ValueError('prune_threshold must be in the range [0,1)')

        # Make sure merge_md_threshold is positive
        if self.merge_md_threshold < 0:
            raise ValueError('merge_md_threshold must be >= 0')

        # Make sure the number of clustering iterations is positive
        if self.max_iter_cluster <= 0:
            raise ValueError('max_iter_cluster must be > 0')

        # Make sure cov_type is one of the valid options
        cov_types = ["full", "cholesky"]
        if self.cov_type not in cov_types:
            raise ValueError('cov_type must be one of "full" or "cholesky"')
