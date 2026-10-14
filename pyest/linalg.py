import numpy as np
from numba import jit


def is_pos_def(x):
    """ check that x is positive definite """
    return np.all(np.linalg.eigvals(x) > 0)


def is_square_matrix(A):
    """ returns true if A is scalar or 2D square matrix """
    if np.isscalar(A):
        return True
    else:
        A = np.array(A)
        return A.shape[0] == A.shape[1]


def is_valid_chol(x):
    """ check the cholesky factor is valid """
    return np.all(np.diagonal(x) != 0)

def choldowndate(A, x):
    """ downdate Cholesky factorization of R with x

    Parameters
    ----------
    A: ndarray
        (n,n) upper-triangular Cholesky factorization of covariance matrix
    x: ndarray
        (n,) vector to downdate R

    Returns
    -------
    ndarray
        (n,n) downdated upper-triangular Cholesky factorization of covariance matrix

    Raises
    ------
    ValueError
        if the downdated Cholesky factorization is not valid
    """
    R = A.copy()
    n = R.shape[0]
    c = np.zeros(n)
    s = np.zeros(n)

    s[0] = x[0]/R[0, 0]
    for j in range(1, n):
        v1 = R[:j, j]
        v2 = s[:j]
        dt = np.dot(v1, v2)
        s[j] = x[j] - dt
        s[j] /= R[j, j]

    nrm = np.linalg.norm(s)
    if nrm >= 1:
        raise ValueError('downdate failed: norm of s >= 1')

    # determine the transformations
    alpha = np.sqrt(1 - nrm**2)
    for i in np.arange(n-1, -1, -1):
        scale = alpha + np.abs(s[i])
        a = alpha/scale
        b = s[i]/scale
        nrm = np.sqrt(a**2 + b**2)
        c[i] = a/nrm
        s[i] = b/nrm
        alpha = scale*nrm

    # apply the transformations to R
    for jj in range(n):
        xx = 0.0
        for ii in range(jj, -1, -1):
            t = c[ii]*xx + s[ii]*R[ii, jj]
            R[ii,jj] = c[ii]*R[ii, jj] - s[ii]*xx
            xx = t
    return R

def make_chol_diag_positive(S):
    """ make the diagonal of the Cholesky factorization positive """
    if np.all(np.diagonal(S) > 0):
        return S
    diag_sign = np.sign(np.diagonal(S))
    return np.tile(diag_sign,(S.shape[0],1))*S

def triangularize(A, upper=False):
    """ triangularize a square root factor A

    Required
    --------
    A : numpy.ndarray
        square root factor of a symmetric positive definite matrix

    Returns
    -------
    numpy.ndarray
        lower triangular matrix
    """
    if upper:
        S = np.linalg.qr(A, mode='r')
    else:
        S = np.linalg.qr(A.T, mode='r').T
        S = make_chol_diag_positive(S)
    return S


def cholesky_from_sqrt_precision(U):
    """
    Compute the Cholesky factor of the covariance matrix from the square root of the precision matrix.

    Required
    --------
    U : numpy.ndarray
        square root of the precision matrix: UU.T = inv(P)

    Returns
    -------
    numpy.ndarray
        Cholesky factor of the covariance matrix: LL.T = P

    """
    P_sqrt = np.linalg.inv(U).T
    return triangularize(P_sqrt)


# Manual triangular solvers so they can be jitted (unlike scipy's)
@jit(nopython=True)
def _solve_lower_triangular(L, B):
    """ Solves LX = B where L is lower triangular.

    Parameters
    ----------
    L : ndarray
        (n,n) lower triangular matrix
    B: ndarray
        (n,) vector or (n,m) matrix

    Returns
    -------
    X : ndarray
        (n,) vector or (n,m) matrix
    """
    n = L.shape[0]

    if B.ndim == 1:
        X = np.empty(n)
        for i in range(n):
            tmp = B[i]
            for j in range(i):
                tmp -= L[i, j] * X[j]
            X[i] = tmp / L[i, i]
        return X

    else:  # B.ndim == 2
        m = B.shape[1]
        X = np.empty((n, m))
        for i in range(n):
            tmp = B[i, :].copy()
            for j in range(i):
                tmp -= L[i, j] * X[j, :]
            X[i, :] = tmp / L[i, i]
        return X


@jit(nopython=True)
def _solve_upper_triangular(U, B):
    """ Solves UX = B where U is upper triangular.

    Parameters
    ----------
    U : ndarray
        (n,n) upper triangular matrix
    B: ndarray
        (n,) vector or (n,m) matrix

    Returns
    -------
    X : ndarray
        (n,) vector or (n,m) matrix
    """
    n = U.shape[0]

    if B.ndim == 1:
        X = np.empty(n)
        for i in range(n - 1, -1, -1):
            tmp = B[i]
            for j in range(i + 1, n):
                tmp -= U[i, j] * X[j]
            X[i] = tmp / U[i, i]
        return X

    else:  # B.ndim == 2
        m = B.shape[1]
        X = np.empty((n, m))
        for i in range(n - 1, -1, -1):
            tmp = B[i, :].copy()
            for j in range(i + 1, n):
                tmp -= U[i, j] * X[j, :]
            X[i, :] = tmp / U[i, i]
        return X


@jit(nopython=True)
def solve_triangular(A, B, lower=True, atol=1e-15):
    """ Solves AX = B where A is triangular.

    Parameters
    ----------
    A : ndarray
        (n,n) triangular matrix
    B : ndarray
        (n,) vector or (n,m) matrix
    lower : bool
        If True, A is treated as lower triangular. If False, A is treated as
        upper triangular.
    atol : float
        Absolute tolerance below which diagonal elements will be considered 0,
        and the matrix will be considered singular

    Returns
    -------
    X : ndarray
        (n,) vector or (n,m) matrix

    Raises
    ------
    ValueError
        if A is singular
    """
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)

    if (np.abs(np.diag(A)) < atol).any():
        raise ValueError("triangular solve failed: A is singular")

    if lower:
        return _solve_lower_triangular(A, B)
    else:
        return _solve_upper_triangular(A, B)
