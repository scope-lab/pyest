import numpy.testing as npt
from pyest import linalg
import numpy as np
import pytest


def test_is_pos_def():
    A = np.array([[1, 0], [0, 1]])
    assert linalg.is_pos_def(A)

    A = np.array([[1, 0], [0, -1]])
    assert not linalg.is_pos_def(A)


def test_is_square_matrix():
    A = np.array([[1, 0], [0, 1]])
    assert linalg.is_square_matrix(A)

    A = np.array([[1, 0], [0, 1], [0, 0]])
    assert not linalg.is_square_matrix(A)

    A = 1
    assert linalg.is_square_matrix(A)


def test_choldowndate():
    S = (np.arange(9).reshape(3, 3)**2).T
    S = np.linalg.cholesky(S@S.T)
    x = np.arange(1, 4)
    Si = linalg.choldowndate(S.T, x).T

    npt.assert_allclose(Si@Si.T, S@S.T - np.outer(x, x))


def test_make_chol_diag_positive():
    S = (np.arange(9).reshape(3, 3)**2).T
    S = np.linalg.cholesky(S@S.T)
    Sc = S.copy()
    Sc[:, 2] *= -1
    assert(np.all(S@S.T == Sc@Sc.T))
    S_pos = linalg.make_chol_diag_positive(Sc)
    assert(np.all(S_pos.diagonal() > 0))
    assert(np.all(S_pos@S_pos.T == S@S.T))


def test_triangularize():
    # generate an arbitrary symmetric positive definite matrix
    R = (np.arange(9).reshape(3, 3)**2).T
    P = R@R.T
    # generate a valid square root factor (non-Cholesky)
    eigvals,eigvecs = np.linalg.eig(P)
    U = eigvecs@np.diag(np.sqrt(eigvals))
    S = linalg.triangularize(U)
    assert(np.all(S == np.tril(S)))
    npt.assert_allclose(S@S.T, P)
    Su = linalg.triangularize(U.T, upper=True)
    assert(np.all(Su == np.triu(Su)))
    npt.assert_allclose(Su.T@Su, P)


def test_cholesky_from_sqrt_precision():
    # generate an arbitrary symmetric positive definite matrix
    R = (np.arange(9).reshape(3, 3)**2).T
    P = R@R.T
    Pinv = np.linalg.inv(P)
    U = np.linalg.cholesky(Pinv)
    S = linalg.cholesky_from_sqrt_precision(U)
    npt.assert_allclose(S@S.T, P)


if __name__ == '__main__':
    pytest.main([__file__])
