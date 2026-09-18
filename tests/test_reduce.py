import numpy as np
import numpy.testing as npt
import pyest.gm as gm
from pyest.gm import GaussianMixture, reduce
import pytest
from itertools import permutations


# Two mixtures for testing: one univariate and one multivariate
prune_truncate_cases = pytest.mark.parametrize("p", [
    GaussianMixture(w=np.array([0.41, 0.39, 0.2, 0]),
                    m=np.array([[1.], [2.], [3.], [4.]]),
                    cov=np.array([[[1.]], [[1.]], [[1.]], [[1.]]])),
    GaussianMixture(w=np.array([0.999999, 0.000001]),
                    m=np.array([[0, 1], [1, 0]]),
                    cov=np.array([np.eye(2), np.eye(2)]))
    ],
    ids=["p1", "p2"])


@prune_truncate_cases
def test_prune(p):
    # Ensuring zero weight components get pruned
    rtol = 0
    p_prune = reduce.prune(p, rtol, warn_tol=1)
    x = p.mean()
    npt.assert_equal(p_prune.size, np.sum(p.w != 0))
    npt.assert_allclose(p(x), p_prune(x), rtol=1e-15, atol=1e-15)
    npt.assert_allclose(np.sum(p.w), np.sum(p_prune.w),
                        rtol=1e-15, atol=1e-15)

    # Ensuring low weight components get pruned
    rtol = 1e-5
    p_prune = reduce.prune(p, rtol, warn_tol=1)
    max_w = np.max(p.w)
    npt.assert_equal(p_prune.size, np.sum(p.w >= rtol*max_w))
    npt.assert_allclose(p(x), p_prune(x), rtol=1e-3, atol=1e-3)
    npt.assert_allclose(np.sum(p.w), np.sum(p_prune.w),
                        rtol=1e-15, atol=1e-15)


@prune_truncate_cases
def test_prune_warning(p):
    # Ensuring warning occurs due to warn_tol
    with pytest.warns(UserWarning):
        _ = reduce.prune(p, 1e-5, warn_tol=0)


@prune_truncate_cases
def test_truncate(p):
    max_w_idx = np.argmax(p.w)
    for K in [1, 2, 3, 4, 10]:
        p_truncate = reduce.truncate(p, K, warn_tol=1)
        npt.assert_equal(p_truncate.size, np.min((K, p.size)))

        max_w_idx_truncate = np.argmax(p_truncate.w)
        npt.assert_array_equal(p_truncate.m[max_w_idx_truncate],
                               p.m[max_w_idx])


@prune_truncate_cases
def test_truncate_warning(p):
    # Ensuring warning occurs due to warn_tol
    with pytest.warns(UserWarning):
        _ = reduce.truncate(p, 1, warn_tol=0)


def test_merge_components():
    # Test mixture
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    m = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -2]])
    Si = np.array([[1, 0], [1, 2]])
    Pi = Si @ Si.T
    S = np.repeat(Si[np.newaxis, :, :], len(w), axis=0)
    P = np.repeat(Pi[np.newaxis, :, :], len(w), axis=0)

    # Merging with both full and cholesky implementations
    w_full, m_full, P_full = reduce.merge_components(w, m, P,
                                                     cov_type='full')
    w_chol, m_chol, S_chol = reduce.merge_components(w, m, S,
                                                     cov_type='cholesky')

    # Making sure output cholesky factor is a cholesky factor
    npt.assert_allclose(S_chol, np.tril(S_chol))
    assert (np.all(np.diag(S_chol) > 0))
    P_chol = S_chol @ S_chol.T

    # Checking weight and mean are what they should be
    npt.assert_allclose(w_full, 0.5)
    npt.assert_allclose(m_full, np.array([0, -0.2]))

    # Checking both implementations agree
    npt.assert_allclose(w_full, w_chol)
    npt.assert_allclose(m_full, m_chol)
    npt.assert_allclose(P_full, P_chol)

    # Check for error if cov_type is inputted wrong
    with pytest.raises(ValueError):
        w_full, m_full, P_full = reduce.merge_components(w, m, P,
                                                         cov_type='fullesky')

    # Check single mixand case
    w = np.array([1])
    m = np.array([[0]])
    P = np.array([[[1]]])
    w_full, m_full, P_full = reduce.merge_components(w, m, P)
    npt.assert_allclose(w_full, w)
    npt.assert_allclose(m_full, m[0])
    npt.assert_allclose(P_full, P[0])

    # Slightly more stressing case, compared against manual calculations using
    # MATLAB
    w = np.array([0.2, 0.3, 0.001])
    m = np.array([[0, 1], [2, 0], [1, -1]])
    S = np.array([[[1, 0], [0, 1]],
                  [[10, 0], [-4, 0.001]],
                  [[100, 0], [0, 0.0001]]])
    P = np.array([Si @ Si.T for Si in S])
    w_full, m_full, P_full = reduce.merge_components(w, m, P,
                                                     cov_type='full')
    w_chol, m_chol, S_chol = reduce.merge_components(w, m, S,
                                                     cov_type='cholesky')

    # Making sure output cholesky factor is a cholesky factor
    npt.assert_allclose(S_chol, np.tril(S_chol))
    assert (np.all(np.diag(S_chol) > 0))
    P_chol = S_chol @ S_chol.T

    # Checking weight, mean, and cov are what they should be
    npt.assert_allclose(w_full, 0.501)
    npt.assert_allclose(m_full,
                        np.array([1.199600798403194, 0.397205588822355]))
    npt.assert_allclose(
        P_full,
        np.array([[81.197684471376604, -24.430579957848771],
                  [-24.430579957848771, 10.223465843980740]]))

    # Checking both implementations agree
    npt.assert_allclose(w_full, w_chol)
    npt.assert_allclose(m_full, m_chol)
    npt.assert_allclose(P_full, P_chol)


def test_merge():
    p = gm.defaults.default_gm()

    # test with super small md threshold
    p_merged_full = gm.merge(p, md=1e-9, cov_type='full')
    p_merged_chol = gm.merge(p, md=1e-9, cov_type='cholesky')

    assert (p == p_merged_full)
    # Even though no merging has been performed, the values of P won't be
    # identical due to floating point errors in the forming of the whole
    # covariance
    npt.assert_array_equal(p.w, p_merged_chol.w)
    npt.assert_array_equal(p.m, p_merged_chol.m)
    npt.assert_array_equal(p.Schol, p_merged_chol.Schol)
    npt.assert_allclose(p.P, p_merged_chol.P, rtol=1e-14)

    # test with really large md threshold
    p_merged_full = gm.merge(p, md=1e9, cov_type='full')
    p_merged_chol = gm.merge(p, md=1e9, cov_type='cholesky')

    assert (len(p_merged_full) == 1)
    npt.assert_allclose(p_merged_full.m[0], p.mean(), rtol=1e-14)
    npt.assert_allclose(p_merged_full.P[0], p.cov(), rtol=1e-14)

    assert (len(p_merged_chol) == 1)
    npt.assert_allclose(p_merged_chol.m[0], p.mean(), rtol=1e-14)
    npt.assert_allclose(p_merged_chol.P[0], p.cov(), rtol=1e-14)

    # Check for error if cov_type is inputted wrong
    with pytest.raises(ValueError):
        p = reduce.merge(p, 1, cov_type='fullesky')


def test_merge_identical():
    # Two mixtures for testing: one univariate and one multivariate
    p1 = GaussianMixture(w=np.array([0.4, 0.4, 0.2, 0]),
                         m=np.array([[1.], [2.], [1.], [1.]]),
                         cov=np.array([[[1.]], [[1.]], [[1.]], [[1.]]]))

    p1_merged = reduce.merge_identical(p1)
    npt.assert_equal(p1_merged.size, 2)
    npt.assert_allclose(p1.mean(), p1_merged.mean())
    npt.assert_allclose(p1.cov(), p1_merged.cov())

    p2 = GaussianMixture(w=np.array([0.9, 0.01, 0.09]),
                         m=np.array([[0, 1], [1, 0], [1, 0]]),
                         cov=np.array([np.eye(2),
                                       0.5*np.eye(2),
                                       0.5*np.eye(2)]))
    p2_merged = reduce.merge_identical(p2)
    npt.assert_equal(p2_merged.size, 2)
    npt.assert_allclose(p2.mean(), p2_merged.mean())
    npt.assert_allclose(p2.cov(), p2_merged.cov())


def test_merge_runnalls():
    # a mixture with two identical components, when merged, should be the same
    # component with double the weight
    w1 = 0.5
    w2 = 0.5
    m1 = np.array([1, 2])
    m2 = np.array([1, 2])
    P1 = 5*np.eye(2)
    P2 = 5*np.eye(2)

    p = GaussianMixture([w1, w2], [m1, m2], [P1, P2])
    K = 1  # reduce to single component
    p_red = reduce.merge_runnalls(p, K)
    assert (len(p_red) == 1)
    assert (p_red.w[0] == 1)
    npt.assert_array_equal(m1, p_red.m[0])
    npt.assert_allclose(P1, p_red.P[0], rtol=1e-14)

    # now use the default mixture (w/ different components) and ensure that the
    # conditional mean and covariance are preserved
    p = gm.defaults.default_gm()

    p_red = reduce.merge_runnalls(p, K)

    npt.assert_allclose(p.mean(), p_red.mean(), rtol=1e-14)
    npt.assert_allclose(p.cov(), p_red.cov(), rtol=1e-14)

    # Check for error if cov_type is inputted wrong
    with pytest.raises(ValueError):
        p_red = reduce.merge_runnalls(p, K,
                                      cov_type='fullesky')

    # Check that no merging occurs if K > p.size
    p_red = reduce.merge_runnalls(p, p.size+1)
    npt.assert_array_equal(p.w, p_red.w)
    npt.assert_array_equal(p.m, p_red.m)
    npt.assert_array_equal(p.Schol, p_red.Schol)
    npt.assert_allclose(p.P, p_red.P, rtol=1e-14)

    # Check that K_max is adhered to if b_max is set very low
    p_red = reduce.merge_runnalls(p, K=1, b_max=-np.inf, K_max=2)
    assert (p_red.size == 2)

    # Check that K is adhered to if greater than 1
    p_red = reduce.merge_runnalls(p, K=2, b_max=np.inf, K_max=2)
    assert (p_red.size == 2)

    # More thorough test case
    # This mixture is set up such that, when reducing to 3 mixands, mixands 1,
    # 2, and 3 should all be merged together, while mixands 4 and 5 should be
    # left unchanged. This is because the means of mixands 1 and 2 are very
    # similar, while the weight of mixand 3 is very small.
    w_orig = np.array([0.2, 0.2, 0.0001, 0.2, 0.1999])
    m_orig = np.array([[0, 0],
                       [-0.1, 0.1],
                       [1, 0],
                       [0, 1],
                       [-1, 0]])
    S_orig = np.array([0.9*np.eye(2), 1.1*np.eye(2), 1.05*np.eye(2),
                       0.95*np.eye(2), 0.9*np.eye(2)])

    # p_true is what we should get after merging
    w_true_1, m_true_1, S_true_1 = reduce.merge_components(
        w_orig[0:3], m_orig[0:3], S_orig[0:3], cov_type='cholesky')
    w_true = np.array([w_true_1, w_orig[3], w_orig[4]])
    m_true = np.array([m_true_1, m_orig[3], m_orig[4]])
    S_true = np.array([S_true_1, S_orig[3], S_orig[4]])
    P_true = np.array([Si @ Si.T for Si in S_true])

    # Looping over permutations of the mixands in the parameter arrays to
    # ensure that doesn't affect things
    for perm in permutations(range(5)):
        perm = np.array(perm)
        w_perm = w_orig[perm]
        m_perm = m_orig[perm]
        S_perm = S_orig[perm]
        p_perm = GaussianMixture(w_perm, m_perm, S_perm, cov_type='cholesky')

        for cov_type in ['full', 'cholesky']:
            # This b_max should allow the merge to 3
            p_merge = reduce.merge_runnalls(
                p_perm, K=3, b_max=0.01, K_max=4, cov_type=cov_type)
            w_merge = p_merge.w
            m_merge = p_merge.m
            S_merge = p_merge.Schol
            P_merge = p_merge.P

            sort_idx = np.argsort(w_merge)[::-1]
            w_merge = w_merge[sort_idx]
            m_merge = m_merge[sort_idx]
            S_merge = S_merge[sort_idx]
            P_merge = P_merge[sort_idx]

            npt.assert_array_equal(w_merge, w_true)
            npt.assert_allclose(m_merge, m_true, rtol=1e-14, atol=1e-14)
            npt.assert_allclose(S_merge, S_true, rtol=1e-14, atol=1e-14)
            npt.assert_allclose(P_merge, P_true, rtol=1e-14, atol=1e-14)


def test_gaussian_mixture_reduction_options():
    # Just checking input validation

    # Non-integer inputs for these parameters should raise errors
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(K_min=1.0)
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(K_max=1.0)
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(K_max_runnalls=1.0)
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(K_max_merge=1.0)
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(max_iter_cluster=1.0)

    # K_min should be less than or equal to K_max
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(K_max=1, K_min=2)

    # prune_threshold should be between 0 and 1
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(prune_threshold=-1)
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(prune_threshold=5)

    # merge_md_threshold should be positive
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(merge_md_threshold=-1)

    # max_iter_cluster should be positive
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(max_iter_cluster=0)

    # cov_type should be one of the valid options
    with pytest.raises(ValueError):
        _ = reduce.GaussianMixtureReductionOptions(cov_type='fullesky')

    # If none of these are broken, then the initialization should work
    _ = reduce.GaussianMixtureReductionOptions()
    _ = reduce.GaussianMixtureReductionOptions(K_min=1,
                                               K_max=10,
                                               K_max_runnalls=15,
                                               K_max_merge=20,
                                               prune_threshold=0,
                                               merge_md_threshold=1,
                                               runnalls_b_max=0.05,
                                               max_iter_cluster=25,
                                               cov_type='full')


if __name__ == '__main__':
    pytest.main([__file__])
