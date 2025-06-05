from copy import deepcopy

import dill as pickle
import numpy as np
import numpy.testing as npt
import pyest.gm as gm
from pyest.sensors.defaults import default_poly_fov
from pyest.sensors import ConvexPolyhedralFieldOfView
from pyest.utils import fail
import pytest


def test_gm_equal_weights():
    n = 3
    npt.assert_almost_equal(np.array(n*[1/n]),
                            gm.equal_weights(n))


def test_distribute_mean_centers():
    npt.assert_almost_equal(np.array([0.2, 0.4, 0.6, 0.8]),
                            gm.distribute_mean_centers(0, 1, 4))


def test_optimal_homoscedastic_std():
    npt.assert_almost_equal(0.34673417302127424,
                            gm.optimal_homoscedastic_std(1))

    npt.assert_almost_equal(0.40010057651803865,
                            gm.optimal_homoscedastic_std(4, 2))


def test_v_eval_mvnpdf():

    nc = 5
    x = np.array([[-0.5331413,  0.49738376],
                  [-0.9223893, -0.53167751],
                  [1.29972541, -0.80055403],
                  [1.50169209,  2.23780065],
                  [0.1007185, -0.74586585],
                  [-0.77959106,  1.14114614],
                  [0.86560002, -1.65330289],
                  [0.02338582,  1.10248293],
                  [-0.86812804,  0.1667633],
                  [-1.44508474, -0.70940488],
                  [-0.25187759,  2.31631117],
                  [0.49324496,  1.10234038],
                  [-0.29318582,  0.68277976],
                  [-1.8647944,  0.69674822],
                  [-0.58019669,  1.50201133]])
    m = np.array([[-1.25119851, -1.19080598],
                  [-0.78650525,  1.43105468],
                  [1.86207408, -0.67981876],
                  [1.63445144,  0.84827199],
                  [-1.24341263,  0.51828046]])
    p = np.array([[1., 0.],
                  [0., 1.]])
    P = np.tile(p[:, :, np.newaxis], (1, 1, nc))
    vals = gm.v_eval_mvnpdf(x, m, P.T)

    npt.assert_almost_equal(
       vals,
       np.array([
          [2.95795376e-02, 1.21339850e-01, 5.69804707e-03, 1.00820686e-05,
           5.78036860e-02, 9.39024209e-03, 1.52184499e-02, 5.09372092e-03,
           5.88510419e-02, 1.39102180e-01, 2.06103195e-04, 2.50697433e-03,
           1.73887299e-02, 2.22023429e-02, 3.38422175e-03],
          [9.96745991e-02, 2.29768491e-02, 1.49729525e-03, 8.38576110e-03,
           1.00422753e-02, 1.52601634e-01, 3.49420745e-04, 1.08628819e-01,
           7.13309793e-02, 1.29647114e-02, 9.32351993e-02, 6.64846683e-02,
           1.06509960e-01, 6.79601480e-02, 1.55411926e-01],
          [4.51970606e-03, 3.26182589e-03, 1.34891627e-01, 2.11422673e-03,
           3.36667384e-02, 9.25634593e-04, 6.03138799e-02, 5.99659197e-03,
           2.67654235e-03, 6.70850345e-04, 1.91490918e-04, 1.27430500e-02,
           6.16549381e-03, 5.94651727e-05, 7.46278582e-04],
          [1.42832539e-02, 2.33737993e-03, 3.86498863e-02, 6.00796239e-02,
           1.37779862e-02, 8.27459500e-03, 5.18294552e-03, 4.20895881e-02,
           5.50798650e-03, 4.12684476e-04, 9.14484273e-03, 8.03527823e-02,
           2.44906563e-02, 3.45085780e-04, 1.10657801e-02],
          [1.23645326e-01, 8.71076199e-02, 2.62857134e-03, 8.38369299e-04,
           2.90059512e-02, 1.17722735e-01, 1.62909384e-03, 6.01500811e-02,
           1.39445912e-01, 7.34009403e-02, 1.93337085e-02, 2.97052984e-02,
           9.99712563e-02, 1.29139754e-01, 7.87349629e-02]]))


def test_bounds():
    '''test gm bounds'''
    m = np.array([[0.5416282,  0.48059004, -0.58374849],
                  [0.80886369,  1.30767534, -0.88941716],
                  [-0.6696043,  0.15051094,  0.78734996],
                  [-2.20821296,  0.44005201,  0.70486189],
                  [1.13080834,  1.11579637, -0.77672268],
                  [-0.76237876, -0.10381707,  1.6920369],
                  [2.34478359,  0.56632462,  0.24846986],
                  [-0.71164582,  1.00977891, -0.40863085],
                  [-0.76718087, -0.87919271, -0.07699782],
                  [0.45705764,  1.07884756, -0.74475335]])

    Si = np.arange(9).reshape((3, 3))
    Pi = Si @ Si.T
    P = np.tile(Pi, (10, 1, 1))
    xmin, xmax = gm.bounds(m, P)

    xmin_des = np.array(
        [-8.91641689249937, -22.092396145596428, -37.5090840072011])
    xmax_des = np.array(
        [9.05298752249937, 22.520878775596426, 38.3117037472011])

    npt.assert_almost_equal(xmin, xmin_des, decimal=15)
    npt.assert_almost_equal(xmax, xmax_des, decimal=15)


def test_marginal_2d():
    """ test marginal 2d """
    w = np.array([0.4, 0.6])
    m = np.array(
        [[-0.3, 0.4, 1.2],
         [1.2, 0.2, 4.5]]
    )
    P = np.tile(np.diag([0.3**2, 0.2**2, 0.8**1]), (2, 1, 1))

    gmm = gm.GaussianMixture(w, m, P)
    gmm2d = gmm.marginal_2d([0, 2])

    m_des = np.array(
        [[-0.3,  1.2],
         [1.2,  4.5]]
    )
    P_des = np.array(
        [[[0.09, 0.],
          [0., 0.8]],
         [[0.09, 0.],
          [0., 0.8]]]
    )
    npt.assert_equal(gmm2d.w, w)
    npt.assert_equal(gmm2d.m, m_des)
    npt.assert_equal(gmm2d.P, P_des)


def test_gm_pdf_2d():
    """ test plotting of 2D GM """
    w = np.array([0.4, 0.6])
    m = np.array(
        [[-0.3, 0.4, 1.2],
         [1.2, 0.2, 4.5]]
    )
    P = np.tile(np.diag([0.3**2, 0.2**2, 0.8**1]), (2, 1, 1))

    gmm = gm.GaussianMixture(w, m, P)

    p, X, Y = gmm.pdf_2d(dimensions=[0, 2])
    # too much data to check, so just test min/max values
    npt.assert_almost_equal(np.max(p), 0.355674547107713, decimal=15)
    npt.assert_almost_equal(np.min(p), 5.054584032207333e-13, decimal=15)

def test_eval_mvnpdfchol():
    """ test evaluation of Gaussian PDF with Cholesky factor """
    m = np.array([1, 2])
    P = np.array([[1, 0.25], [0.25, 4]])
    S = np.linalg.cholesky(P)
    cov = gm.Covariance.from_cholesky(S)
    x = gm.multivariate_normal.rvs(m, cov, 10)
    pdes = gm.multivariate_normal.pdf(x, m, cov)
    pvals = gm.eval_mvnpdfchol(x, m, S)
    npt.assert_almost_equal(pvals, pdes, decimal=15)

def test_eval_gmpdfchol():
    w = np.array([0.4, 0.6])
    # choose means to be very far apart in the sense of Mahalanobis distance
    m = np.array([[-1e5, -2e5], [3e5, 4e5]])
    P = np.array([[[1, 0.25], [0.25, 4]], [[2, 0.5], [0.5, 3]]])
    S = np.array([np.linalg.cholesky(Pi) for Pi in P])
    x1 = m[1] + S[1] @ np.random.randn(2)
    x2 = m[0] + S[0] @ np.random.randn(2)
    x3 = m[0] + S[0] @ np.random.randn(2)
    evals = gm.eval_gmpdfchol(np.vstack((x1, x2, x3)), w, m, S)
    evals_des = np.array([
        w[1]*gm.multivariate_normal.pdf(x1.T, m[1], P[1]),
        w[0]*gm.multivariate_normal.pdf(x2.T, m[0], P[0]),
        w[0]*gm.multivariate_normal.pdf(x3.T, m[0], P[0])])
    npt.assert_almost_equal(evals, evals_des, decimal=15)

    # test failure case where the dimension of samples is wrong
    fail(gm.eval_gmpdf, ValueError, np.vstack((x1, x2, x3)).T, w, m, S)


def test_integral_gauss_product_chol():
    m1 = np.array([1, 2])
    m2 = np.array([3, 4])
    P1 = np.array([[3, 0.25], [0.25, 4]])
    P2 = np.array([[4, 0.35], [0.35, 5]])
    S1 = np.linalg.cholesky(P1)
    S2 = np.linalg.cholesky(P2)
    integral = gm.integral_gauss_product_chol(m1, S1, m2, S2)
    integrand = lambda x2, x1 : gm.multivariate_normal.pdf([x1,x2], m1, P1)*gm.multivariate_normal.pdf([x1,x2], m2, P2)
    from scipy.integrate import dblquad
    integral_numeric, err = dblquad(integrand, -15, 15, -15, 15)
    npt.assert_almost_equal(integral, integral_numeric, decimal=14)




def test_gm_rvs():
    """ test gm random sampling """
    gmm = gm.defaults.default_gm()
    np.random.seed(46)
    Y = gmm.rvs(2)
    npt.assert_allclose(
        Y,
        np.array([
            [15.419169134640605,  4.084490154472613,
             0.136294534030536, -0.116798840587642],
            [14.962682295485461,  4.014234989060866,
             0.016649473626872, 0.419306834014983]]
        ),
        rtol=1e-7
    )


def test_optimize_gauss_split():
    L = 2
    lam = 0.001

    p1 = gm.optimize_gauss_split(L, lam)
    # check that the weights are equal (only true for L=2)
    npt.assert_allclose(p1.w, 2*[0.5], rtol=1e-8)

    # check with L=3 with published values. Should give equally good or better
    # results
    L = 3
    lam = 0.001
    p1 = gm.optimize_gauss_split(L, lam)
    # find mean spacing
    my_eps = p1.m[1] - p1.m[0]
    # find sigma
    my_sig = np.sqrt(p1.P[0][0][0])
    my_w_half = p1.w[:int(np.ceil(L/2))]
    my_l2 = gm.obj_l2_gauss_split(my_eps, my_sig, my_w_half, L, lam)
    their_w_half = [0.2252246249, 0.5495507502]
    their_eps = -1.0575154615
    their_sig = 0.6715662887
    their_l2 = gm.obj_l2_gauss_split(
        their_eps, their_sig, their_w_half, L, lam)

    assert(my_l2 <= their_l2)


def test_optimize_gauss_split_variance_preserving(benchmark):
    # clear cached results
    gm.gm_split_l2_cov_cache.clear()
    # call function twice to test cache
    L = 4
    lam = 0.001
    p_4_comp = gm.split_1d_standard_gaussian(L, lam, variance_preserving=True)
    p_4_comp_cache = gm.split_1d_standard_gaussian(L, lam, variance_preserving=True)
    assert(p_4_comp == p_4_comp_cache)

    # using the same regularization parameter, the cost function value should
    # monotically decrease as the number of split components increases.
    p = gm.GaussianMixture(w=1, m=0, cov=1)
    split_cost = np.inf
    for L in np.arange(2, 9):
        if L == 9:
            p_split = benchmark(gm.split_1d_standard_gaussian, L, lam, variance_preserving=True)
        else:
            p_split = gm.split_1d_standard_gaussian(L, lam, variance_preserving=True)
        p_split = gm.split_1d_standard_gaussian(L, lam, variance_preserving=True)
        new_split_cost = gm.l2_dist(p, p_split) + lam*p_split.P[0][0][0]
        assert(new_split_cost < split_cost)
        split_cost = new_split_cost
        # the variance should also equal the original variance
        npt.assert_almost_equal(p.cov(), p_split.cov(), decimal=10)

    # using the same number of splits, the approximation error should
    # monotically decrease as the regularization term decreases.
    L = 4
    split_err = np.inf
    for lam in (1e-2, 1e-3, 1e-4):
        p_split = gm.split_1d_standard_gaussian(L, lam, variance_preserving=True)
        new_split_err = gm.l2_dist(p, p_split)
        assert(new_split_err < split_err)
        split_err = new_split_err
        # the variance should also equal the original variance
        npt.assert_almost_equal(p.cov(), p_split.cov(), decimal=10)

def test_split_1d_standard_gaussian(benchmark):
    # clear cached results
    gm.gm_split_l2_cache.clear()

    # call function twice to test cache
    L = 4
    lam = 0.001
    p_4_comp = gm.split_1d_standard_gaussian(L, lam, variance_preserving=False)
    p_4_comp_cache = benchmark(gm.split_1d_standard_gaussian, L, lam, variance_preserving=False)
    assert(p_4_comp == p_4_comp_cache)

    # using the same regularization parameter, the cost function value should
    # monotically decrease as the number of split components increases.
    p = gm.GaussianMixture(w=1, m=0, cov=1)
    split_cost = np.inf
    for L in np.arange(2, 11):
        p_split = gm.split_1d_standard_gaussian(L, lam, variance_preserving=False)
        new_split_cost = gm.l2_dist(p, p_split) + lam*p_split.P[0][0][0]
        assert(new_split_cost < split_cost)
        split_cost = new_split_cost

    # using the same number of splits, the approximation error should
    # monotically decrease as the regularization term decreases.
    L = 4
    split_err = np.inf
    for lam in (1e-2, 1e-3, 1e-4):
        p_split = gm.split_1d_standard_gaussian(L, lam, variance_preserving=False)
        new_split_err = gm.l2_dist(p, p_split)
        assert(new_split_err < split_err)
        split_err = new_split_err


def test_split_gaussian():
    p = gm.defaults.default_gm(covariance_rotation=np.pi/6)

    split_err = np.inf

    # using the same number of splits, the approximation error should
    # monotically improve as the regularization term decreases.
    L = 4
    for lam in (1e-2, 1e-3, 1e-4):
        split_opts = gm.GaussSplitOptions(L=L, lam=lam)
        p_copy = deepcopy(p)
        # for simplicity, only split the first component
        split_comp = gm.split_gaussian(*p_copy.pop(0), split_opts)
        p_copy += split_comp
        new_split_err = gm.l2_dist(p, p_copy)
        assert(new_split_err < split_err)
        split_err = new_split_err

    # test the case where sqrt factors are provided
    Seig = p.Seig[0]
    split_err = np.inf
    for lam in (1e-2, 1e-3, 1e-4):
        split_opts = gm.GaussSplitOptions(L=L, lam=lam)
        split_comp = gm.split_gaussian(*p_copy.pop(0), split_opts)
        p_copy = deepcopy(p)
        # for simplicity, only split the first component
        Schol = p_copy.Schol[0]
        w,m,_ = p_copy.pop(0)
        split_comp = gm.split_gaussian(w, m, Schol, split_opts,  cov_type='cholesky')
        p_copy += split_comp
        new_split_err = gm.l2_dist(p, p_copy)
        assert(new_split_err < split_err)
        split_err = new_split_err


def test_eig_sqrt_factor():
    p = gm.defaults.default_gm(covariance_rotation=np.pi/6)

    for i, S in enumerate(p.Seig):
        npt.assert_array_almost_equal(S@S.T, p.P[i], decimal=10)
    # now try manipulating one of the sqrt factors
    fail(setattr, AttributeError, p, 'Seig', 2*p.Seig)
    # now manipulate one of the cov matrices and test that Seig is also updated
    p.P[0] *= 2
    for i, S in enumerate(p.Seig):
        npt.assert_array_almost_equal(S@S.T, p.P[i], decimal=10)


def test_cholesky_sqrt_factor():
    p = gm.defaults.default_gm(covariance_rotation=np.pi/6)

    for i, S in enumerate(p.Schol):
        npt.assert_array_almost_equal(S@S.T, p.P[i], decimal=10)
    # now try manipulating one of the sqrt factors
    fail(setattr, AttributeError, p, 'Schol', 2*p.Schol)
    # now manipulate one of the cov matrices and test that Schol is also updated
    p.P[0] *= 2
    for i, S in enumerate(p.Schol):
        npt.assert_array_almost_equal(S@S.T, p.P[i], decimal=10)


def test_precision_sqrt_factor():
    p = gm.defaults.default_gm(covariance_rotation=np.pi/6)

    for i, U in enumerate(p.prec_sqrt):
        npt.assert_array_almost_equal(np.linalg.inv(U@U.T), p.P[i], decimal=10)
    # now try manipulating one of the sqrt factors
    fail(setattr, AttributeError, p, 'prec_sqrt', 2*p.prec_sqrt)
    # now manipulate one of the cov matrices and test that Sprec is also updated
    p.P[0] *= 2
    for i, U in enumerate(p.prec_sqrt):
        npt.assert_array_almost_equal(np.linalg.inv(U@U.T), p.P[i], decimal=10)


def test_group_preserving_split_dir():
    # more intact rows --> split across vertical axis
    intact_rows = np.array([True, False, True])
    intact_cols = np.array([False, False, True])
    eigvals = np.array([1, 2])
    des_idx = 1
    assert(gm.group_preserving_split_dir(intact_rows, intact_cols, eigvals)==des_idx)

    # more intact cols --> split across horizontal axis
    intact_rows = np.array([True, False, True])
    intact_cols = np.array([True, True, True])
    des_idx = 0
    assert(gm.group_preserving_split_dir(intact_rows, intact_cols, eigvals)==des_idx)

    # same number of rows/cols, then pick one with largest eigval
    intact_rows = np.array(3*[True])
    intact_cols = np.array(3*[True])
    des_idx = 1
    assert(gm.group_preserving_split_dir(intact_rows, intact_cols, eigvals)==des_idx)


def test_split_for_fov():
    p = gm.defaults.default_gm(covariance_rotation=np.pi/6)
    fov = default_poly_fov()
    split_opts = gm.GaussSplitOptions(recurse_depth=4, L=5, lam=1e-4)
    p_split = gm.split_for_fov(p, fov, split_opts)
    split_err = gm.l2_dist(p, p_split)
    assert(split_err < 1e-3)

    # check that the integral of pdf outside fov matches split
    pp, XX, YY = p.pdf_2d(dimensions=(0, 1), res=400)
    box_area = (np.max(XX) - np.min(XX))*(np.max(YY) - np.min(YY))
    # compute the integral of the density over the box
    int_inside_box = box_area*np.mean(pp)
    # find which points are inside fov
    in_mask = fov.contains(np.vstack((XX.flatten(), YY.flatten())).T)
    in_mask_mat = np.reshape(in_mask, XX.shape)
    # set pdf evaluations inside fov to zero
    pp[in_mask_mat] = 0
    int_outside_fov_inside_box = box_area*np.mean(pp)
    true_int_outside_fov = (1 - int_inside_box) + int_outside_fov_inside_box

    # compute the sum of the weights of components outside the FoV
    comp_mask_in_fov = np.array(fov.contains(p_split.m[:, :2]))
    int_outside_fov = np.sum(p_split.w[~comp_mask_in_fov])
    assert(np.abs(true_int_outside_fov - int_outside_fov) < 0.02)

    # -------test 3D fov--------
    fov2d = default_poly_fov()
    # construct a 3D fov by adding a z coordinate
    # choose z limits of FoV to be +/- 1 sigma of the distribution
    z_ub = np.sqrt(p.cov()[2, 2])
    z_lb = -z_ub
    verts_3d = np.array([np.hstack((v,z)) for v in fov2d.verts for z in (z_lb, z_ub)])
    fov3d = ConvexPolyhedralFieldOfView(verts_3d)
    split_opts = gm.GaussSplitOptions(L=5, lam=1e-3, state_idxs=np.arange(3), min_weight=0.001)
    p_split = gm.split_for_fov(p, fov3d, split_opts)
    split_err = gm.l2_dist(p, p_split)
    assert(split_err < 1e-3)

    # check that the integral of pdf outside fov matches split
    p_lb = np.min(p.comp_bounds(sigma_mult=3)[0],axis=0)[:3]
    p_ub = np.max(p.comp_bounds(sigma_mult=3)[1],axis=0)[:3]
    XX,YY,ZZ = np.meshgrid(*[np.linspace(lb,ub,200) for lb,ub in zip(p_lb,p_ub)], indexing='ij')
    pp = p.marginal_nd((0,1,2))(np.vstack((XX.flatten(), YY.flatten(), ZZ.flatten())).T).reshape(XX.shape)

    box_area = np.prod(p_ub - p_lb)
    # compute the integral of the density over the box
    int_inside_box = box_area*np.mean(pp)
    # find which points are inside fov
    in_mask = fov3d.contains(np.vstack((XX.flatten(), YY.flatten(), ZZ.flatten())).T)
    in_mask_tensor = np.reshape(in_mask, XX.shape)
    # set pdf evaluations inside fov to zero
    pp[in_mask_tensor] = 0
    int_outside_fov_inside_box = box_area*np.mean(pp)
    true_int_outside_fov = (1 - int_inside_box) + int_outside_fov_inside_box

    # compute the sum of the weights of components outside the FoV
    comp_mask_in_fov = np.array(fov3d.contains(p_split.m[:, :3]))
    int_outside_fov = np.sum(p_split.w[~comp_mask_in_fov])
    assert(np.abs(true_int_outside_fov - int_outside_fov) < 0.02)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # p_oofov = gm.GaussianMixture(p_split.w[~comp_mask_in_fov], p_split.m[~comp_mask_in_fov], p_split.P[~comp_mask_in_fov])
    # pp, XX, YY = p_oofov.pdf_2d(dimensions=(0, 1), res=400)
    # ax.contourf(XX, YY, pp, 100)
    # ax.plot(fov3d.verts[:, 0], fov3d.verts[:, 1], 'k', lw=2)
    # ax.plot(p_split.m[~comp_mask_in_fov][:, 0], p_split.m[~comp_mask_in_fov][:, 1], 'ro', markersize=5)
    # plt.show()

def test_merge():
    p = gm.defaults.default_gm()

    # test with super small md threshold
    p_merged = gm.merge(p, md=1e-9)
    assert(p == p_merged)

    # test with really large md threshold
    p_merged = gm.merge(p, md=1e9)
    assert(len(p_merged) == 1)
    npt.assert_array_almost_equal(p_merged.m[0], p.mean())
    npt.assert_array_almost_equal(p_merged.P[0], p.cov())


def test_merge_runnalls(benchmark):
    # a mixture with two identical components, when merged, should be the same
    # component with double the weight
    w1 = 0.5
    w2 = 0.5
    m1 = np.array([1, 2])
    m2 = np.array([1, 2])
    P1 = 5*np.eye(2)
    P2 = 5*np.eye(2)

    p = gm.GaussianMixture([w1, w2], [m1, m2], [P1, P2])
    K = 1  # reduce to single component
    p_red = gm.merge_runnals(p, K)
    assert(len(p_red) == 1)
    assert(p_red.w[0] == 1)
    npt.assert_array_equal(m1, p_red.m[0])
    npt.assert_array_equal(P1, p_red.P[0])

    # now use the default mixture (w/ different components) and ensure that the
    # conditional mean and covariance are preserved
    p = gm.defaults.default_gm()
    p_red = benchmark(gm.merge_runnals, p, K)
    npt.assert_array_equal(p.mean(), p_red.mean())
    npt.assert_array_almost_equal(p.cov(), p_red.cov())


def test_pickle_gm():
    p = gm.defaults.default_gm()
    ptest = pickle.loads(pickle.dumps(p))
    assert(p == ptest)


def test_get_comp():
    p = gm.defaults.default_gm()
    # test with index
    comp = p.get_comp(0)
    npt.assert_array_equal(comp[0], p.w[0])
    npt.assert_array_equal(comp[1], p.m[0])
    npt.assert_array_equal(comp[2], p.P[0])

    # test with index
    comp = p.get_comp(1)
    npt.assert_array_equal(comp[0], p.w[1])
    npt.assert_array_equal(comp[1], p.m[1])
    npt.assert_array_equal(comp[2], p.P[1])


def test_init_mismatch_fail():

    # try initializing a GaussianMixture with mismatched weights, means, and covariances
    w = np.array([0.4, 0.6])
    m = np.array([[30., 0.], [10., np.pi/2]])
    P = np.array([[1., 0.5], [0.5, 3.4]])

    fail(gm.GaussianMixture, ValueError, w, m, P)

    w = np.array([0.4])
    m = np.array([[30., 0.], [10., np.pi/2]])
    P = np.array([[1., 0.5], [0.5, 3.4]])

    fail(gm.GaussianMixture, ValueError, w, m, P)


if __name__ == '__main__':
    pytest.main([__file__])
