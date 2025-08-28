import warnings
from copy import copy, deepcopy

import numpy as np
import scipy.optimize as sciopt
from diskcache import Cache
from scipy.linalg import cholesky, solve_triangular

from ..filters.sigma_points import unscented_transform, SigmaPoints
from ..linalg import choldowndate
from ..metrics import l2_dist
from ..tens import tensor_2_norm_trials_shifted, symmetrize_tensor
from .gm import _PSD, GaussianMixture

import jax

# create/load split cache
gm_split_l2_cache = Cache(__file__[:-3] + "l2_cache")
gm_split_l2_cov_cache = Cache(__file__[:-3] + "l2_cov_cache")


def _reflect_weights(w, L):
    """reflect weights so that the L-element set is symmetric

    Parameters
    ----------
    w: arraylike
        (ceil(L/2),) array of weights
    L: int
        length of symmetric weight set

    Returns
    -------
    ndarray
        L-element symmetric weight array

    """
    assert len(w) == np.ceil(L / 2)
    if L % 2:
        w = np.hstack((w[:-1], np.flip(w)))
    else:
        w = np.hstack((w, np.flip(w)))
    return w


def obj_l2_gauss_split(eps, sig, w, L, lam):
    """objective function for optimal Gausian split

    Parameters
    ----------
    L: int
        number of components in split
    lam: float
        optimization parameter that specifies the importance of standard
        deviation size and overall L2 distance. lam=0 will place zero
        importance on standard deviation, making objective equivalent to L2
        distance
    eps: float
        spacing between means
    sig: float
        standard deviation of components
    w: ndarray
        (ceil(L/2),) weights of candidate Gaussian mixture. To enforce
        symmetry, only the left-half weights are used

    Returns
    -------
    float
        value of objective function J

    """
    # reflect the weights
    w = _reflect_weights(w, L)

    assert w.shape[0] == L
    m2 = np.atleast_2d(equally_spaced_centered_means(L, eps)).T
    P2 = np.atleast_3d(L * [sig**2]).reshape((L, 1, 1))
    p2 = GaussianMixture(w=w, m=m2, cov=P2)
    p1 = GaussianMixture(w=1, m=0, cov=1)
    l2 = l2_dist(p1, p2)
    return l2 + lam * sig**2


def equally_spaced_centered_means(L, eps):
    """generate equally spaced means centered at zero

    Parameters
    ----------
    L: int
        number of components
    eps: float
        spacing between means

    Returns
    -------
    ndarray
        (L,) array of means
    """
    return eps * (np.arange(-(L - 1) / 2, (L - 1) / 2 + 1, 1))


def optimize_gauss_split_variance_preserving(L, lam):
    """optimize L-wise split of standard univariate Gaussian, preserving variance

    Parameters
    ----------
    L: int
        number of components to split into
    lam: float
        optimization parameter that specifies the importance of standard
        deviation size and overall L2 distance. lam=0 will place zero
        importance on standard deviation, making objective equivalent to L2
        distance

    Returns
    -------
    GaussianMixture
        L-component GM approximation of standard normal

    Notes
    -----
    This optimization process uses the following constraints to restrict the
    dimensionality of the search space:
    1. Means are evenly spaced
    2. The components are homoscedastic
    3. The distribution variance is preserved
    """

    L_half = int(np.ceil(L / 2))
    w_min = L_half * [0]
    w_max = L_half * [1]
    eps_min = 0
    eps_max = 3
    lb = np.hstack((eps_min, w_min))
    ub = np.hstack((eps_max, w_max))

    bounds = sciopt.Bounds(lb, ub)

    # initial guess at optimal parameters
    w0 = L_half * [1 / L]
    # for the chosen weights, compute the maximum spacing that guarantees non-negative variance
    idxs = np.arange(1, L + 1)
    eps_max_nonnegative = (
        1
        / L
        / np.sqrt(np.dot(_reflect_weights(w0, L), ((idxs - 1) / (L - 1) - 0.5) ** 2))
    )
    eps0 = 0.99 * eps_max_nonnegative

    x0 = np.hstack((eps0, w0))
    weight_constr = sciopt.NonlinearConstraint(
        lambda x: _gauss_split_weight_constr(_reflect_weights(x[1:], L)), lb=0, ub=0
    )

    var_pos_semi_def_constr = sciopt.NonlinearConstraint(
        lambda x: _gauss_split_pos_semi_def_constr(
            _reflect_weights(x[1:], L), equally_spaced_centered_means(L, x[0])
        ),
        lb=0,
        ub=1,
    )

    # perform optimization to find best mean spacing and weights
    def fun(x):
        m = equally_spaced_centered_means(L, x[0])
        # compute mixand variance that preserves variance
        w = _reflect_weights(x[1:], L)
        sqrt_arg = 1 - np.dot(w, m**2)
        if sqrt_arg < 0:
            return np.inf
        sig_opt = np.sqrt(sqrt_arg)
        return obj_l2_gauss_split(x[0], sig_opt, x[1:], L, lam)

    x_opt = sciopt.minimize(
        fun,
        x0,
        constraints=[weight_constr, var_pos_semi_def_constr],
        bounds=bounds,
        options={"ftol": 1e-17, "maxiter": 2000},
        method="SLSQP",
    )
    eps_opt, *w_half_opt = x_opt.x
    if not x_opt.success:
        warnings.warn("Optimization did not converge: " + x_opt.message)

    m_opt = np.atleast_2d(equally_spaced_centered_means(L, eps_opt)).T
    w_opt = _reflect_weights(w_half_opt, L)
    sig_opt = np.sqrt(1 - np.dot(w_opt, m_opt**2))
    S_opt = np.atleast_3d(L * [sig_opt]).reshape((L, 1, 1))
    return GaussianMixture(w=w_opt, m=m_opt, cov=S_opt, cov_type="cholesky")


def _gauss_split_weight_constr(w):
    return 1 - np.sum(w)


def _gauss_split_pos_semi_def_constr(w, m):
    # must be less than or equal to one
    return np.dot(w, m**2)


def optimize_gauss_split(L, lam):
    """optimize L-wise split of standard univariate Gaussian

    Parameters
    ----------
    L: int
        number of components to split into
    lam: float
        optimization parameter that specifies the importance of standard
        deviation size and overall L2 distance. lam=0 will place zero
        importance on standard deviation, making objective equivalent to L2
        distance

    Returns
    -------
    GaussianMixture
        L-component GM approximation of standard normal

    Notes
    -----
    This optimization process uses the following constraints to restrict the
    dimensionality of the search space:
    1. Means are evenly spaced
    2. The components are homoscedastic
    """

    L_half = int(np.ceil(L / 2))
    w_min = L_half * [0]
    w_max = L_half * [1]
    eps_min = 0
    eps_max = 3
    sig_min = 1e-6
    sig_max = 2
    lb = np.hstack((eps_min, sig_min, w_min))
    ub = np.hstack((eps_max, sig_max, w_max))

    bounds = sciopt.Bounds(lb, ub)

    # initial guess at optimal parameters
    eps0 = 3 / (L - 1)  # initial guess of mean spacing
    sig0 = 0.3  # initial guess of standard deviation
    w0 = L_half * [1 / L]

    x0 = np.hstack((eps0, sig0, w0))
    weight_constr = sciopt.NonlinearConstraint(
        lambda x: _gauss_split_weight_constr(_reflect_weights(x[2:], L)), lb=0, ub=0
    )

    # perform optimization to find best mean spacing, standard deviation, and
    # weights
    def fun(x):
        return obj_l2_gauss_split(x[0], x[1], x[2:], L, lam)

    x_opt = sciopt.minimize(
        fun, x0, constraints=[
            weight_constr], bounds=bounds, options={"ftol": 1e-17}
    )
    eps_opt, sig_opt, *w_half_opt = x_opt.x
    if not x_opt.success:
        warnings.warn("Optimization did not converge: " + x_opt.message)

    m_opt = np.atleast_2d(
        eps_opt * (np.arange(-(L - 1) / 2, (L - 1) / 2 + 1, 1))).T
    S_opt = np.atleast_3d(L * [sig_opt]).reshape((L, 1, 1))
    w_opt = _reflect_weights(w_half_opt, L)
    return GaussianMixture(w=w_opt, m=m_opt, cov=S_opt, cov_type="cholesky")


def split_1d_standard_gaussian(L, lam, variance_preserving=True):
    """split 1D standard univariate Gaussian into L-component GM

    Parameters
    ----------
    L: int
        number of components to split into
    lam: float
        optimization parameter that specifies the importance of standard
        deviation size and overall L2 distance. lam=0 will place zero
        importance on standard deviation, making objective equivalent to L2
        distance
    variance_preserving: bool
        if True, the L2 distance will be minimized while preserving the
        variance of the original Gaussian. If False, the L2 distance will be
        minimized without regard to the variance of the original Gaussian

    Returns
    -------
    GaussianMixture
        L-component GM approximation of standard normal
    """
    if not variance_preserving:
        # first check cache
        if gm_split_l2_cache.get((L, lam)) is not None:
            gmm = gm_split_l2_cache.get((L, lam))
        # if not in cache, perform optimization
        else:
            gmm = optimize_gauss_split(L, lam)
            gm_split_l2_cache[(L, lam)] = gmm

        return gmm
    else:
        # first check cache
        if gm_split_l2_cov_cache.get((L, lam)) is not None:
            gmm = gm_split_l2_cov_cache.get((L, lam))
        # if not in cache, perform optimization
        else:
            gmm = optimize_gauss_split_variance_preserving(L, lam)
            gm_split_l2_cov_cache[(L, lam)] = gmm

        return gmm


def split_gaussian(w, m, cov, split_options, cov_type="full", direction=None):
    """split multivariate Gaussian into smaller Gaussians

    Parameters
    ----------
    w: float
        component weight
    m: ndarray
        (nx,) component mean
    cov: ndarray
        (nx, nx) component covariance
    split_options: GaussSplitOptions
        splitting options
    cov_type: str, optional
        type of covariance provided. Default is 'full'
    direction: ndarray, optional
        (nx,) desired direction along which to split. The covariance
        eigenvector that closest matches this direction is used. If
        direction is None, the direction of largest variance will be used.
    """
    # load solution from library
    gm_1d = split_1d_standard_gaussian(
        split_options.L,
        split_options.lam,
        variance_preserving=split_options.variance_preserving,
    )

    # the Cholesky factor is used in all cases
    if cov_type == "full":
        S = cholesky(cov, lower=True)
    elif cov_type == "cholesky":
        S = cov
    else:
        raise ValueError("Covariance matrix type not recognized")

    if direction is not None:
        # normalize direction

        direction /= np.linalg.norm(direction)

    # split in arbitrary direction
    if direction is not None and not split_options.spectral_direction_only:
        std_u = 1 / np.linalg.norm(solve_triangular(S, direction, lower=True))
        m_split = np.array([m + std_u * m1d * direction for m1d in gm_1d.m])
        # compute downdate factor
        a = std_u * np.sqrt(np.dot(gm_1d.w, gm_1d.m**2)) * direction
        # downdate to find split cholesky factor
        S_bar = choldowndate(S.T, a).T
        assert np.all(np.diag(S_bar) >= 0)
        w_split = w * gm_1d.w
        S_split = np.tile(S_bar, (split_options.L, 1, 1))
        p_split = GaussianMixture(
            w_split, m_split, S_split, cov_type="cholesky")
        return p_split

    # spectral splitting
    if cov_type == "full":
        P = cov
    elif cov_type == "cholesky":
        P = cov @ cov.T

    # if splitting in a spectral direction, find the eigenvals and eigenvecs
    eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(P))

    if direction is not None:
        # compute which eigenvector is closest to desired direction
        eig_idx = np.argmax(np.abs(eigvecs.T @ direction))
    else:
        # split along maximum variance direction
        eig_idx = np.argmax(eigvals)

    p_split = split_spectral_direction(
        w, m, S, eigvecs, eigvals, eig_idx, gm_1d.w, gm_1d.m, gm_1d.P[0, 0, 0]
    )
    return p_split


def split_spectral_direction(w, m, S, V, D, eig_idx, wt, mt, Pt):
    """split Gaussian along spectral direction

    Parameters
    ----------
    w: float
        component weight
    m: ndarray
        (nx,) component mean
    S: ndarray
        (nx, nx) component covariance lower Cholesky factor
    V: ndarray
        (nx, nx) spectral decomposition eigenvectors
    D: ndarray
        (nx,,nx) spectral decomposition eigenvalues diagonal matrix
    eig_idx: int
        index of eigenvector to split along
    wt: ndarray
        (L,) weights of the standard split library
    mt: ndarray
        (L, nx) means of the standard split library
    Pt: ndarray

    Returns
    -------
    GaussianMixture
        L-component GM approximation of the input weighted Gaussian
    """

    eig_val = D[eig_idx]
    eigValSqrt = np.sqrt(np.real(eig_val))

    # update Cholesky factor
    a = (
        V[:, eig_idx] * eigValSqrt * np.sqrt(1 - Pt)
    )  # downdate parameter a*a' = S*S' - Si*Si'

    S_new = choldowndate(S.T, a).T  # downdate to find split cholesky factor

    nX = len(m)
    L = len(wt)

    wgm = w * wt
    Sgm = np.tile(S_new, (L, 1, 1))

    mgm = np.full((L, nX), np.nan)

    for i in range(L):
        mgm[i] = m + eigValSqrt * mt[i] * V[:, eig_idx]
    assert np.all(np.isreal(mgm))

    p_split = GaussianMixture(wgm, mgm, Sgm, cov_type="cholesky")
    return p_split


def identify_split_components(p, fovs, split_opts):
    """Identify what components of a GM should be split for a FoV

    Parameters
    ----------
    p: GaussianMixture
      Gaussian mixture to split
    fovs: PolygonalFieldOfView or list
      FoV, the geometry of which determines which components to split
    split_opts:  GaussSplitOptions
        splitting options

    Returns
    -------
    split_mask: ndarray
      boolean array, where true elements denote a component that should be
      split
    split_dir: ndarray
      (nC, nX) array where each row indicates the best direction to split
      along. Non-positional state elements are left as zero

    """

    pos_idxs = split_opts.state_idxs

    assert split_opts.recurse_depth > 0

    split_dir = np.zeros((len(p), p.dim))
    split_mask = np.array(len(p) * [False])

    if not isinstance(fovs, list):
        fovs = [fovs]

    # generate the grid of test points
    # TODO: check case where FOV is wholly contained with grid bounds, but
    # no grid points are within FOV.
    L_grid = 15
    zmin = -2
    zmax = -zmin
    mt = np.linspace(zmin, zmax, L_grid)
    XX, YY = np.meshgrid(mt, mt)
    test_pts = np.vstack((XX.flatten(), YY.flatten())).T
    in_sphere_mask = XX**2 + YY**2 <= zmax**2
    num_pts_in_slice = np.sum(in_sphere_mask, axis=0)

    gm_lb, gm_ub = p.comp_bounds(sigma_mult=3)

    for i, (w, m, P) in enumerate(p):
        # if weight is below threshold, don't split and move on
        if w < split_opts.min_weight:
            split_mask[i] = False
            continue
        # compute eigenvectors and eigenvalues. Eigenvectors and eigenvalues
        # are used to perform a change of variables so that the component
        # density is the standard Gaussian density with zero mean and unit
        # covariance.
        eigvals, V = np.linalg.eigh(P[pos_idxs[:, np.newaxis], pos_idxs])

        # compute the corners of the fov relative to the mean, then rotate them
        # into eigenspace and scale them by the variance

        multifov_intact_rows = np.full(L_grid, True)
        multifov_intact_cols = np.full(L_grid, True)
        # TODO: for each component, maintain list of which FoVs are irrelevant and skip
        for fov in fovs:
            if any(fov.ub < gm_lb[i, pos_idxs]) or any(fov.lb > gm_ub[i, pos_idxs]):
                split_mask[i] = False
                continue

            fov_rot_scaled = fov.apply_linear_transformation(
                np.diag(eigvals**-0.5) @ V.T, pre_shift=-m[pos_idxs]
            )
            # for each fov, find out which columns and rows are intact, i.e.
            # totally contained within the fov or totally excluded
            in_mask = np.array(fov_rot_scaled.contains(test_pts))
            in_mask_mat = in_mask.reshape(XX.shape)
            # ignore points outside the sphere
            in_mask_mat[~in_sphere_mask] = False
            fov_intact_rows, fov_intact_cols = find_intact_rows_cols(
                in_mask_mat, num_pts_in_slice
            )

            multifov_intact_rows = np.logical_and(
                multifov_intact_rows, fov_intact_rows)
            multifov_intact_cols = np.logical_and(
                multifov_intact_cols, fov_intact_cols)

        if np.all(multifov_intact_rows) and np.all(multifov_intact_cols):
            split_mask[i] = False
            continue
        else:
            split_mask[i] = True

        best_dir_idx = group_preserving_split_dir(
            multifov_intact_rows, multifov_intact_cols, eigvals
        )

        split_dir[i, pos_idxs] = deepcopy(V[:, best_dir_idx])
        split_dir[i] = P @ split_dir[i]

    return split_mask, split_dir


def identify_split_components_3d_fov(p, fovs, split_opts):
    """Identify what components of a GM should be split for a set of 3D FoVs

    Parameters
    ----------
    p: GaussianMixture
      Gaussian mixture to split
    fovs: FieldOfView or list
      FoV, the geometry of which determines which components to split
    split_opts:  GaussSplitOptions
        splitting options

    Returns
    -------
    split_mask: ndarray
      boolean array, where true elements denote a component that should be
      split
    split_dir: ndarray
      (nC, nX) array where each row indicates the best direction to split
      along. Non-positional state elements are left as zero

    """

    pos_idxs = split_opts.state_idxs
    assert len(
        pos_idxs) == 3, 'split_opts.state_idxs must contain 3 unique indices'

    assert split_opts.recurse_depth > 0

    split_dir = np.zeros((len(p), p.dim))
    split_mask = np.array(len(p) * [False])

    if not isinstance(fovs, list):
        fovs = [fovs]

    # generate the grid of test points
    # TODO: check case where FOV is wholly contained with grid bounds, but
    # no grid points are within FOV.
    L_grid = 15
    zmin = -2
    zmax = -zmin
    mt = np.linspace(zmin, zmax, L_grid)
    XX, YY, ZZ = np.meshgrid(mt, mt, mt, indexing='ij')
    test_pts = np.vstack((XX.flatten(), YY.flatten(), ZZ.flatten())).T
    in_sphere_mask = XX**2 + YY**2 + ZZ**2 <= zmax**2 + 1e-10
    num_pts_in_slice = np.sum(in_sphere_mask, axis=(0, 1))

    gm_lb, gm_ub = p.comp_bounds(sigma_mult=3)

    for i, (w, m, P) in enumerate(p):
        # if weight is below threshold, don't split and move on
        if w < split_opts.min_weight:
            split_mask[i] = False
            continue
        # compute eigenvectors and eigenvalues. Eigenvectors and eigenvalues
        # are used to perform a change of variables so that the component
        # density is the standard Gaussian density with zero mean and unit
        # covariance.
        eigvals, V = np.linalg.eigh(P[pos_idxs[:, np.newaxis], pos_idxs])

        # compute the corners of the fov relative to the mean, then rotate them
        # into eigenspace and scale them by the variance

        multifov_intact_yz_plane = np.full(L_grid, True)
        multifov_intact_xz_plane = np.full(L_grid, True)
        multifov_intact_xy_plane = np.full(L_grid, True)
        # TODO: for each component, maintain list of which FoVs are irrelevant and skip
        for fov in fovs:
            if any(fov.ub < gm_lb[i, pos_idxs]) or any(fov.lb > gm_ub[i, pos_idxs]):
                split_mask[i] = False
                continue

            fov_rot_scaled = fov.apply_linear_transformation(
                np.diag(eigvals**-0.5) @ V.T, pre_shift=-m[pos_idxs]
            )
            # for each fov, find out which slices are intact, i.e.
            # totally contained within the fov or totally excluded
            in_mask = np.array(fov_rot_scaled.contains(test_pts))
            in_mask_tensor = in_mask.reshape(XX.shape)
            # ignore points outside the sphere
            in_mask_tensor[~in_sphere_mask] = False

            fov_intact_xy_plane, fov_intact_xz_plane, fov_intact_yz_plane = find_intact_slices(
                in_mask_tensor, num_pts_in_slice
            )

            multifov_intact_xy_plane = np.logical_and(
                multifov_intact_xy_plane, fov_intact_xy_plane)
            multifov_intact_xz_plane = np.logical_and(
                multifov_intact_xz_plane, fov_intact_xz_plane)
            multifov_intact_yz_plane = np.logical_and(
                multifov_intact_yz_plane, fov_intact_yz_plane)

        if np.all(multifov_intact_xy_plane) and np.all(multifov_intact_xz_plane) and np.all(multifov_intact_yz_plane):
            split_mask[i] = False
            continue
        else:
            split_mask[i] = True

        multifov_intact_dims = np.array([np.sum(multifov_intact_yz_plane),
                                         np.sum(multifov_intact_xz_plane),
                                         np.sum(multifov_intact_xy_plane)])

        # plot the collocation points, shading according to whether they are in the FoV
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel('v1')
        # ax.set_ylabel('v2')
        # ax.set_zlabel('v3')
        # ax.scatter(XX[in_sphere_mask], YY[in_sphere_mask], ZZ[in_sphere_mask], c=in_mask_tensor[in_sphere_mask])
        # plot the 3D polyhedral FoV geometry. The polehdron is represented by a scipy.spatial hull
        # Plot the tetrahedra
        # for simplex in fov_rot_scaled._hull.simplices:
        #    # Each simplex is a tetrahedron; plot its faces
        #    for j in range(4):
        #        face = np.delete(simplex, j)
        #        poly = fov_rot_scaled.verts[face]
        #        ax.add_collection3d(Poly3DCollection([poly], alpha=0.2, color='cyan'))

        # the position split direction is chosen as the direction that is
        # orthogonal to the most grid planes of consistent inclusion/exclusion.
        slice_dim = np.argmax(multifov_intact_dims)
        # check if there are dimensions with equally intact slices
        if np.sum(multifov_intact_dims[slice_dim] == multifov_intact_dims) > 1:
            equally_intact_dims = np.where(
                multifov_intact_dims[slice_dim] == multifov_intact_dims)[0]
            # split along equally intact direction with largest variance
            best_dir_idx = equally_intact_dims[0] if eigvals[equally_intact_dims[0]
                                                             ] >= eigvals[equally_intact_dims[1]] else equally_intact_dims[1]
        else:
            best_dir_idx = slice_dim

        split_dir[i, pos_idxs] = deepcopy(V[:, best_dir_idx])
        split_dir[i] = P @ split_dir[i]

    return split_mask, split_dir


def group_preserving_split_dir(intact_rows, intact_cols, eigvals):
    """choose direction for FoV splitting to minimize number of downstream
    components

    Parameters
    ----------
    intact_cols: ndarray
        (L,) logical, where intact_cols[i]=True indicates that all grid points
        in the ith column are either entirely included in the FoV or entirely
        excluded
    intact_rows: ndarray
        (L,) logical, where intact_rows[i]=True indicates that all grid points
        in the ith row are either entirely included in the FoV or entirely
        excluded
    eigvals: ndarray
        (2,) positional covariance eigenvalues. In the case that the
        direction cannot be determined by point inclusion, the direction will
        be chosen according to the largest positional covariance eigenvalue.

    Returns
    -------
    int
        index of direction to split along. 0 corresponds to a split along the
        horizontal axis, and 1 corresponds to a split along the vertical axis
    """
    num_intact_rows = np.sum(intact_rows)
    num_intact_cols = np.sum(intact_cols)

    # if there are more intact rows than intact columns, split along the y
    # axis first
    if num_intact_rows > num_intact_cols:
        # split along y axis
        best_dir_idx = 1
    elif num_intact_rows < num_intact_cols:
        # split along x axis
        best_dir_idx = 0
    else:
        # split along direction with largest variance
        max_eig_idx = np.argmax(eigvals)
        best_dir_idx = max_eig_idx
    return best_dir_idx


def find_intact_slices(in_mask_tensor, num_pts_in_slice):
    """find slices that are totally included/excluded

    Parameters
    ----------
    in_mask_tensor: ndarray
        (L,L,L) boolean mask indicating point-wise inclusion by the FoV, where
        L is the number of equally spaced points per dimension
    num_pts_in_slice: int
        (L,) number of valid collocation points in each slice

    Returns
    -------
    intact_xy_plane: ndarray
        (L,) logical, where intact_xy[i]=True indicates that all grid points
        in the ith horizontal slice [i,:,:] are either entirely included in the FoV or entirely
        excluded
    intact_xz_plane: ndarray
        (L,) logical, where intact_xz_plane[j]=True indicates that all grid points
        in the jth lateral slice [:,j,:] are either entirely included in the FoV or entirely
        excluded
    intact_yz_plane: ndarray
        (L,) logical, where intact_xz_plane[k]=True indicates that all grid points
        in the kth frontal slice [:,:,k] are either entirely included in the FoV or entirely
        excluded
    """
    assert in_mask_tensor.shape[0] == in_mask_tensor.shape[1]
    assert in_mask_tensor.shape[1] == in_mask_tensor.shape[2]

    intact_xy_plane = np.sum(in_mask_tensor, axis=(0, 1))
    intact_xy_plane = np.logical_or(
        intact_xy_plane == num_pts_in_slice, intact_xy_plane == 0)
    intact_xz_plane = np.sum(in_mask_tensor, axis=(0, 2))
    intact_xz_plane = np.logical_or(
        intact_xz_plane == num_pts_in_slice, intact_xz_plane == 0)
    intact_yz_plane = np.sum(in_mask_tensor, axis=(1, 2))
    intact_yz_plane = np.logical_or(
        intact_yz_plane == num_pts_in_slice, intact_yz_plane == 0)
    return intact_xy_plane, intact_xz_plane, intact_yz_plane


def find_intact_rows_cols(in_mask_mat, num_pts_in_slice):
    """find rows and columns that are totally included/excluded

    Parameters
    ----------
    in_mask_mat: ndarray
        (L,L) boolean mask indicating point-wise inclusion by the FoV, where
        L is the number of equally spaced points per dimension
    num_pts_in_slice: int
        (L,1) number of collocation points in each slice

    Returns
    -------
    intact_cols: ndarray
        (L,) logical, where intact_cols[i]=True indicates that all grid points
        in the ith column are either entirely included in the FoV or entirely
        excluded
    intact_rows: ndarray
        (L,) logical, where intact_rows[i]=True indicates that all grid points
        in the ith row are either entirely included in the FoV or entirely
        excluded
    """
    assert in_mask_mat.shape[0] == in_mask_mat.shape[1]
    intact_cols = np.sum(in_mask_mat, axis=0)

    intact_cols = np.logical_or(
        intact_cols == num_pts_in_slice, intact_cols == 0)
    intact_rows = np.sum(in_mask_mat, axis=1)
    intact_rows = np.logical_or(
        intact_rows == num_pts_in_slice, intact_rows == 0)
    return intact_rows, intact_cols


def split_for_fov(p, fovs, split_opts):
    """
    SPLIT_FOR_FOV Splits Gaussians that are close to FoV edges

        Assumes that first two dimensions correspond to the same coordinates
        that the FOV is expressed in

        Parameters
        ----------
        p: GaussianMixture
            mixture to be split
        fovs: list
            FieldOfView, where the boundaries are used to determine where
            to split the distribution
        split_opts:  GaussSplitOptions
            splitting options

        Returns
        -------
        p_split: GaussianMixture
            split Gaussian mixture

        Written by Keith LeGrand
    """

    if split_opts.recurse_depth == 0:
        return p
    # identify the components that need split and compute the split direction
    if len(split_opts.state_idxs) == 2:
        split_mask, split_dir = identify_split_components(p, fovs, split_opts)
    elif len(split_opts.state_idxs) == 3:
        split_mask, split_dir = identify_split_components_3d_fov(
            p, fovs, split_opts)
    else:
        raise ValueError(
            "split_opts.state_idxs must contain 2 or 3 unique indices")
    n2split = np.sum(split_mask)
    if n2split == 0:
        return p

    # allocate memory
    w_split = np.full([n2split * split_opts.L], np.nan)
    m_split = np.full([n2split * split_opts.L, p.dim], np.nan)
    P_split = np.full([n2split * split_opts.L, p.dim, p.dim], np.nan)
    S_split = np.full([n2split * split_opts.L, p.dim, p.dim], np.nan)

    # create a queue of components for splitting
    w_q, m_q, P_q = p[split_mask]
    S_q = p.Seig[split_mask]

    split_dir_q = split_dir[split_mask]
    idx = 0
    while len(w_q) > 0:
        # pop the elements from the queue
        wi, w_q = w_q[0], w_q[1:]
        mi, m_q = m_q[0], m_q[1:]
        Pi, P_q = P_q[0], P_q[1:]
        Si, S_q = S_q[0], S_q[1:]
        diri, split_dir_q = split_dir_q[0], split_dir_q[1:]

        pi_split = split_gaussian(wi, mi, Pi, split_opts, "full", diri)

        assert np.all(np.isreal(pi_split.m))
        w_split[idx: idx + split_opts.L] = pi_split.w
        m_split[idx: idx + split_opts.L] = pi_split.m
        P_split[idx: idx + split_opts.L] = pi_split.P
        S_split[idx: idx + split_opts.L] = pi_split.Seig
        idx += split_opts.L

    p_split = GaussianMixture(w_split, m_split, P_split, Seig=S_split)
    # recurse until no further splitting is needed
    split_opts_copy = copy(split_opts)
    split_opts_copy.recurse_depth -= 1
    p_split = split_for_fov(p_split, fovs, split_opts_copy)

    if np.any(~split_mask):
        # remove originals
        p_no_split = GaussianMixture(*p[~split_mask])
        return p_no_split + p_split
    else:
        return p_split


def recursive_split(p, split_opts, identify_split_components, *args):
    """
    RECURSIVE_SPLIT Splits Gaussians recursively based on a splitting criterion

        Parameters
        ----------
        p: GaussianMixture
            mixture to be split
        split_opts:  MixtureSplittingOptions
            splitting options
        indentify_split_components: callable
            function to identify components to split, of the form
            split_mask, split_dir = identify_split_components(p, *args)

        Returns
        -------
        p_split: GaussianMixture
            split Gaussian mixture

    """

    if split_opts.recurse_depth == 0:
        return p
    # identify the components that need split and compute the split direction
    split_mask, split_dir = identify_split_components(p, *args)
    n2split = np.sum(split_mask)
    if n2split == 0:
        return p

    # allocate memory
    w_split = np.full([n2split * split_opts.L], np.nan)
    m_split = np.full([n2split * split_opts.L, p.dim], np.nan)
    S_split = np.full([n2split * split_opts.L, p.dim, p.dim], np.nan)

    # create a queue of components for splitting
    w_q, m_q, _ = p[split_mask]
    S_q = p.Schol[split_mask]

    split_dir_q = split_dir[split_mask]
    idx = 0
    while len(w_q) > 0:
        # pop the elements from the queue
        wi, w_q = w_q[0], w_q[1:]
        mi, m_q = m_q[0], m_q[1:]
        Si, S_q = S_q[0], S_q[1:]
        diri, split_dir_q = split_dir_q[0], split_dir_q[1:]

        pi_split = split_gaussian(wi, mi, Si, split_opts, "cholesky", diri)

        assert np.all(np.isreal(pi_split.m))
        w_split[idx: idx + split_opts.L] = pi_split.w
        m_split[idx: idx + split_opts.L] = pi_split.m
        S_split[idx: idx + split_opts.L] = pi_split.Schol
        idx += split_opts.L

    p_split = GaussianMixture(w_split, m_split, S_split, "cholesky")
    # recurse until no further splitting is needed
    split_opts_copy = copy(split_opts)
    split_opts_copy.recurse_depth -= 1
    p_split = recursive_split(
        p_split, split_opts_copy, identify_split_components, *args
    )

    if np.any(~split_mask):
        # remove originals
        p_no_split = GaussianMixture(*p[~split_mask])
        return p_no_split + p_split
    else:
        return p_split


def id_variance(p, tol):
    """identify components and split directions based on variance

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    tol : float
        tolerance for splitting, given as a standard deviation.
        If the weighted standard deviation of the considered component is greater
        than tol in any spectral direction, the component is marked for splitting

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component
    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, P in enumerate(p.P):
        eigvals, eigvecs = np.linalg.eigh(P)
        if np.any(np.sqrt(eigvals) > tol):
            split_mask[i] = True
            split_dir[i] = eigvecs[:, -1]
    return split_mask, split_dir


def id_fos(p, jacobian_func, tol):
    """identify components and split directions based on first-order stretching

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    jacobian_func : callable
        function that returns the Jacobian of the nonlinear function
    tol : float
        tolerance for splitting. If the weighted norm exceeds tol, the component
        is marked for splitting

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component

    Notes
    -----
    Let :math:`\mathbf{G}` be the Jacobian of the nonlinear function. The first order
    stretching (FOS) measure is given by

    .. math::

        \max_{|\mathbf{x}\|_{2} = 1} \|\mathbf{G}\mathbf{x}\|_{2}

    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    See Also
    --------
    id_usfos : uncertainty scaled first-order stretching
    id_safos : spherical average first-order stretching

    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        # compute right singular value corresponding to the largest singular value:
        U, S, Vh = np.linalg.svd(jacobian_func(*m))
        if p.w[i] * S[0] > tol:
            split_mask[i] = True
            split_dir[i] = Vh[0]
    return split_mask, split_dir


def id_safos(p, jacobian_func, tol):
    """identify components and split directions based on spherical average first-order stretching

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    jacobian_func : callable
        function that returns the Jacobian of the measurement model
    tol : float
        tolerance for splitting.

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component

    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        S = p.Schol[i]
        P = p.P[i]
        G = jacobian_func(*m)
        M = S.T @ G.T @ G @ S
        A = P * np.trace(M) + 2 * P @ G.T @ G @ P
        eigs = np.linalg.eigh(A)
        if p.w[i] * np.sqrt(eigs.eigenvalues[-1]) > tol:
            split_mask[i] = True
            split_dir[i] = eigs.eigenvectors[:, -1]
    return split_mask, split_dir


def id_usfos(p, jacobian_func, tol):
    """identify components and split directions based on scaled first-order stretching

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    jacobian_func : callable
        function that returns the Jacobian of the measurement model
    tol : float
        tolerance for splitting. If the weighted norm exceeds tol, the component
        is marked for splitting

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component

    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    """

    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        # compute right singular value corresponding to the largest singular value:
        U, S, Vh = np.linalg.svd(jacobian_func(*m) @ p.Schol[i])
        if p.w[i] * S[0] > tol:
            split_mask[i] = True
            # take the maximal right singular vector of GS and the inverse coordinate transform
            split_dir[i] = p.Schol[i] @ Vh[0]
    return split_mask, split_dir


def id_sos(p, pdt_func, jacobian_func, tol, single_fn=False):
    """identify components and split directions based on nonlinear stretching

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    pdt_func : callable
        nonlinear function partial derivative tensor of the form
        pdt_func(x1, x2, ..., xn)
    jacobian_func : callable
        nonlinear function Jacobian of the form jacobian_func(x1, x2, ..., xn)
    tol : float
        tolerance for splitting, given as a Mahalanobis distance. If
        e^T Pf^-1 e > tol, the component is marked for splitting, where
        Pf is the linearly-mapped covariance of the considered component
    single_fn : bool
        interpret pdt_func argument as returning the tuple (x_f, jac, hess)
        and disregard jacobian function completely

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component

    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        if single_fn:
            fn_info = pdt_func(*m)
            G = fn_info[1]
            mpdt = fn_info[2]
        else:
            # compute measurement partial derivative matrix
            G = jacobian_func(*m)
            # compute measurement partial derivative tensor
            mpdt = np.array(pdt_func(*m))
        tens_norm, split_dir_i = tensor_2_norm_trials_shifted(mpdt)
        if p.w[i] * tens_norm > tol:
            split_mask[i] = True
            split_dir[i] = split_dir_i
    return split_mask, split_dir


def id_wussos(p, pdt_func, jacobian_func, tol, single_fn=False):
    """identify components and split directions based on scaled nonlinear stretching

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    pdt_func : callable
        nonlinear function second-order partial derivative tensor of the form
        pdt_func(x1, x2, ..., xn)
    jacobian_func : callable
        nonlinear function Jacobian of the form jacobian_func(x1, x2, ..., xn)
    tol : float
        tolerance for splitting, given as a Mahalanobis distance. If
          e^T Pf^-1 e > tol, the component is marked for splitting, where
          Pf is the linearly-mapped covariance of the considered component
    single_fn : bool
        interpret pdt_func argument as returning the tuple (x_f, jac, hess)
        and disregard jacobian function completely

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component


    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        if single_fn:
            fn_info = pdt_func(*m)
            G = fn_info[1]
            mpdt = fn_info[2]
        else:
            # compute measurement partial derivative matrix
            G = jacobian_func(*m)
            # compute measurement partial derivative tensor
            mpdt = np.array(pdt_func(*m))

        # compute linearly-mapped covariance
        Pf = G @ p.P[i] @ G.T
        # find square root factor of precision matrix Pf^-1 = U@U^T
        # U = np.linalg.inv(cholesky(Pf, lower=True)).T
        # # output-whitened wcovariance adjusted measurement partial derivative tensor
        # owcampdt = np.einsum("ir,rlm,lj,mk->ijk", U.T,
        #                      mpdt, p.Schol[i], p.Schol[i])
        campdt = np.einsum("ilm,lj,mk->ijk",
                              mpdt, p.Schol[i], p.Schol[i])
        owcampdt = np.transpose(jax.scipy.linalg.solve_triangular(np.tile(cholesky(Pf, lower=True).T, (campdt.shape[1],campdt.shape[2],1,1)), np.transpose(campdt, (1,2,0)), lower=False), (2,0,1))

        tens_norm, split_dir_transformed = tensor_2_norm_trials_shifted(
            owcampdt)
        if p.w[i] * tens_norm > tol:
            split_mask[i] = True
            split_dir[i] = p.Schol[i] @ split_dir_transformed
    return split_mask, split_dir


def id_solc(p, pdt_func, tol):
    """identify components and split directions based on second-order
    linearization change

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    pdt_func : callable
        nonlinear function partial derivative tensor of the form
        pdt_func(x1, x2, ..., xn)


    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024
    .. [2] K. Tuggle and R. Zanetti, “Automated Splitting Gaussian mixture
           Nonlinear Measurement Update,” Journal of Guidance, Control, and Dynamics,
           vol. 41, no. 3, pp. 725–734, 2018.
    .. [3] K. Tuggle, “Model Selection for Gaussian Mixture Model Filtering and
           Sensor Scheduling,” Ph.D. dissertation, 2020.

    See Also
    --------
    id_ussolc : uncertainty scaled second-order linearization change
    id_wussolc : whitened uncertainty scaled second-order linearization change

    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        mpdt = np.array(pdt_func(*m))
        # max right singular vec of the measurement partial derivative tensor
        # flattened to be tall and skinny matrix
        U, S, Vh = np.linalg.svd(
            np.reshape(mpdt, (np.prod(mpdt.shape[:2]), mpdt.shape[2]))
        )
        if p.w[i] * S[0] > tol:
            split_mask[i] = True
            split_dir[i] = Vh[0]
    return split_mask, split_dir


def id_ussolc(p, pdt_func, tol):
    """identify components and split directions based on uncertainty scaled
    second-order linearization change

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    pdt_func : callable
        function that returns the second-order partial derivative tensor of the
        nonlinear function
    tol : float
        tolerance for splitting. If the weighted norm exceeds tol, the component
        is marked for splitting

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting


    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024
    .. [2] K. Tuggle and R. Zanetti, “Automated Splitting Gaussian mixture
           Nonlinear Measurement Update,” Journal of Guidance, Control, and Dynamics,
           vol. 41, no. 3, pp. 725–734, 2018.
    .. [3] K. Tuggle, “Model Selection for Gaussian Mixture Model Filtering and
           Sensor Scheduling,” Ph.D. dissertation, 2020.

    See Also
    --------
    id_solc : second-order linearization change
    id_wussolc : whitened uncertainty scaled second-order linearization change
    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        # compute measurement partial derivative tensor
        mpdt = np.array(pdt_func(*m))
        # covariance adjusted measurement partial derivative tensor
        campdt = np.einsum("ilm,lj,mk->ijk", mpdt, p.Schol[i], p.Schol[i])
        # max right singular vec of the covariance adjusted measurement partial derivative tensor
        # flattened to be tall and skinny matrix
        U, S, Vh = np.linalg.svd(
            np.reshape(campdt, (np.prod(mpdt.shape[:2]), mpdt.shape[2]))
        )
        if p.w[i] * S[0] > tol:
            split_mask[i] = True
            split_dir[i] = p.Schol[i] @ Vh[0]
    return split_mask, split_dir


def id_wussolc(p, pdt_func, jacobian_func, tol, single_fn=False):
    """identify components and split directions based on output-whitened
    uncertainty scaled second-order linearization change

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    pdt_func : callable
        function that returns the nonlinear function's second-order partial
        derivative tensor of the nonlinear function of the form
        pdt_func(x1, x2, ..., xn)
    jacobian_func : callable
        function that returns the Jacobian of the nonlinear function
    tol : float
        tolerance for splitting. If the weighted norm exceeds tol, the component
        is marked for splitting
    single_fn : bool
        interpret pdt_func argument as returning the tuple (x_f, jac, pdt)
        and disregard jacobian function completely

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component


    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    See Also
    --------
    id_solc : second-order linearization change
    id_ussolc : uncertainty scaled second-order linearization change

    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        if single_fn:
            fn_info = pdt_func(*m)
            G = fn_info[1]
            mpdt = fn_info[2]
        else:
            # compute measurement partial derivative matrix
            G = jacobian_func(*m)
            # compute measurement partial derivative tensor
            mpdt = np.array(pdt_func(*m))

        # compute linearly-mapped covariance
        Pf = G @ p.P[i] @ G.T
        campdt = np.einsum("ilm,lj,mk->ijk",
                              mpdt, p.Schol[i], p.Schol[i])
        owcampdt = np.transpose(jax.scipy.linalg.solve_triangular(np.tile(cholesky(Pf, lower=True).T, (campdt.shape[1],campdt.shape[2],1,1)), np.transpose(campdt, (1,2,0)), lower=False), (2,0,1))


        # max right singular vec of the covariance adjusted measurement partial derivative tensor
        # flattened to be tall and skinny matrix
        U, S, Vh = np.linalg.svd(
            np.reshape(owcampdt, (np.prod(mpdt.shape[:2]), mpdt.shape[2]))
        )
        if p.w[i] * S[0] > tol:
            split_mask[i] = True
            split_dir[i] = p.Schol[i] @ Vh[0]
    return split_mask, split_dir


def id_sasos(p, pdt_func, tol):
    """identify components and split directions based on scaled nonlinear stretching

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    pdt_func : callable
        nonlinear function second-order partial derivative tensor of the form
        pdt_func(x1, x2, ..., xn)
    tol : float
        tolerance for splitting, given as a Mahalanobis distance. If
          e^T Pf^-1 e > tol, the component is marked for splitting, where
          Pf is the linearly-mapped covariance of the considered component


    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component


    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        # compute measurement partial derivative tensor
        mpdt = np.array(pdt_func(*m))
        cov = p.P[i]
        scaled_sixth_moment_unsym = np.einsum(
            "ab,cd,ef->abcdef", cov, cov, cov)
        # proportional to the 6th central moment
        moment = symmetrize_tensor(scaled_sixth_moment_unsym)
        mat = np.einsum("abcdef,iab,icd->ef", moment, mpdt, mpdt)
        eigs = np.linalg.eigh(mat)
        if p.w[i] * eigs.eigenvalues[-1] > tol:
            split_mask[i] = True
            split_dir[i] = eigs.eigenvectors[:, -1]
    return split_mask, split_dir


def id_wsasos(p, pdt_func, jacobian_func, tol, single_fn=False):
    """identify components and split directions based on scaled nonlinear stretching

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    pdt_func : callable
        nonlinear function second-order partial derivative tensor of the form
        pdt_func(x1, x2, ..., xn)
    jacobian_func : callable
        nonlinear function Jacobian of the form jacobian_func(x1, x2, ..., xn)
    tol : float
        tolerance for splitting, given as a Mahalanobis distance. If
          e^T Pf^-1 e > tol, the component is marked for splitting, where
          Pf is the linearly-mapped covariance of the considered component
    single_fn : bool
        interpret pdt_func argument as returning the tuple (x_f, jac, pdt)
        and disregard jacobian function completely

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component


    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        if single_fn:
            fn_info = pdt_func(*m)
            G = fn_info[1]
            mpdt = fn_info[2]
        else:
            # compute measurement partial derivative matrix
            G = jacobian_func(*m)
            # compute measurement partial derivative tensor
            mpdt = np.array(pdt_func(*m))
        # compute linearly-mapped covariance
        Pf = G @ p.P[i] @ G.T
        owmpdt = np.transpose(jax.scipy.linalg.solve_triangular(np.tile(cholesky(Pf, lower=True).T, (mpdt.shape[1],mpdt.shape[2],1,1)), np.transpose(mpdt, (1,2,0)), lower=False), (2,0,1))

        cov = p.P[i]
        scaled_sixth_moment_unsym = np.einsum(
            "ab,cd,ef->abcdef", cov, cov, cov)
        # proportional to the 6th central moment
        moment = symmetrize_tensor(scaled_sixth_moment_unsym)
        mat = np.einsum("abcdef,iab,icd->ef", moment, owmpdt, owmpdt)
        eigs = np.linalg.eigh(mat)
        if p.w[i] * eigs.eigenvalues[-1] > tol:
            split_mask[i] = True
            split_dir[i] = eigs.eigenvectors[:, -1]
    return split_mask, split_dir


def id_alodt(p, g, sigma_pt_opts, tol):
    """identify components and split directions based on sigma point curvature

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    g : callable
        nonlinear function through which to propagate the sigma points
    sigma_pt_opts : SigmaPointOptions
        options for sigma point generation
    tol : float
        tolerance for splitting based on the deviation of the sigma points from
        a linear fit

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component

    Notes
    -----
    This method is referred to by the original authors as the "Adaptive Level of
    Detail" method [1].

    References
    ----------
    [1] F. Faubel and D. Klakow, “Further improvement of the adaptive level of
      detail transform: Splitting in direction of the nonlinearity,” in 2010
      18th European Signal Processing Conference, Aug. 2010, pp. 850–854.
    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    n = p.dim
    for i, m in enumerate(p.m):
        Seig = p.Seig[i]
        # Compute the unscented transform
        sigmas = SigmaPoints(m, Seig, sigma_pt_opts)

        y0 = g(sigmas.X[:, 0])
        y = np.zeros((len(y0), 2 * n + 1))
        y[:, 0] = y0

        for j in range(1, 2 * n + 1):
            y[:, j] = g(sigmas.X[:, j])

        dev_from_linear_fit = 0.5 * np.sum(
            (y[:, 1: n + 1] + y[:, n + 1:] - 2 * y[:, 0, np.newaxis]) ** 2, axis=0
        )
        max_idx = np.argmax(dev_from_linear_fit)

        if p.w[i] * np.sqrt(dev_from_linear_fit[max_idx]) > tol:
            split_mask[i] = True
            split_dir[i] = Seig[:, max_idx] / np.linalg.norm(Seig[:, max_idx])
    return split_mask, split_dir


def id_sadl(p, jacobian_func, g, sigma_pt_opts, tol):
    """identify components and split directions based on the difference in
    deterministic and statistical linearization

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    jacobian_func : callable
        function that returns the Jacobian of the nonlinear function
    g : callable
        nonlinear function through which to propagate the sigma points
    sigma_pt_opts : SigmaPointOptions
        options for sigma point generation
    tol : float
        tolerance for splitting based on the difference between the deterministic
        and statistical linearizations

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component


    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    """
    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        Px = p.P[i]
        # Compute the unscented transform
        my, Py, Dt, sigmas, _ = unscented_transform(
            m, Px, g, sigma_pt_opts=sigma_pt_opts
        )
        # compute cross covariance
        Pxz = (
            (sigmas.X.T - m).T @ np.diag(sigmas.wc) @ Dt.T
        )  # state-measurement cross-covariance
        S = p.Schol[i]
        G_statlin = np.linalg.solve(Px, Pxz).T  # statistical linearization
        G = jacobian_func(*m)  # deterministic linearization

        eigvals, eigvecs = np.linalg.eigh(
            S.T @ (G_statlin - G).T @ (G_statlin - G) @ S)
        max_idx = np.argmax(eigvals)
        if p.w[i] * np.sqrt(eigvals[max_idx]) > tol:
            split_mask[i] = True
            split_dir[i] = S @ eigvecs[:, max_idx]
    return split_mask, split_dir


def id_wussadl(p, jacobian_func, g, sigma_pt_opts, tol, deterministic_whitening=True):
    """identify components and split directions based on output-whitened
    uncertainty scaled statistical and deterministic linearization difference

    Parameters
    ----------
    p : GaussianMixture
        input Gaussian mixture to be considered for splitting
    jacobian_func : callable
        function that returns the Jacobian of the nonlinear function
    g : callable
        nonlinear function through which to propagate the sigma points
    sigma_pt_opts : SigmaPointOptions
        options for sigma point generation
    tol : float
        tolerance for splitting. If the weighted norm exceeds tol, the component
        is marked for splitting

    Returns
    -------
    split_mask : np.ndarray
        (nC,) boolean array indicating which components are marked for splitting
    split_dir : np.ndarray
        (nC, nX) array of split directions for each component


    References
    ----------
    .. [1] Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
           Informed Moment-Matching Gaussian Mixture Splitting,"
           https://arxiv.org/abs/2412.00343, 2024

    """

    split_mask = np.full(p.w.shape, False)
    split_dir = np.full(p.m.shape, np.nan)
    for i, m in enumerate(p.m):
        Px = p.P[i]
        # Compute the unscented transform
        my, Py_UT, Dt, sigmas, _ = unscented_transform(
            m, Px, g, sigma_pt_opts=sigma_pt_opts
        )
        # compute cross covariance
        Pxz = (
            (sigmas.X.T - m).T @ np.diag(sigmas.wc) @ Dt.T
        )  # state-measurement cross-covariance
        S = p.Schol[i]
        G_statlin = np.linalg.solve(Px, Pxz).T  # statistical linearization
        G = jacobian_func(*m)  # deterministic linearization

        # find square root factor of precision matrix Pf^-1 = U@U^T
        if deterministic_whitening:
            werror_mat = jax.scipy.linalg.solve_triangular(cholesky(G @ Px @ G.T, lower=True).T, (G_statlin - G) @ S, lower=False)
            # print("Using deterministic whitening")
        else:
            werror_mat = jax.scipy.linalg.solve_triangular(cholesky(Py_UT, lower=True).T, (G_statlin - G) @ S, lower=False)
            # print("Using UT-based whitening")

        # compute right singular value corresponding to the largest singular value:
        U, Svs, Vh = np.linalg.svd(werror_mat)
        if p.w[i] * Svs[0] > tol:
            split_mask[i] = True
            # take the maximal right singular vector of GS and the inverse coordinate transform
            split_dir[i] = S @ Vh[0]
        # print(split_dir)
        # print("WUSSADL")
    return split_mask, split_dir
