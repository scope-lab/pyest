import os
from .. utils import protected_cached_property

import cvxopt
import jax
import numpy as np
import pandas as pd
from numpy import pi, sqrt
from numpy.random import rand, randn
from scipy.stats._multivariate import _LOG_2PI, _PSD, _squeeze_output
from scipy.stats import Covariance, _covariance
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
from pyest.linalg import triangularize, cholesky_from_sqrt_precision

from huggingface_hub import hf_hub_download

__all__ = [
    'GaussianMixture',
    'GaussSplitOptions',
    'Covariance',
    'multivariate_normal',
    'fit_1d_uniform',
    'gm_fit_1d',
    'equal_weights',
    'eval_gmpdf',
    'eval_gmpdfchol',
    'eval_mvnpdf',
    'eval_mvnpdfchol',
    'v_eval_mvnpdf',
    'v_eval_mvnpdfchol',
    'integral_gauss_product',
    'integral_gauss_product_chol',
    'marginal_2d',
    'marginal_nd',
    'comp_bounds',
    'bounds',
    'gm_pdf_2d',
    'sigma_contour',
    'distribute_mean_centers',
    'optimal_homoscedastic_std',
]

sigmavals = None


def load_1d_unifit_sigmavals():
    file_in_repo = "l2optimalunifitsiglib.parquet"
    repo_id = "klegrand/l2optimalunifitsiglib"
    local_file_name = ".data/l2optimalunifitsiglib.parquet"
    # Check if file exists locally
    if not os.path.exists(local_file_name):
        print(f"Local file '{local_file_name}' not found. Downloading from Hugging Face...")
        hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=file_in_repo,
            local_dir=".data",
            local_dir_use_symlinks=False
        )
        print("Download complete!")
    else:
        print(f"Local file '{local_file_name}' already exists.")
    return np.array(pd.read_parquet(local_file_name))


def distribute_mean_centers(lb, ub, L):
    """ distribute L mean centers between lb and ub"""

    dmi = 1.0/(L + 1.0)
    mi = np.cumsum(dmi*np.ones([L, 1]))
    return lb + (ub - lb)*mi


def __check_width(width):
    """asserts that width is positive"""
    assert width > 0


def __check_num_comp(L):
    assert isinstance(L, int)
    assert L >= 1
    assert L <= 15000


def find_gm_res(stdmax, width=1):
    """find necessary number of components needed to meet comp. std max"""
    global sigmavals  # noqa: F824
    __check_width(width)
    if sigmavals is None:
        sigmavals = load_1d_unifit_sigmavals()

    L = int(np.where(sigmavals*width <= stdmax)[0][0])

    return L


def optimal_homoscedastic_std(L, width=1):
    """generates optimal std for an L component GM under homoscedasticity constraint"""
    global sigmavals  # noqa: F824
    __check_width(width)
    __check_num_comp(L)

    if sigmavals is None:
        sigmavals = load_1d_unifit_sigmavals()
    return width*sigmavals[L-1]


def fit_1d_uniform(lb, ub, L=100):
    """approximates 1D uniform distribution with GM

    Parameters
    ----------
    lb: float
        lower bound of uniform distribution
    ub: float
        upper bound of uniform distribution
    L : int, optional
        number of components to use in approximation. Default is 100

    Returns
    -------
    GaussianMixture
    """

    sx = optimal_homoscedastic_std(L, width=ub-lb)
    mk = distribute_mean_centers(lb, ub, L).reshape((L, 1))
    wk = equal_weights(L)
    Pk = np.tile(sx*sx, (L, 1, 1))
    return GaussianMixture(wk, mk, Pk)


def fit_weights_to_f_1d(xvals, fvals, m, std):
    """optimize weights of GM components m,std to fit xvals,fvals
    assumes m values are evenly-spaced
    assumes homoscedasticity (equal covariances/std)
    """
    L = len(m)
    d = 1.0/(sqrt(2.0*pi)*std)

    X, M = np.meshgrid(xvals, m)
    H = d*np.exp(-0.5*((X.T - M.T)/std)**2)

    """
    for i in range(len(marg)):
        rhoi = xvals[i]
        for j in range(Lx):
            H[i,j] = d*exp(-0.5*((rhoi - mx[j])/sx)**2)
    """
    Hqp = cvxopt.matrix(H.T @ H)
    fqp = cvxopt.matrix((-H.T @ fvals).reshape((H.shape[1], 1)))
    Aqp = cvxopt.matrix(-np.eye(L))
    bqp = cvxopt.matrix(np.zeros(L).reshape((L, 1)))
    Aeqqp = cvxopt.matrix(np.ones(L).reshape((1, L)))
    beqqp = cvxopt.matrix(1.0)
    cs = cvxopt.solvers
    cs.options['show_progress'] = False
    cs.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
    qp_sol = cs.qp(Hqp, fqp, Aqp, bqp, Aeqqp, beqqp)
    w = np.array(qp_sol['x']).squeeze()
    return w


def gm_fit_1d(xvals, pvals, support=None, stdmax=None, L=None):
    global sigmavals  # noqa: F824

    if support is None:
        support = (min(xvals), max(xvals))

    # TODO: check that only either stdmax or L is defined
    lb, ub = support
    width = ub - lb

    L = find_gm_res(stdmax, width)
    mx = distribute_mean_centers(lb, ub, L).reshape((L, 1))
    sx = optimal_homoscedastic_std(L, width)
    wx = fit_weights_to_f_1d(xvals, pvals, mx, sx)

    Sk = np.tile(sx, (1, 1, L))
    return GaussianMixture(wx, mx, Sk, 'cholesky')


def equal_weights(L):
    """
    generate L equal weights that sum to 1

    Parameters
    ----------
    L : int
        number of weights
    Returns
    -------
    ndarray
    """
    __check_num_comp(L)
    return np.tile(1.0/L, L)


def eval_gmpdf(x, w, m, P):
    """
    evaluates Gaussian mixture at points x

    Required
    --------
    x : array_like
        (n_samp, vect_length) points at which to evaluate the GM
    w : array_like
        (nC,) Gaussian mixture weights
    m : array_like
        (nC, vect_length) Gaussian mixture means
    P : array_like
        (nC, vect_length, vect_length) Gaussian mixture covariances

    Returns
    -------
    ndarray
        (n_samp,) Gaussian mixture evaluated at points x

    """
    x = np.array(x)
    if x.shape[-1] != m.shape[-1]:
        raise ValueError('The last dimensions of x and m must match.')

    q = (w[:, np.newaxis]*np.array(v_eval_mvnpdf(x, m, P))).sum(axis=0)
    return q.squeeze()


def eval_gmpdfchol(x, w, m, S):
    """
    evaluates Gaussian mixture at points x

    Required
    --------
    x : array_like
        (n_samp, vect_length) points at which to evaluate the GM
    w : array_like
        (nC,) Gaussian mixture weights
    m : array_like
        (nC, vect_length) Gaussian mixture means
    S : array_like
        (nC, vect_length, vect_length) Gaussian mixture lower-triangular
        Cholesky factors

    Returns
    -------
    ndarray
        (n_samp,) Gaussian mixture evaluated at points x
    """
    x = np.array(x)
    if x.shape[-1] != m.shape[-1]:
        raise ValueError('The last dimensions of x and m must match.')

    q = (w[:, np.newaxis]*np.array(v_eval_mvnpdfchol(x, m, S))).sum(axis=0)
    return q.squeeze()


def optimized_eval_gmpdf(x, w, m, Schol, log_pdet):
    """
    Evaluates Gaussian mixture at points x

    Required
    --------
    x : array_like
        (n_samp, vect_length) points at which to evaluate the GM
    w : array_like
        (nC,) Gaussian mixture weights
    m : array_like
        (nC, vect_length) Gaussian mixture means
    Schol : array_like
        (num_comp, vect_length, vect_length) Gaussian mixture covariance
        lower-triangular Cholesky factors
    log_pdet : array_like
        (nC,) log of the determinant of the covariance matrix


    Returns
    -------
    ndarray
        (n_samp,) Gaussian mixture evaluated at points x
    """
    x = np.asarray(x)
    w = np.asarray(w)
    m = np.asarray(m)

    if x.shape[1] != m.shape[1]:
        raise ValueError('The last dimensions of x and m must match.')

    vect_length = x.shape[1]

    # Vectorized computation of multivariate normal PDFs
    diff = x[:, np.newaxis, :] - m[np.newaxis, :, :]  # (n_eval, n_comp, vect_length)

    # y = np.vectorize(
    #     lambda S,x:solve_triangular(S,x.T,lower=True).T,
    #     signature='(n,n),(n)->(n)')(Schol[np.newaxis,:,:], diff)

    y = jax.scipy.linalg.solve_triangular(np.tile(Schol, (x.shape[0], 1, 1, 1)), diff, lower=True)
    maha_dist = np.sum(y**2, axis=2)

    # Compute log-likelihoods
    log_likelihood = -0.5 * (vect_length * _LOG_2PI +
                             log_pdet +
                             maha_dist)

    # Compute weighted probabilities
    q = np.sum(w[np.newaxis, :] * np.exp(log_likelihood), axis=1)

    return q.squeeze()


def eval_logmvnpdf(x, mean, prec_U, log_det_cov, rank):
    """
    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the log of the probability
        density function
    mean : ndarray
        Mean of the distribution
    prec_U : ndarray
        A decomposition such that np.dot(prec_U, prec_U.T)
        is the precision matrix, i.e. inverse of the covariance matrix.
    log_det_cov : float
        Logarithm of the determinant of the covariance matrix
    rank : int
        Rank of the covariance matrix.

    Notes
    -----
    As this function does no argument checking, it should not be
    called directly; use 'logpdf' instead.

    """
    dev = x - mean
    maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
    return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)


def eval_mvnpdf(x, m, P, allow_singular=False):
    """
    Multivariate normal probability density function.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_mvn_doc_default_callparams)s

    Returns
    -------
    pdf : ndarray or scalar
        Probability density function evaluated at `x`

    Notes
    -----
    %(_mvn_doc_callparams_note)s

    """
    psd = _PSD(P, allow_singular=allow_singular)
    out = np.exp(eval_logmvnpdf(x, m, psd.U, psd.log_pdet, psd.rank))
    return _squeeze_output(out)


def eval_mvnpdfchol(x, m, S, allow_singular=False):
    """
    Multivariate normal probability density function.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    m : ndarray
        (nx,) mean vector
    S : ndarray
        (nx, nx) lower-triangular Cholesky factor of covariance matrix

    Returns
    -------
    pdf : ndarray or scalar
        Probability density function evaluated at `x`

    """
    nx = m.shape[-1]
    log_pdet = 2*np.log(np.diag(S)).sum(axis=-1)
    residuals = x - m[np.newaxis, :]
    mahalanobis_dist_sqrd = np.array([np.linalg.norm(np.linalg.solve(S, res))**2 for res in residuals])
    out = np.exp(-0.5*(nx*_LOG_2PI + log_pdet + mahalanobis_dist_sqrd))
    return _squeeze_output(out)


def v_eval_mvnpdf(x, m, P):
    return [np.atleast_1d(eval_mvnpdf(x, mi, Pi)) for mi, Pi in zip(m, P)]


def v_eval_mvnpdfchol(x, m, S):
    return [np.atleast_1d(eval_mvnpdfchol(x, mi, Si)) for mi, Si in zip(m, S)]


def eig_sqrt_factor(P):
    """ compute eigenvector based sqrt factor of covariance

    Parameters
    ----------
    P: ndarray
      (n,n) covariance matrix

    Returns
    -------
    ndarray
      eigenvector based sqrt factor
    """
    eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(P))
    return eigvecs@np.diag(eigvals**0.5)


def integral_gauss_product_chol(m1, S1, m2, S2):
    ''' compute integral of product of two Gaussians

    Required
    --------
    m1: np.ndarray
        mean of Gaussian 1
    S1: np.ndarray
        Cholesky factor of covariance of Gaussian 1
    m2: np.ndarray
        mean of Gaussian 2
    S2: np.ndarray
        Cholesky factor of covariance of Gaussian 2

    Returns
    -------
    integral: float
        integral of the product of the two Gaussians
    '''
    assert(all(np.diag(S1) >= 0))
    assert(all(np.diag(S2) >= 0))
    # compute the product of the two Gaussians
    S1pS2 = triangularize(np.hstack((S1, S2)))
    return eval_mvnpdfchol(m1, m2, S1pS2)


def integral_gauss_product(m1, P1, m2, P2, allow_singular=False):
    ''' compute integral of product of two Gaussians

    Required
    --------
    m1: np.ndarray
        mean of Gaussian 1
    P1: np.ndarray
        covariance of Gaussian 1
    m2: np.ndarray
        mean of Gaussian 2
    P2: np.ndarray
        covariance of Gaussian 2

    Returns
    -------
    integral: float
        integral of the product of the two Gaussians
    '''

    return multivariate_normal.pdf(m1, m2, P1 + P2, allow_singular=allow_singular)


def marginal_2d(m, P, dimensions=[0, 1]):
    """ compute 2D marginal of GM"""
    return marginal_nd(m, P, dimensions)


def marginal_nd(m, P, dimensions):
    """ returns nd-mariginal GM over specified dimensions """
    dimensions = np.array(dimensions)
    mnd = m[:, dimensions]
    Pnd = P[:, dimensions[:, np.newaxis], dimensions]
    return mnd, Pnd


def comp_bounds(m, P, sigma_mult=3):
    """ find lower and upper sigma bounds for each component """
    diag_ind = np.arange(m.shape[-1])
    dev = sigma_mult*np.sqrt(P[:, diag_ind, diag_ind])
    lb = m - dev
    ub = m + dev
    return lb, ub


def bounds(m, P, sigma_mult=3):
    """ find bounds of GM """
    lb, ub = comp_bounds(m, P, sigma_mult)
    return np.min(lb, axis=0), np.max(ub, axis=0)


def print_3d_mat(A):
    for idx in range(A.shape[2]):
        print("[:, :, {}] = \n{}\n".format(idx, str(A[:, :, idx])))


def gm_pdf_2d(w, m, P, dimensions=(0, 1), res=100, xbnd=None, ybnd=None):
    """ evaluate GM pdf in two dimensions

    Required
    --------
    w: ndarray
        (nC,) weights
    m: ndarray
        (nC, nx) means
    P: ndarray
        (nC, nx, nx) covariances

    Optional
    --------
    dimensions: tuple
        dimensions to evaluate pdf over. Defaults to (0, 1)
    res: int
        resolution of the grid. Defaults to 100
    xbnd: tuple
        x-axis bounds. Determined automatically if not provided
    ybnd: tuple
        y-axis bounds. Determined automatically if not provided

    Returns
    -------
    ndarray
        (res, res) pdf values
    ndarray
        (res, res) x values
    ndarray
        (res, res) y values
    """
    w = np.atleast_1d(w)
    m = np.atleast_2d(m)
    if P.ndim == 2:
        # make P a 3D array
        P = np.expand_dims(P, axis=0)

    m2d, P2d = marginal_2d(m, P, dimensions)

    if not xbnd or not ybnd:
        lb, rb = bounds(m2d, P2d)

    if not xbnd:
        xbnd = (lb[0], rb[0])

    if not ybnd:
        ybnd = (lb[1], rb[1])

    xvec = np.linspace(*xbnd, res)
    yvec = np.linspace(*ybnd, res)
    X, Y = np.meshgrid(xvec, yvec)

    p = 0.0
    for wi, mi, Pi in zip(w, m2d, P2d):
        S, U = np.linalg.eigh(Pi)  # eigenvalues, eigenvectors
        V = U.T
        L = 1.0/S
        D = 1.0/np.sqrt(np.prod(2*np.pi*S))  # normal distribution denominator
        N = V @ mi

        dx = V[0, 0]*X + V[0, 1]*Y - N[0]
        dy = V[1, 0]*X + V[1, 1]*Y - N[1]
        pe = np.exp(-0.5*(dx*dx*L[0] + dy*dy*L[1]))
        p = p + wi*D*pe

    return p, X, Y


def sigma_contour(m, P, sig_mul=1, num_pts=100):
    """ compute sigma contours of a 2D Gaussian

    Parameters
    ----------
    m: ndarray
        (2,) mean vector
    P: ndarray
        (2, 2) covariance matrix
    sig_mul: float, optional
        sigma multiplier factor. For example, sig_mul=2 will return points
        corresponding to the 2-sigma contour. Defaults to sig_mul=1
    num_pts: int, optional
        number of points to compute. Defaults to num_pts=100

    Returns
    -------
    ndarray
        (num_pts, 2) array of points along the contour
    """

    theta = np.linspace(0, 2*np.pi, num_pts)
    ze = np.vstack((np.cos(theta), np.sin(theta)))
    S = cholesky(P, lower=True)  # cholesky sqrt factor
    return m + np.dot(sig_mul*S, ze).T


class GaussianMixture(object):

    def __init__(self, w, m, cov, cov_type='full', Seig=None):
        """
        w is a 1d array
        m is a 2d array (nC,nx)
        cov is a 3d array (nC,nx,nx)
        """
        self.set_w(w)
        self.set_m(m)
        self._Seig = Seig  # directly write Seig here as to not overwrite P
        self._set_cov(cov, cov_type)

        # check that equal numbers of weights, means, and covariances are provided
        if len(self.w) != len(self.m) or len(self.w) != len(self._cov):
            raise ValueError("Number of weights, means, and covariances must match.")

        self.set_msize(self._cov[0].covariance.shape[-1])

    def __getitem__(self, ind):
        return self.get_comp(ind)

    def _not_allowed(self):
        raise RuntimeError("Can not set size directly.")

    def __add__(self, gm2):
        if isinstance(gm2, int) and gm2 == 0:  # Explicitly check for integer 0
            return self

        if not isinstance(gm2, GaussianMixture):
            raise TypeError("Can only add GaussianMixture objects or 0.")

        w = np.hstack((self.w, gm2.w))
        m = np.vstack((self.m, gm2.m))
        # TODO: use cov objects to add covariances
        P = np.vstack((self.P, gm2.P))
        return GaussianMixture(w, m, P)

    def __radd__(self, gm2):
        return self.__add__(gm2)

    def __mul__(self, a):
        return GaussianMixture(a*self.w, self.m, self._cov)

    def __rmul__(self, a):
        return self.__mul__(a)

    def __call__(self, x):
        return self.eval(x)

    def __len__(self):
        return len(self.w)

    def __eq__(self, oth):
        if ~np.all(self.w == oth.w):
            return False
        elif ~np.all(self.m == oth.m):
            return False
        elif ~np.all(self.P == oth.P):
            return False

        return True

    def _compute_eig_sqrt_factors(self):
        return np.array([eig_sqrt_factor(cov.covariance) for cov in self._cov])

    def get_size(self):
        return len(self.w)

    # def _set_size(self):
    #    self._size = len(self._w)

    def get_w(self):
        return self._w

    def set_w(self, w):
        self._w = np.atleast_1d(w)

    def get_m(self):
        return self._m

    def set_m(self, m):
        self._m = np.atleast_2d(m)

    def get_P(self, ind=None):
        if ind is not None:
            if np.isscalar(ind):
                return self._cov[ind].covariance
            else:
                return np.array([P.covariance for P in np.array(self._cov)[ind]])
        return np.array([P.covariance for P in self._cov])

    def set_P(self, P):
        self._set_cov(P, 'full')

    def _set_cov(self, cov, cov_type):
        if isinstance(np.atleast_1d(cov)[0], Covariance):
            self._cov = np.atleast_1d(cov)
            return
        if cov_type == 'full' or cov_type == 'cholesky':
            cov = np.atleast_2d(cov)
            if cov.ndim == 2:
                covarr = cov[np.newaxis, :, :]
            else:
                covarr = cov
        if cov_type == 'full':
            # compute PSD objects
            psd = [_PSD(P) for P in covarr]
            self._cov = [_covariance.CovViaPSD(psdi) for psdi in psd]
            return

        elif cov_type == 'cholesky':
            self._cov = [Covariance.from_cholesky(Si) for Si in covarr]
            return

        elif cov_type == 'eigendecomposition':
            self._cov = [Covariance.from_eigendecomposition(ed) for ed in covarr]
            return
        else:
            raise ValueError('cov_type must be one of "full", "cholesky", or "eigendecomposition"')

    def set_Seig(self, S):
        S = np.atleast_2d(S)
        if S.ndim == 2:
            self._Seig = S[np.newaxis, :, :]
        else:
            self._Seig = S
        # updated covariance values
        # TODO: rather than compute Si@Si.T, instantiate covariance object
        # using the eigendecomposition
        self._set_cov(np.array([Si@Si.T for Si in self._Seig]), 'full')

    @protected_cached_property
    def Schol(self):
        '''
        return the covariance matrix lower cholesky square root factor
        '''
        if isinstance(self._cov[0], _covariance.CovViaCholesky):
            return np.array([cov._factor for cov in self._cov])
        elif isinstance(self._cov[0], _covariance.CovViaPrecision):
            return np.array([cholesky_from_sqrt_precision(cov._chol_P) for cov in self._cov])
        else:
            return np.array([cholesky(cov.covariance, lower=True) for cov in self._cov])

    @protected_cached_property
    def Seig(self):
        '''
        return the covariance matrix eigenvector square root factor
        '''
        return self._compute_eig_sqrt_factors()

    @protected_cached_property
    def _PSD(self):
        if isinstance(self._cov[0], _covariance.CovViaPSD):
            return [cov._psd for cov in self._cov]
        return [_PSD(P) for P in self.P]

    @protected_cached_property
    def log_pdet(self):
        return np.array([cov._log_pdet for cov in self._cov])

    @protected_cached_property
    def prec_sqrt(self):
        """ compute  precision sqrt factor

        Returns
        -------

        ndarray
            (nC,nx,nx) array of precision matrix square root factors, i.e.
            A decomposition such that np.dot(prec_U, prec_U.T)
            is the precision matrix, i.e. inverse of the covariance matrix."""
        if isinstance(self._cov[0], _covariance.CovViaCholesky):
            return np.array([np.linalg.inv(cov._factor.T) for cov in self._cov])
        if isinstance(self._cov[0], _covariance.CovViaPrecision):
            return np.array([cov._chol_P for cov in self._cov])
        if isinstance(self._cov[0], _covariance.CovViaPSD):
            return np.array([cov._LP for cov in self._cov])
        return np.array([psd.U for psd in self._PSD])

    def get_msize(self):
        """get vector length of mean"""
        return self._msize

    def set_msize(self, msize):
        """set vector length of mean"""
        self._msize = msize

    def get_comp(self, ind):
        return self.w[ind], self.m[ind], self.get_P(ind)

    def eval(self, x):
        """ evaluate gm at n_eval points stored in numpy array x

        Required
        --------
        x: ndarray
            (n_eval, nx) array of points at which to evaluate the GM

        Returns
        -------
        ndarray
            (n_eval,) array of GM evaluations at points x

        """

        x = np.atleast_2d(x)
        if x.shape[-1] != self.m.shape[-1]:
            raise ValueError(
                'The last dimension of x must match the state dimension.')
        return _squeeze_output(optimized_eval_gmpdf(x, self.w, self.m, self.Schol, self.log_pdet))

    def mean(self):
        """ return the mean of the Gaussian mixture """
        return np.sum(self.w[:, np.newaxis] * self.m, axis=0)

    def cov(self):
        """ return the conditional covariance """
        wsum = np.sum(self.w)
        mean = self.mean()
        return np.sum([
            w/wsum*(P + np.outer(m, m)) for w, m, P in self
        ], axis=0) - np.outer(mean, mean)

    def cdf(self, x, allow_singular=False):
        """ evaluate GM CDF at points x"""
        x = np.atleast_2d(x)
        if x.shape[-1] != self.m.shape[-1]:
            raise ValueError(
                'The last dimension of x must match the state dimension.')
        return np.sum([self.w[i]*multivariate_normal.cdf(x, self.m[i], self._cov[i], allow_singular=allow_singular) for i in range(len(self))], axis=0)

    def comp_bounds(self, sigma_mult=3):
        return comp_bounds(self.m, self.P, sigma_mult)

    def marginal_2d(self, dimensions=(0, 1)):
        """ return 2D marginalized GM """
        return GaussianMixture(self.w, *marginal_2d(self.m, self.P, dimensions))

    def marginal_nd(self, dimensions):
        """ return nD marginalization of GM over specified dimensions """
        return GaussianMixture(self.w, *marginal_nd(self.m, self.P, dimensions))

    def pdf_2d(self,  dimensions=(0, 1), res=100, xbnd=None, ybnd=None):
        """ evaluate GM pdf in two dimensions

        Optional
        --------
        dimensions: tuple
            dimensions to evaluate pdf over. Defaults to (0, 1)
        res: int
            resolution of the grid. Defaults to 100
        xbnd: tuple
            x-axis bounds. Determined automatically if not provided
        ybnd: tuple
            y-axis bounds. Determined automatically if not provided

        Returns
        -------
        ndarray
            (res, res) pdf values
        ndarray
            (res, res) x values
        ndarray
            (res, res) y values
        """
        return gm_pdf_2d(self.w, self.m, self.P, dimensions, res, xbnd, ybnd)

    def pop(self, idx):
        """ remove and return component by index """
        w, m, P = self[idx]
        self._w = np.delete(self._w, idx, 0)
        self._m = np.delete(self._m, idx, 0)
        self._cov = np.delete(self._cov, idx, 0)
        return w, m, P

    def rvs(self, size=None):
        """ Gaussian Mixture generated random variates

        Parameters
        ----------
        size : int, optional
            Defining number of random variates (Default is 1)

        Returns
        -------
        ndarray
            Random variates

        Notes
        -----
        If N is not specified, a single (possibly-vector) random variate is
        returned. If N is specified, an ndarray of random variates is
        returned, even for N=1.
        """
        sort_idx = np.argsort(-self._w)
        # sort and normalize weight
        w_sorted = self._w[sort_idx]/np.sum(self._w)
        cum_weights = np.cumsum(w_sorted)
        S_sorted = self.Schol[sort_idx]
        if size is None:
            comp_idx = np.where(rand() < cum_weights)[0][0]
            return self.m[sort_idx][comp_idx] + S_sorted[comp_idx]@randn(self._msize)
        else:
            comp_idx = [np.where(rand() < cum_weights)[0][0] for i in range(size)]
            return _squeeze_output(np.array([self.m[sort_idx][c] + S_sorted[c]@randn(self._msize) for c in comp_idx]))

    size = property(get_size, _not_allowed)
    w = property(get_w, set_w)
    m = property(get_m, set_m)
    P = property(get_P, set_P)
    dim = property(get_msize, set_msize)


class GaussSplitOptions(object):

    def __init__(
        self, L=3, lam=1e-3, recurse_depth=np.inf,
        min_weight=1e-3, state_idxs=(0, 1),
        spectral_direction_only=False, variance_preserving=True
    ):
        """ Options for FoV splitting

        Parameters
        ----------
        L: int, optional
            number of components to spit Gaussian into. Default is 3
        lam: float, optional
            Gauss split optimization regularization parameter. Lower values
            result in less component spread and larger component covariances.
            This value should be decreased as L increases. Default is 1e-3
        recurse_depth: int, optional
            maximum recursion depth. Default is inf
        min_weight: float, optional
            smallest component weight that is considered for splitting. Default
            is 1e-3
        state_idxs: tuple
            optional (2,) indices of state vector corresponding to the FoV space. Default is (0,1)
        spectral_direction_only: bool, optional
            if True, only use spectral direction for splitting. Default is False
        variance_preserving: bool, optional
            if True, preserve variance in splitting. Default is True

        """

        self.L = L
        self.lam = lam
        self.recurse_depth = recurse_depth
        self.min_weight = min_weight
        self.state_idxs = np.array(state_idxs)
        self.spectral_direction_only = spectral_direction_only
        self.variance_preserving = variance_preserving
