import numpy as np
from scipy.linalg import cholesky, qr

from pyest.linalg import choldowndate, make_chol_diag_positive
from pyest.utils import BadCholeskyFactor


class SigmaPointOptions:
    """ Options for sigma points

    Attributes
    ----------
    alpha : float
        unscented transform parameter, normally 1e-3. Controls the
        size of the sigma point distributions. Higher alpha
        pre-scales the sigma points farther from the center sigma
        point.
    beta : double
        unscented transform parameters, beta=2 is optimal for
        Gaussian prior. Can be used to control the error in
        kurtosis
    kappa : double
        unscented transform parameter (normally set to 0). It is
        not recommended to use anything but 0 unless you know what
        you're doing as you can end up with non-positive definite
        covariance matrices

    Methods
    -------
    lambda_(nx)
        Compute the lambda parameter for the unscented transform

    """
    def __init__(self, alpha=1e-3, beta=2, kappa=0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def lambda_(self, nx):
        lam = self.alpha**2 * (nx + self.kappa) - nx  # UT lambda parameter
        return lam

def unscented_transform(m, cov, g, sigma_pt_opts, cov_type='full', add_noise=None, residual_fun=None, mean_fun=None, g_args=()):
    """
    Computes the unscented transform of an input vector and covariance.

    Parameters
    ----------
    m : numpy.ndarray
        (nx,1) mean of prior.
    cov : numpy.ndarray
        (nx,nx) covariance of prior, represented as a full covariance matrix or
        as a lower-triangular cholesky square-root factor. Covariance type, if
        not 'full', should be specified using 'cov_type' parameter.
    g : function
        Nonlinear function of form g(x, *g_args). g_args contains all relevant
        parameters, including possibly the times for propagation in the case
        where g represents a time-update.
    sigma_pt_opts : object
        Options for sigma point.
    cov_type : str, optional
        Type of covariance supplied. Valid options are 'full' and 'cholesky'.
    add_noise : numpy.ndarray, optional
        Covariance of additive noise. Default is no noise. Can be full
        covariance matrix or cholesky factor.
    residual_fun : function, optional
        Function to use to compute residuals in cases where simple subtraction
        can not be used (e.g., angles near 2pi).
    mean_fun : function, optional
        Function to compute the mean of the transformed sigma points in the
        case that mean is not a simple Euclidean mean. For example, if using
        angles, quaternions, etc.
    g_args : tuple, optional
        Extra arguments to be passed to function g.

    Returns
    -------
    mt : numpy.ndarray
        (ny,1) transformed mean.
    cov_t : numpy.ndarray
        (ny,ny) transformed covariance. Form of covariance (full, cholesky)
        will match specified input type.
    Dt : numpy.ndarray
        (ny,2*nx+1) transformed deviations from mean.
    sigmas : numpy.ndarray
        (nx,2*nx+1) set of prior sigma points.
    wc : numpy.ndarray
        Covariance weights.
    y : numpy.ndarray
        (ny,2*nx+1) set of posterior sigma points.
    """
    n = len(m)
    if cov_type == 'full':
        S = cholesky(cov, lower=True)
    elif cov_type == 'cholesky':
        S = cov
        if not np.allclose(S, np.tril(S)):
            raise BadCholeskyFactor('Provided cholesky factor is not lower-triangular')
        #assert np.allclose(S, np.tril(S)), 'Provided cholesky factor is not lower-triangular'
    else:
        raise ValueError('Unrecognized covariance type.')

    sigmas = SigmaPoints(m, S, sigma_pt_opts)

    y0 = g(sigmas.X[:, 0], *g_args)
    y = np.zeros((len(y0), 2*n+1))
    y[:, 0] = y0

    for i in range(1, 2*n+1):
        y[:, i] = g(sigmas.X[:, i], *g_args)

    if mean_fun is None:
        mt = np.sum(y * sigmas.wm, axis=1)
    else:
        mt = mean_fun(y, sigmas.wm)

    if residual_fun is None:
        Dt = y - mt[:, np.newaxis]
    else:
        Dt = residual_fun(y, mt)

    if cov_type == 'full':
        cov_t = Dt @ np.diag(sigmas.wc) @ Dt.T
        if add_noise is not None:
            cov_t += add_noise
    elif cov_type == 'cholesky':
        if add_noise is not None:
            _, cov_t = qr(np.hstack([np.tile(np.sqrt(sigmas.wc[1:]),(len(y0),1)) * Dt[:, 1:].T, add_noise]).T, mode='economic')
        else:
            _, cov_t = qr((np.tile(np.sqrt(sigmas.wc[1:]),(len(y0),1)) * Dt[:, 1:]).T, mode='economic')
        cov_t = make_chol_diag_positive(cov_t.T).T
        cov_t = choldowndate(cov_t,np.sqrt(-sigmas.wc[0]) * Dt[:, 0]).T
    else:
        raise ValueError('Unrecognized covariance type.')

    return mt, cov_t, Dt, sigmas, y

def angular_residual(y, m):
    ''' Compute angular residuals between y and m

    Required
    --------
    y : numpy.ndarray
        array of angles in radians
    m : numpy.ndarray
        possibly-singleton array of angles in radians

    Returns
    -------
    numpy.ndarray
        array of angular residuals(y-m) in radians over [-pi, pi]
    '''
    res = y - m
    res[res > np.pi] -= 2*np.pi
    res[res < -np.pi] += 2*np.pi
    return res

def optimize_angle(angles, weights, max_iterations=100, tolerance=1e-6):

    theta_hat = angles[0]
    for _ in range(max_iterations):
        residuals = angular_residual(angles, theta_hat)
        jacobian = -np.ones_like(angles)

        numerator = np.sum(weights * jacobian * residuals)
        denominator = np.sum(weights * jacobian**2)

        delta = numerator / denominator
        theta_hat_new = (theta_hat - delta) % (2*np.pi)

        if np.abs(delta) < tolerance:
            break

        theta_hat = theta_hat_new

    if theta_hat > np.pi:
        theta_hat = theta_hat - 2*np.pi
    elif theta_hat < -np.pi:
        theta_hat = theta_hat + 2*np.pi
    return theta_hat

class SigmaPoints(object):
    """
    Generate sigma points for the Unscented Kalman Filter.

    Parameters
    ----------
    x : numpy.ndarray
        Mean of the distribution.
    S : numpy.ndarray
        Covariance Cholesky square root factor

    Returns
    -------
    numpy.ndarray
        Sigma points.

    Notes
    -----
    The function follows the 2n+1 sigma point rule, where n is the dimension of the mean vector x.
    """
    def __init__(self, m, S, opts):
        n = len(m)
        lambda_ = opts.lambda_(n)
        gamma_squared = lambda_ + n
        # composite scaling factor for scaled unscented transform
        gamma = np.sqrt(gamma_squared)
        self.wm = np.hstack([lambda_/gamma_squared, np.repeat(1/(2*gamma_squared), 2*n)])
        self.wc = self.wm.copy()
        self.wc[0] += 1 - opts.alpha**2 + opts.beta
        delta = gamma*S
        y = np.tile(m, (n, 1)).T
        self.X = np.hstack([m[:, np.newaxis], y + delta, y - delta])