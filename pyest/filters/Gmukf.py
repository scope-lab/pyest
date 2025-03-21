import numpy as np
import pyest.gm as pygm
from pyest.filters import GaussianMixturePredict, GaussianMixtureUpdate
from pyest.filters.UnscentedKalmanFilter import UkfPredict, UkfUpdate


class GmukfPredict(UkfPredict, GaussianMixturePredict):
    """ Gaussian mixture unscented Kalman filter prediction

    Parameters
    ----------
    f : callable
        system function of the form :math:`x_{k+1} = f(x_{k}, t_{k}, t_{k+1})`
    Q : ndarray or callable
        process noise covariance matrix of the form Q(tkm1, tkm). If provided
        an ndarray instead, Q will automatically be recast as a callable.
    M : (optional) ndarray or callable
        process noise mapping matrix of the form M(tkm1, tkm). If provided
        an ndarray instead, M will automatically be recast as a callable.
    alpha : (optional) float
        alpha controls the “size” of the sigma-point distribution and should
        ideally be a small number to avoid sampling non-local effects when the
        nonlinearities are strong. Here “locality” is defined in terms on the
        probabilistic spread of x as summarized by its covariance.
    beta : (optional) float
        beta is a non-negative weighting term which can be used to incorporate
        knowledge of the higher order moments of the distribution. For a Gaussian
        prior the optimal choice is beta=2. This parameter can also be used to
        control the error in the kurtosis which affects the ’heaviness’ of the
        tails of the posterior distribution.
    kappa : (optional) float
        Choose kappa >= 0 to guarantee positive semi-definiteness of the
        covariance matrix. The specific value of kappa is not critical
        though, so a good default choice is kappa =0.

    """
    def __init__(self, f, Q, M=None, sigma_pt_opts=None, cov_type='full'):
        super().__init__(f, Q, M=M, sigma_pt_opts=sigma_pt_opts, cov_type=cov_type)

    def predict(self, tv, pkm1, f_args=()):
        """ perform GMUKF prediction step

        Parameters
        ----------
        tv : tuple
            start and stop times given as tv=(tkm1, tk)
        pkm1 : GaussianMixture
            posterior GM at time k-1
        f_args : (optional) tuple
            tuple of arguments to be additionally passed to the system function

        Returns
        -------
        GaussianMixture
            prior GM at time k
        """
        # weights are constant over prediction
        wkm = pkm1.w
        mkm = np.empty_like(pkm1.m, dtype=float)
        Pkm = np.empty_like(pkm1.P, dtype=float)
        # perform time-update on each component
        for i,(_, mkm1, Pkm1) in enumerate(pkm1):
            mkm[i], Pkm[i] = super().predict(tv, mkm1, Pkm1, f_args=f_args)

        return pygm.GaussianMixture(wkm, mkm, Pkm)


class GmukfUpdate(UkfUpdate, GaussianMixtureUpdate):

    def __init__(self, h, R, sigma_pt_opts=None, cov_type='full', p=None):
        """ Unscented Kalman filter update

        Parameters
        ----------
        h : callable
            measurement function of the form :math:`h(x, *h_args)`
        R : ndarray
            measurement noise covariance matrix
        sigma_pt_opts: (optional) SigmaPointOptions
            options for sigma point placement
        cov_type: (optional) str
            type of covariance form to use. Default is 'full'
        p : (optional) scalar
            underweighting factor. p=1 results in no underweighting. p-->0 results
            in no covariance update
        """
        super().__init__(h, R, sigma_pt_opts=sigma_pt_opts, cov_type=cov_type, p=p)

    def __gauss_likelihood_agreement(self, z, zhat, W):
        return pygm.eval_mvnpdf(z, zhat, W)

    def cond_likelihood_prod(self, m, P, z, h_args=(), interm_vals=False):
        """ compute the product of the (non)linear Gaussian likelihood function and
        another Gaussian pdf

        Parameters
        ----------
        m : ndarray
            (nx,) prior mean
        P : ndarray
            (nx,nx) prior covariance matrix
        z  : ndarray
            (nz,) measurement
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function
        interm_vals : (optional) Boolean
            if True, returns intermediate values from computation. False by default

        Returns
        -------
        mp : ndarray
            (nx,) posterior mean
        Pp : ndarray
            (nx,nx) posterior state error covariance
        q : float
            likelihood agreement, :math:`q = N(z; h(m), E{(z-\hat{z})(z-\hat{z})^T} + R)`

        If interm_vals is true, additionally returns a dictionary containing:
        W : ndarray
            (nz,nz) innovatations covariance
        C : ndarray
            (nx,nz) cross-covariance
        K : ndarray
            gain matrix
        zhat : (ndarray)
            predicted measurement

        """
        mp, Pp, interm = super().update(m, P, z, interm_vals=True, h_args=h_args)
        q = self.__gauss_likelihood_agreement(z, interm['zhat'], interm['W'])

        if not interm_vals:
            return mp, Pp, q
        else:
            return mp, Pp, q, interm

    def update(self, pkm, zk, unnormalized=False, h_args=(), R=None):
        """ perform unscented Kalman filter update

        Parameters
        ----------
        pkm : GaussianMixture
            prior density at time tk
        zk : ndarray
            (nz,) measurement at time k
        unnormalized : optional
            if True, returns unnormalized distribution. False by default
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function
        R  : (optional) ndarray
            (ny,ny) measurement noise covariance matrix

        Returns
        -------
        GaussianMixture
            posterior density at time tk
        """
        wkp = np.empty_like(pkm.w, dtype=float)
        mkp = np.empty_like(pkm.m, dtype=float)
        Pkp = np.empty_like(pkm.P, dtype=float)
        for i, (wm,mm,Pm) in enumerate(pkm):
            mkp[i], Pkp[i], q = self.cond_likelihood_prod(mm, Pm, zk, h_args=h_args)
            wkp[i] = wm*q

        if not unnormalized:
            wkp /= np.sum(wkp)

        return pygm.GaussianMixture(wkp, mkp, Pkp)
