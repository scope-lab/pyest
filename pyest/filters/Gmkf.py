import numpy as np
import pyest.gm as pygm
from pyest.filters import GaussianMixturePredict, GaussianMixtureUpdate
from pyest.filters import KfdPredict, KfdUpdate, EkfdPredict
from pyest.utils import make_tuple


class GmkfPredict(KfdPredict, GaussianMixturePredict):
    """ Discrete Gaussian Mixture Kalman Filter Prediction

    Parameters
    ----------
    F : ndarray or callable
        (nx,nx) is state transition matrix of the form
        x_k = F(tkm, tk, *args) @ x. If provided an ndarray instead, F will
        automatically be recast as a callable.
    Q : ndarray or callable
        process noise covariance matrix of the form Q(tkm1, tkm). If provided
        an ndarray instead, Q will automatically be recast as a callable.
    M : (optional) ndarray or callable
        process noise mapping matrix of the form M(tkm1, tkm). If provided
        an ndarray instead, M will automatically be recast as a callable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, tv, pkm1, f_args=()):
        """ perform GMKF prediction step

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


class GmekfPredict(EkfdPredict, GaussianMixturePredict):
    """ Discrete Gaussian Mixture Kalman Filter Prediction

    Parameters
    ----------
    f : callable
        nonlinear difference equation of the form
        x[k+1] = f(t[k+1], t[k], x[k], *args).
    F : callable
        (nx,nx) is state transition matrix of the form
        F(tkm, tk, x, *args). If provided an ndarray instead, F will
        automatically be recast as a callable.
    Q : ndarray or callable
        process noise covariance matrix of the form Q(tkm1, tkm). If provided
        an ndarray instead, Q will automatically be recast as a callable.
    M : (optional) ndarray or callable
        process noise mapping matrix of the form M(tkm1, tkm). If provided
        an ndarray instead, M will automatically be recast as a callable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, tv, pkm1, f_args=()):
        """ perform GMKF prediction step

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

class GmkfUpdate(KfdUpdate, GaussianMixtureUpdate):
    """ Discrete Gaussian Mixture Kalman Filter Update

    Parameters
    ----------
    H  : ndarray
        (nz x nx) measurement matrix
    R  : ndarray
        (ny,ny) measurement noise covariance matrix
    L  : (optional) ndarray
        (nz,ny) mapping matrix mapping measurement noise into
        measurement space
    cov_method : (optional) string
        method to use for covariance update. Valid options include 'general'
        (default), 'Joseph', 'standard', and 'KWK'.

    Written by Keith LeGrand, March 2019
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __lin_gauss_likelihood_agreement(self, z, zhat, W):
        return pygm.eval_mvnpdf(z, zhat, W)

    def lin_gauss_cond_likelihood_prod(self, m, P, z, h_args=(), interm_vals=False):
        """ compute the product of the linear Gaussian likelihood function and
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
            likelihood agreement, :math:`q = N(z; Hm, HPH' + R)`

        If interm_vals is true, additionally returns a dictionary containing:
        W : ndarray
            (nz,nz) innovatations covariance
        C : (ndarray)
            (nx,nz) cross-covariance
        K : (ndarray)
            gain matrix
        zhat : (ndarray)
            predicted measurement

        """
        mp, Pp, interm = super().update(m, P, z, interm_vals=True, h_args=h_args)
        q = self.__lin_gauss_likelihood_agreement(z, interm['zhat'], interm['W'])

        if not interm_vals:
            return mp, Pp, q
        else:
            return mp, Pp, q, interm

    def cond_likelihood_prod(self, m, P, z, h_args=(), interm_vals=False):
        """ compute the product of the linear Gaussian likelihood function and
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
            likelihood agreement, :math:`q = N(z; Hm, HPH' + R)`

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
        return self.lin_gauss_cond_likelihood_prod(m, P, z, h_args=h_args, interm_vals=interm_vals)

    def update(self, pkm, zk, unnormalized=False, h_args=(), *args, **kwargs):
        """ measurement-update of gm

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
        """
        wkp = np.empty_like(pkm.w, dtype=float)
        mkp = np.empty_like(pkm.m, dtype=float)
        Pkp = np.empty_like(pkm.P, dtype=float)
        for i, (wm,mm,Pm) in enumerate(pkm):
            mkp[i], Pkp[i], q = self.lin_gauss_cond_likelihood_prod(mm, Pm, zk, h_args=h_args)
            wkp[i] = wm*q

        if not unnormalized:
            wkp /= np.sum(wkp)

        return pygm.GaussianMixture(wkp, mkp, Pkp)

def lin_gauss_likelihood_agreement(z, m, P, H, R, L=None, h_args=()):
    """ compute the likelihood agreement of a measurment z with the linear
    Gaussian likelihood function N(z; Hx, HPH' + LRL')
    """
    if L is None:
        LRLt = R
    else:
        LRLt = L@R@L.t

    h_args = make_tuple(h_args)
    Hk = H(*h_args)
    return pygm.eval_mvnpdf(z, Hk@m, Hk@P@Hk.T + LRLt)
