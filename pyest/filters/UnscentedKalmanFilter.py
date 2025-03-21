import numpy as np
from pyest.filters.sigma_points import unscented_transform, SigmaPointOptions
from pyest.filters.KalmanFilter import KalmanDiscretePredict, KalmanDiscreteUpdate
from pyest.utils import make_tuple

class UkfPredict(KalmanDiscretePredict):
    """ Unscented Kalman Filter Prediction

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
    sigma_pt_opts: SigmaPointOptions
        options for sigma point placement
    cov_type: (optional) str
        type of covariance matrix. Default is 'full'

    """

    def __init__(self, f, Q, M=None, sigma_pt_opts = None, cov_type='full'):

        super().__init__(Q=Q, M=M)
        self.f = f

        # determine number of elements in state
        self.nx = self.M(0, 0).shape[0]

        if not sigma_pt_opts:
            self.sigma_pt_opts = SigmaPointOptions()
        else:
            self.sigma_pt_opts = sigma_pt_opts

        self.cov_type = cov_type
        #TODO: add cov_type to KalmanDiscretePredict class


    def predict(self, tv, m_post, cov_post, \
        f_args=()):
        """ perform Kalman filter prediction step

        Parameters
        ----------
        tv : tuple
            start and stop times given as tv=(tkm1, tk)
        m_post : ndarray
            (1 x nx) posterior mean at tkm1
        cov_post : ndarray
            (nx,nx) posterior covariance at tkm1
        f_args : (optional) tuple
            tuple of arguments to be additionally passed to the system function

        Returns
        -------
        m_prior : ndarray
            (1 x nx) prior mean at time tk
        P_prior : ndarray
            (nx,nx) prior covariance at tk
        """

        if not isinstance(f_args, tuple):
            f_args = (f_args,)

        return unscented_transform(m_post, cov_post, self.f, self.sigma_pt_opts, \
                                   self.cov_type, self.Q(*tv), g_args=(*tv, *f_args))[:2]


    def _set_f(self, f):
        self._f = f

    def _get_f(self):
        return self._f

    f = property(_get_f, _set_f)


class UkfUpdate(KalmanDiscreteUpdate):

    def __init__(self, h, R, L=None, sigma_pt_opts=None, cov_type='full', p=None):
        """ Unscented Kalman filter update

        Parameters
        ----------
        h : callable
            measurement function of the form :math:`h(x, ...)`
        R : ndarray
            (ny,ny) measurement noise covariance matrix
        L  : (optional) ndarray
            (nz,ny) mapping matrix mapping measurement noise into
        sigma_pt_opts: SigmaPointOptions
            options for sigma point placement
        p : (optional) float
            underweighting factor. p=1 results in no underweighting. p-->0 results
            in no covariance update
        """
        super().__init__(R=R, L=L, p=p)
        self.h = h
        self.cov_type = cov_type
        if not sigma_pt_opts:
            self.sigma_pt_opts = SigmaPointOptions()
        else:
            self.sigma_pt_opts = sigma_pt_opts

    def _set_h(self, h):
        self._h = h

    def _get_h(self):
        return self._h

    @staticmethod
    def gain(C, W):
        """ compute filter gain

        Parameters
        ----------
        C : ndarray
            (nx,nz) cross-covariance
        W : ndarray
            (nz,nz) innovations covariance

        Returns
        -------
        K : ndarray
            gain matrix

        """
        return np.linalg.solve(W.T, C.T).T

    def innovations_cov(self, mkm, Pkm, h_args=()):
        """ compute inovations covariance
        Parameters
        ----------
        mkm : ndarray
            (nx,) prior mean at time k
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function

        Returns
        -------
        W : ndarray
            (nz,nz) innovations covariance
        """
        return unscented_transform(
            mkm, Pkm,self.h, self.R, sigma_pt_opts=self.sigma_pt_opts, cov_type=self.cov_type, g_args=h_args)[1]

    def cross_cov(self, mkm, Pkm, h_args=()):
        """ compute cross covariance
        Parameters
        ----------
        mkm : ndarray
            (nx,) prior mean at time k
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function

        Returns
        -------
        C : ndarray
            (nx,nz) cross covariance
        """
        # calculate expected z, innovation covariance, UT deviations, UT sigma
        # points, and UT weights
        zhat, Wkp, D, S, wc = unscented_transform(
            mkm, Pkm, self.h,self.R, self.sigma_pt_opts, cov_type=self.cov_type, g_args=h_args)

        # compute cross covariance
        return (S.T - mkm).T @ np.diag(wc) @ D.T # state-measurement cross-covariance

    def expected_meas(self, mkm, Pkm, h_args=()):
        """ compute measurement expectation
        Parameters
        ----------
        mkm : ndarray
            (nx,) prior mean at time k
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function

        Returns
        -------
        w  : ndarray
            (nz,) predicted measurement at time k
        """
        return unscented_transform(
            mkm, Pkm, self.h, self.R, self.sigma_pt_opts, cov_type=self.cov_type, g_args=h_args)[0]

    def update(self, mkm, Pkm, z, R=None, \
     h_args=(), interm_vals=False):
        """ perform unscented Kalman filter update

        Parameters
        ----------
        mkm : ndarray
            (nx,) prior mean at time k
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k
        z  : ndarray
            (nz,) measurement at time k
        R  : (optional) ndarray
            (ny,ny) measurement noise covariance matrix
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function

        Returns
        -------
        mkp : ndarray
            (nx,) posterior mean
        Pkp : ndarray
            (nx,nx) posterior state error covariance

        If interm_vals is true, additionally returns a dictionary containing:
        W : ndarray
            (nz,nz) innovatations covariance
        C : (ndarray)
            (nx,nz) cross-covariance
        K : (ndarray)
            gain matrix
        zhat : (ndarray)
            predicted measurement

        Written by Keith LeGrand, May 2019
        """
        h_args = make_tuple(h_args)

        if R is not None:
            self.R = R

        p = self.p
        # calculate expected z, innovation covariance, UT deviations, UT sigma
        # points, and UT weights
        zhat, Wkp, D, sigmas, _ = unscented_transform(
            mkm, Pkm, self.h, self.sigma_pt_opts, self.cov_type, self._LRLt, g_args=h_args)

        # compute cross covariance
        Pxz = (sigmas.X.T - mkm).T @ np.diag(sigmas.wc) @ D.T # state-measurement cross-covariance

        if p is not None:
            if np.linalg.norm(Wkp - self.R) > p/(1-p)*np.linalg.norm(self.R):
                # apply underweighting
                U = (1-p)/p * (Wkp - self.R)
                Wkp += U

        K = UkfUpdate.gain(Pxz, Wkp) #Pxz/Wkp  Kalman gain

        r = z - zhat # innovation

        mkp = mkm + K @ r      # updated state estimate
        Pkp = Pkm - K @ Pxz.T  # updated covariance

        if not interm_vals:
            return mkp, Pkp
        else:
            return mkp, Pkp, {'W':Wkp, 'C':Pxz, 'K':K, 'zhat':zhat}
