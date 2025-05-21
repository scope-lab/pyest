import numpy as np
from abc import ABC, abstractmethod, abstractstaticmethod
from pyest.utils import convert_mat_to_fun, check_covariance, make_tuple

class KalmanDiscreteUpdate(ABC):
    """ KalmanDiscreteUpdate Discrete linear Kalman filter update

    Parameters
    ----------
    R  : ndarray
        (ny, ny) measurement noise covariance matrix
    L  : (optional) ndarray
        (nz,ny) mapping matrix mapping measurement noise into
        measurement space
    cov_method : (optional) string
        method to use for covariance update. Valid options include 'general'
        (default), 'Joseph', 'standard', and 'KWK'.
    p : (optional) scalar
        underweighting factor. p=1 results in no underweighting. p-->0 results
        in no covariance update

    Written by Keith LeGrand, March 2019
    """

    def __init__(self, R, L=None, p=None, *args, **kwargs):

        #TODO: check cov_update method
        #TODO: residual editing

        self.L = L
        self.R = R
        self.p = p

    def __set_R(self, R):
        check_covariance(R)
        self._R = R
        self._update_mapped_meas_cov()

    def __get_R(self):
        return self._R


    def _update_mapped_meas_cov(self):
        if self.L is not None:
            self._LRLt = self.L @ self._R @ self.L.T
        else:
            self._LRLt = self._R

    def __get_p(self):
        return self._p

    def __set_p(self, p):
        if p is None:
            self._p = p
        elif p <=0 or p>1:
            raise(ValueError(p))
        else:
            self._p = p


    R = property(__get_R, __set_R)
    p = property(__get_p, __set_p)

    @abstractstaticmethod
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
        pass

    @abstractmethod
    def innovations_cov(self, Pkm):
        """ compute inovations covariance
        Parameters
        ----------
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k

        Returns
        -------
        W : ndarray
            (nz,nz) innovations covariance
        """
        pass

    @abstractmethod
    def cross_cov(self, Pkm):
        """ compute cross covariance
        Parameters
        ----------
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k

        Returns
        -------
        C : ndarray
            (nx,nz) cross covariance
        """
        pass

    @abstractmethod
    def expected_meas(self, mkm, h_args):
        """ compute measurement expectation
        Parameters
        ----------
        mkm : ndarray
            (nx,) prior mean at time k
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function

        Returns
        -------
        w  : ndarray
            (nz,) predicted measurement at time k
        """
        pass

    @abstractmethod
    def update(self, mkm, Pkm, z, R=None, h_args=(), interm_vals=False):
        """ perform Kalman filter update

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
        C : ndarray
            (nx,nz) cross-covariance
        K : ndarray
            gain matrix
        zhat : ndarray
            predicted measurement

        Written by Keith LeGrand, March 2019
        """
        pass


class KalmanDiscretePredict(ABC):

    def __init__(self, Q, M=None):
        """ Discrete Kalman Filter Prediction

        Parameters
        ----------
        Q : ndarray or callable
            process noise covariance matrix of the form Q(tkm1, tkm). If provided
            an ndarray instead, Q will automatically be recast as a callable.
        M : (optional) ndarray or callable
            process noise mapping matrix of the form M(tkm1, tkm). If provided
            an ndarray instead, M will automatically be recast as a callable.
        """

        self.Q = Q

        if M is None:
            # if M not specified, default as function that returns
            M = convert_mat_to_fun(np.eye(self.Q(0,0).shape[0]))

        self.M = M

    @abstractmethod
    def predict(self, tv, m_post, P_post, S_post=NotImplemented, *args, **kwargs):
        """ perform Kalman filter prediction step

        Parameters
        ----------
        tv : tuple
            start and stop times given as tv=(tkm1, tk)
        m_post : ndarray
            (nx,) posterior mean at tkm1
        P_post : ndarray
            (nx,nx) posterior covariance at tkm1

        Returns
        -------
        m_prior : ndarray
            (nx,) prior mean at time tk
        P_prior : ndarray
            (nx,nx) prior covariance at tk
        """
        pass

    def _set_M(self, M):
        if isinstance(M, np.ndarray):
            self._M = convert_mat_to_fun(M)
        else:
            self._M = M

    def _get_M(self):
        return self._M

    def _set_Q(self, Q):
        if isinstance(Q,np.ndarray):
            self._Q = convert_mat_to_fun(Q)
        else:
            self._Q = Q

    def _get_Q(self):
        return self._Q

    M = property(_get_M, _set_M)
    Q = property(_get_Q, _set_Q)

class KfdUpdate(KalmanDiscreteUpdate):
    """ KfdUpdate Discrete linear Kalman filter update

    Parameters
    ----------
    R  : ndarray
        (ny,ny) measurement noise covariance matrix
    H  : ndarray
        (nz,nx) measurement matrix
    L  : (optional) ndarray
        (nz,ny) mapping matrix mapping measurement noise into
        measurement space
    cov_method : (optional) string
        method to use for covariance update. Valid options include 'general'
        (default), 'Joseph', 'standard', and 'KWK'.
    p : (optional) scalar
        underweighting factor. p=1 results in no underweighting. p-->0 results
        in no covariance update

    Written by Keith LeGrand, March 2019
    """

    def __init__(self, R, H, L=None, cov_method='general', p=None):

        super().__init__(R=R, L=L, p=p)

        #TODO: check cov_update method
        #TODO: residual editing

        self.cov_method = cov_method
        self.H = H

    def __set_H(self, H):
        if isinstance(H, np.ndarray):
            H = convert_mat_to_fun(H)
        self._H = H


    def __get_H(self):
        return self._H

    H = property(__get_H, __set_H)

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

    def __innovations_cov(self, Pkm, Hk):
        """ compute inovations covariance
        Parameters
        ----------
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k
        Hk : ndarray
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function

        Returns
        -------
        W : ndarray
            (nz,nz) innovations covariance
        """
        return Hk @ Pkm @ Hk.T + self._LRLt

    def innovations_cov(self, Pkm, h_args=()):
        """ compute inovations covariance
        Parameters
        ----------
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function

        Returns
        -------
        W : ndarray
            (nz,nz) innovations covariance
        """
        h_args = make_tuple(h_args)
        Hk = self._H(*h_args)
        return self.__innovations_cov(Pkm, Hk)

    def __cross_cov(self, Pkm, Hk):
        """ compute cross covariance
        Parameters
        ----------
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k
        Hk : ndarray
            (nz,nx) measurement matrix

        Returns
        -------
        C : ndarray
            (nx,nz) cross covariance
        """
        return Pkm @ Hk.T

    def cross_cov(self, Pkm, h_args=()):
        """ compute cross covariance
        Parameters
        ----------
        Pkm : ndarray
            (nx,nx) prior covariance matrix at time k
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function

        Returns
        -------
        C : ndarray
            (nx,nz) cross covariance
        """
        h_args = make_tuple(h_args)
        Hk = self._H(*h_args)
        return self.__cross_cov(Pkm, Hk)


    def __expected_meas(self, mkm, Hk):
        """ compute measurement expectation
        Parameters
        ----------
        mkm : ndarray
            (nx,) prior mean at time k
        Hk : ndarray
            (nz,nx) measurement matrix

        Returns
        -------
        w  : ndarray
            (nz,) predicted measurement at time k
        """
        return Hk @ mkm

    def expected_meas(self, mkm, h_args=()):
        """ compute measurement expectation
        Parameters
        ----------
        mkm : ndarray
            (nx,) prior mean at time k
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function

        Returns
        -------
        w  : ndarray
            (nz,) predicted measurement at time k
        """
        h_args = make_tuple(h_args)
        Hk = self._H(*h_args)
        return self.__expected_meas(mkm, Hk)

    def update(self, mkm, Pkm, z,  R=None, H=None, h_args=(), interm_vals=False):
        """ perform Kalman filter update

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
        H  : (optional) ndarray
            (nz,nx) measurement matrix
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
        C : ndarray
            (nx,nz) cross-covariance
        K : ndarray
            gain matrix
        zhat : ndarray
            predicted measurement

        Written by Keith LeGrand, March 2019
        """
        #TODO: check measurement size
        if H is not None:
            self.H = H

        if R is not None:
            self.R = R

        h_args = make_tuple(h_args)
        Hk = self._H(*h_args)

        # predicted measurement
        w = self.__expected_meas(mkm, Hk)
        W = self.__innovations_cov(Pkm, Hk)
        C = self.__cross_cov(Pkm, Hk)
        K = KfdUpdate.gain(C, W)
        mkp = mkm + K@(z - w)

        if self.cov_method == 'general':
            # general form. does not require gain be Kalman gain. gain must
            # be linear
            Pkp = Pkm - C@K.T - K@C.T + K@W@K.T
        elif self.cov_method == 'Joseph':
            # Joseph form. does not require gain be Kalman gain.
            # measurements must be linear
            I = np.eye(self._H.shape[1])
            ImKH = I - K@H
            Pkp = ImKH@Pkm@ImKH.T + K@self._LRLt@K.T
        elif self.cov_method == 'standard':
            # standard form. requires gain be Kalman gain and linear
            # measurements
            I = np.eye(self._H.shape[1])
            Pkp = (I - K@self._H)@Pkm
        elif self.cov_method == 'KWK':
            # yet another form. requires gain be Kalman gain but does not
            # require linear measurements
            Pkp = Pkm - K@W@K.T

        if not interm_vals:
            return mkp, Pkp
        else:
            return mkp, Pkp, {'W':W, 'C':C, 'K':K, 'zhat':w}

class KfdPredict(KalmanDiscretePredict):

    def __init__(self, F, Q, M=None):
        """ Discrete Kalman Filter Prediction

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
        if F is None:
            RuntimeError("gotta specify F")

        self.F = F

        super().__init__(Q=Q, M=M)

    def predict(self, tv, m_post, P_post, S_post=NotImplemented, \
        f_args=()):
        """ perform Kalman filter prediction step

        Parameters
        ----------
        tv : tuple
            start and stop times given as tv=(tkm1, tk)
        m_post : ndarray
            (nx,) posterior mean at tkm1
        P_post : ndarray
            (nx,nx) posterior covariance at tkm1
        f_args : (optional) tuple
            tuple of arguments to be additionally passed to state transition
            matrix F during prediction

        Returns
        -------
        m_prior : ndarray
            (nx,) prior mean at time tk
        P_prior : ndarray
            (nx,nx) prior covariance at tk
        """

        f_args = make_tuple(f_args)

        Fk = self.F(*tv, *f_args)
        Mk = self.M(*tv)
        Qk = self.Q(*tv)

        # predict mean
        m_prior = Fk @ m_post

        # predict covariance
        P_prior = Fk @ P_post @ Fk.T + Mk @ Qk @ Mk.T

        return m_prior, P_prior

    def _set_F(self, F):
        if isinstance(F, np.ndarray):
            self._F = convert_mat_to_fun(F)
        else:
            self._F = F

    def _get_F(self):
        return self._F

    F = property(_get_F, _set_F)

class EkfdPredict(KalmanDiscretePredict):

    def __init__(self, f, F, Q, M=None):
        """ Discrete Kalman Filter Prediction

        Parameters
        ----------
        f : callable
            discrete time nonlinear dynamics function of the form xk = f(tkm1, tk, xkm1)
        F : callable
            (nx,nx) is state transition matrix of the form
            dx_k/dx_km1 = F(tkm1, tk, xkm1, *args).
        Q : ndarray or callable
            process noise covariance matrix of the form Q(tkm1, tkm). If provided
            an ndarray instead, Q will automatically be recast as a callable.
        M : (optional) ndarray or callable
            process noise mapping matrix of the form M(tkm1, tkm). If provided
            an ndarray instead, M will automatically be recast as a callable.
        """
        if F is None:
            RuntimeError("gotta specify F")

        self.F = F
        self.f = f

        super().__init__(Q=Q, M=M)

    def predict(self, tv, m_post, P_post, S_post=NotImplemented, \
        f_args=()):
        """ perform Kalman filter prediction step

        Parameters
        ----------
        tv : tuple
            start and stop times given as tv=(tkm1, tk)
        m_post : ndarray
            (nx,) posterior mean at tkm1
        P_post : ndarray
            (nx,nx) posterior covariance at tkm1
        f_args : (optional) tuple
            tuple of arguments to be additionally passed to state transition
            matrix F during prediction

        Returns
        -------
        m_prior : ndarray
            (nx,) prior mean at time tk
        P_prior : ndarray
            (nx,nx) prior covariance at tk
        """

        f_args = make_tuple(f_args)

        Fk = self.F(*tv, m_post, *f_args)
        Mk = self.M(*tv) # TODO: make Mk and Qk functions of m_post
        Qk = self.Q(*tv)

        # predict mean
        m_prior = self.f(*tv, m_post, *f_args)

        # predict covariance
        P_prior = Fk @ P_post @ Fk.T + Mk @ Qk @ Mk.T

        return m_prior, P_prior

    def _set_F(self, F):
        if isinstance(F, np.ndarray):
            self._F = convert_mat_to_fun(F)
        else:
            self._F = F

    def _get_F(self):
        return self._F

    F = property(_get_F, _set_F)