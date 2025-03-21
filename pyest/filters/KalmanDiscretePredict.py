import numpy as np

class KalmanDiscretePredict(object):

    def __init__(self, F, Q, M=None):
        """ Discrete Kalman Filter Prediction

        Parameters
        ----------
        F : ndarray or callable
            (nx x nx) is state transition matrix of the form
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

        if M is None:
            # if M not specified, default as function that returns
            M = convert_mat_to_fun(np.eye(self.F(0,0).shape[0]))

        self.M = M
        self.Q = Q

    def predict(self, tv, m_post, P_post, S_post=NotImplemented, \
        F_args=()):
        """ perform Kalman filter prediction step

        Parameters
        ----------
        tv : tuple
            start and stop times given as tv=(tkm1, tk)
        m_post : ndarray
            (1 x nx) posterior mean at tkm1
        P_post : ndarray
            (nx x nx) posterior covariance at tkm1
        F_args : (optional) tuple
            tuple of arguments to be additionally passed to state transition
            matrix F during prediction

        Returns
        -------
        m_prior : ndarray
            (1 x nx) prior mean at time tk
        P_prior : ndarray
            (nx x nx) prior covariance at tk
        """

        if not isinstance(F_args, tuple):
            F_args = (F_args,)

        Fk = self.F(*tv, *F_args)
        Mk = self.M(*tv)
        Qk = self.Q(*tv)

        # predict mean
        m_prior =  Fk @ m_post

        # predict covariance
        P_prior = Fk @ P_post @ Fk.T + Mk @ Qk @ Mk.T

        return m_prior, P_prior

    def _set_F(self, F):
        if isinstance(F,np.ndarray):
            self._F = convert_mat_to_fun(F)
        else:
            self._F = F

    def _get_F(self):
        return self._F

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

    F = property(_get_F, _set_F)
    M = property(_get_M, _set_M)
    Q = property(_get_Q, _set_Q)

def convert_mat_to_fun(A,*x):
    return lambda *x: A
