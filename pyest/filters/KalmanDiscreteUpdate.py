import numpy as np

class KalmanDiscreteUpdate(object):
    """ KalmanDiscreteUpdate Discrete linear Kalman filter update

    Parameters
    ----------
    H  : ndarray
        (nz x nx) measurement matrix
    R  : ndarray
        (ny x ny) measurement noise covariance matrix
    L  : (optional) ndarray
        (nz x ny) mapping matrix mapping measurement noise into
        measurement space
    cov_method : (optional) string
        method to use for covariance update. Valid options include 'general'
        (default), 'Joseph', 'standard', and 'KWK'.

    Written by Keith LeGrand, March 2019
    """

    def __init__(self, H, R, L=None, cov_method='general'):

        #TODO: check cov_update method
        #TODO: residual editing
        #TODO: downweighting

        self.H = H
        self.L = L
        self.R = R
        self.cov_method = cov_method

    def __set_R(self, R):
        #TODO: check is square
        self._R = R
        self.__update_mapped_meas_cov()

    def __get_R(self):
        return self._R

    def __set_H(self, H):
        self._H = H
        self.__Ht = H.T

    def __get_H(self):
        return self._H

    def __update_mapped_meas_cov(self):
        if self.L is not None:
            self.__LRLt = self.L @ self._R @ self.L.T
        else:
            self.__LRLt = self._R

    R = property(__get_R, __set_R)
    H = property(__get_H, __set_H)

    @staticmethod
    def gain(C, W):
        """ compute filter gain

        Parameters
        ----------
        C : ndarray
            (nx x nz) cross-covariance
        W : ndarray
            (nz x nz) innovations covariance

        Returns
        -------
        K : ndarray
            gain matrix

        TODO: list conditions when Kalman gain
        """
        return np.linalg.solve(W.T, C.T).T

    def innovations_cov(self, Pkm):
        """ compute inovations covariance
        Parameters
        ----------
        Pkm : ndarray
            (nx x nx) prior covariance matrix at time k

        Returns
        -------
        W : ndarray
            (nz x nz) innovations covariance
        """
        return self._H @ Pkm @ self.__Ht + self.__LRLt

    def cross_cov(self, Pkm):
        """ compute cross covariance
        Parameters
        ----------
        Pkm : ndarray
            (nx x nx) prior covariance matrix at time k

        Returns
        -------
        C : ndarray
            (nx x nz) cross covariance
        """
        return Pkm @ self.__Ht

    def expected_meas(self, mkm):
        """ compute measurement expectation
        Parameters
        ----------
        mkm : ndarray
            (nx x 1) prior mean at time k

        Returns
        -------
        w  : ndarray
            (nz x 1) predicted measurement at time k
        """
        return self._H @ mkm

    def update(self, mkm, Pkm, z, H=None, R=None, interm_vals=False):
        """ perform Kalman filter update

        Parameters
        ----------
        mkm : ndarray
            (nx x 1) prior mean at time k
        Pkm : ndarray
            (nx x nx) prior covariance matrix at time k
        z  : ndarray
            (nz x 1) measurement at time k
        H  : (optional) ndarray
            (nz x nx) measurement matrix
        R  : (optional) ndarray
            (ny x ny) measurement noise covariance matrix

        Returns
        -------
        mkp : ndarray
            (nx x 1) posterior mean
        Pkp : ndarray
            (nx x nx) posterior state error covariance

        If interm_vals is true, additionally returns a dictionary containing:
        W : ndarray
            (nz x nz) innovatations covariance
        C : (ndarray)
            (nx x nz) cross-covariance
        K : (ndarray)
            gain matrix

        Written by Keith LeGrand, March 2019
        """
        #TODO: check measurement size
        if H is not None:
            self.H = H

        if R is not None:
            self.R = R

        # predicted measurement
        w = self.expected_meas(mkm)
        W = self.innovations_cov(Pkm)
        C = self.cross_cov(Pkm)
        K = KalmanDiscreteUpdate.gain(C, W)
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
            Pkp = ImKH@Pkm@ImKH.T + K@self.__LRLt@K.T
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
            return mkp, Pkp, {'W':W, 'C':C, 'K':K}
