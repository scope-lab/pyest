import numpy as np
from abc import ABC, abstractmethod
import pyest.gm as pygm

class GaussianMixturePredict(ABC):
    """ Discrete Gaussian Mixture Filter Prediction

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def predict(self, tv, pkm1, *args, **kwargs):
        """ propagate gm forward in time
        """
        pass


class GaussianMixtureUpdate(ABC):
    """ Discrete Gaussian Mixture Kalman Filter Update
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def cond_likelihood_prod(self, m, P, z, h_args=(), interm_vals=False):
        """ compute the product of the linear Gaussian likelihood function and
        another Gaussian pdf

        Parameters
        ----------
        m : ndarray
            (nx x 1) prior mean
        P : ndarray
            (nx x nx) prior covariance matrix
        z  : ndarray
            (nz x 1) measurement
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function
        interm_vals : (optional) Boolean
            if True, returns intermediate values from computation. False by default

        Returns
        -------
        mp : ndarray
            (nx x 1) posterior mean
        Pp : ndarray
            (nx x nx) posterior state error covariance
        q : float
            likelihood agreement, :math:`q = N(z; h(m), E[(z-h(m))(z-h(m))^T])`

        If interm_vals is true, additionally returns a dictionary containing:
        W : ndarray
            (nz x nz) innovatations covariance
        C : (ndarray)
            (nx x nz) cross-covariance
        K : (ndarray)
            gain matrix
        zhat : (ndarray)
            predicted measurement

        """
        pass

    @abstractmethod
    def update(self, pkm, zk, unnormalized=False, h_args=(), *args, **kwargs):
        """ measurement-update of gm

        Parameters
        ----------
        pkm : GaussianMixture
            prior density at time tk
        zk : ndarray
            (nz,) measurement at time k
        h_args : (optional) tuple
            deterministic parameters to be passed to measurement function
        unnormalized : optional
            if True, returns unnormalized distribution. False by default

        Returns
        -------
        GaussianMixture

        """
        pass
