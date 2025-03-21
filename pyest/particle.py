import numpy as np


class DiracMixture(object):

    def __init__(self, w, m):
        """
        dirac mixture model

        Parameters
        ----------
        w: ndarray
            (n,) weights
        m: ndarray
            (n,nx) means
        """

        self._w = w
        self._m = m

    def __len__(self):
        return len(self.w)

    def __getitem__(self, ind):
        return self.get_comp(ind)

    def get_comp(self, ind):
        return self.w[ind], self.m[ind]

    # getter
    def get_w(self):
        return self._w

    def get_m(self):
        return self._m

    # setter
    def set_w(self, w):
        self._w = w

    def set_m(self, m):
        self._m = m

    def get_size(self):
        return len(self.w)

    w = property(get_w, set_w)
    m = property(get_m, set_m)

    def mean(self):
        """ return the mean of the Dirac mixture """
        return np.sum(self.w[:, np.newaxis] * self.m, axis=0)

    def cov(self):
        """ return the covariance of the distribution """
        wsum = np.sum(self.w)
        mean = self.mean()
        return np.sum([
            w/wsum*(np.outer(m, m)) for w, m in self
        ], axis=0) - np.outer(mean, mean)
