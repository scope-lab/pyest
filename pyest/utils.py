import numpy as np
import subprocess
from datetime import datetime
from typing import Any, Callable
from . linalg import is_pos_def, is_square_matrix


def create_rev_metadata(filename=None):

    calling_filename = __file__
    git_call = 'git describe --match=NeVeRmAtCh --always --abbrev=40 --dirty'
    sha = subprocess.check_output(git_call, shell=True, universal_newlines=True)

    metadata = "Created by "  + calling_filename + " at " + datetime.now().strftime("%Y_%m_%d %H:%M:%S") + ".\n"
    metadata += "Repository at commit " +  sha

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(metadata)

    return metadata, calling_filename


def convert_mat_to_fun(A, *x):
    return lambda *x: A


def has_unique_columns(A):
    unique_vectors = np.unique(A, axis=1)
    num_unique_vectors = unique_vectors.shape[1]
    num_columns = A.shape[1]

    return num_unique_vectors == num_columns


def check_covariance(P):
    """ check if covariance is square and pos-def """
    if not is_square_matrix(P):
        return CovarianceNotSquare(P)

    if not is_pos_def(P):
        raise CovarianceNotPositiveDefinite(P)


def make_tuple(args):
    if not isinstance(args, tuple):
        args = (args,)

    return args


def fail(f, error, *args, **kwargs):
    try:
        f(*args, **kwargs)
        raise AssertionError('should throw exception')
    except error:
        pass
    except Exception as e:
        raise AssertionError(
            'received {} instead of {} exception'.format(type(e), error))


class protected_cached_property:
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.name = func.__name__

    def __get__(self, obj: Any, cls: Any) -> Any:
        if obj is None:
            return self
        # Use the cached_property naming convention
        cached_name = f"__{self.name}"
        if not hasattr(obj, cached_name):
            setattr(obj, cached_name, self.func(obj))
        return getattr(obj, cached_name)

    def __set__(self, obj: Any, value: Any) -> None:
        raise AttributeError(f"Can't modify {self.name} directly")


class BadCovariance(ValueError):
    """ Raised when covariance matrix is bad """


class CovarianceNotSquare(BadCovariance):
    """ raised when covariance matrix not square """


class CovarianceNotPositiveDefinite(BadCovariance):
    """ raised when covariance matrix not positive definite """


class BadCholeskyFactor(ValueError):
    """ Raised when Cholesky factor will result in non semi-def covariance """
    pass
    ValueError()
