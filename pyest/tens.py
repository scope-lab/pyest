import numpy as np
import string
import math
import itertools
import functools


def power_iterate_string(tens):
    """Function to calculate the index string for einsum (up to 26 dimensional tensor)

    Args:
        tens (np array)
            Tensor

    Returns:
        einsum string to perform power iteration (string)
    """
    assert tens.ndim <= 26
    # looks like "zabcd,a,b,c,d->z"
    stringEin = "z"
    stringContract = string.ascii_lowercase[: tens.ndim - 1]
    secondString = ""
    for char in stringContract:
        secondString += "," + char
    stringEin += stringContract + secondString + "->" "z"
    return stringEin


def tensor_square_string(tens):
    """Function to calculate the index string for einsum (up to 1-13 dimensional tensor)
    Args:
        tens (np array)
            Tensor

    Returns:
        einsum string to perform tensor squaring (string)
    """
    assert tens.ndim < 13
    # looks like "abcd,azyx-bcdzyx>"
    firstString = string.ascii_lowercase[1 : tens.ndim]
    secondString = string.ascii_lowercase[26 : 26 - tens.ndim : -1]
    stringEin = (
        "a" + firstString + ",a" + secondString + "->" + firstString + secondString
    )
    return stringEin


def power_iterate(stringEin, tensOrder, tens, vec):
    """Function to perform one higher order power iteration on a symmetric tensor

    Single step

    Args:
        stringEin (string)
            String to instruct einsum to perform contractions

        tensOrder (int)
            Order of the tensor

        tens (np array)
            Tensor

        vec (np array)
            Vector

    Returns:
        vecNew (np array)

        vecNorm (float)
    """
    vecNew = np.einsum(stringEin, tens, *([vec] * (tensOrder - 1)))
    vecNorm = np.linalg.norm(vecNew)
    return vecNew / vecNorm, vecNorm


def power_iteration(tens, vecGuess, maxIter, tol):
    """Function to perform higher order power iteration on a symmetric (enough) tensore

    Args:
        tens (np array)
            Tensor

        vec (np array)
            Vector

        maxIter (int)
            Max number of iterations to perform

        tol (float)
            Tolerance for difference and iterates

    Returns:
        eigVec (np array)

        eigValue (np array)
    """
    stringEin = power_iterate_string(tens)
    tensOrder = tens.ndim
    vec = vecGuess
    vecNorm = None
    for i in range(maxIter):
        vecPrev = vec
        vec, vecNorm = power_iterate(stringEin, tensOrder, tens, vecPrev)
        if np.linalg.norm(vec - vecPrev) < tol:
            break
    return vec, vecNorm


def tensor_2_norm(tens, guessVec):
    """Function to calculate tensor 2-norm and argmax

    The square root of the maximum eigenvalue of the tensor squared and its argmax

    Args:
        tens (np array)
            Arbitrary 1-m tensor
        guessVec (np array)
            Guess vector for input that maximizes the tensor

    Returns:
        tensor_norm (float), tensor_arg_max (np array)
    """
    tensSquared = np.einsum(tensor_square_string(tens), tens, tens)
    tensorArgMax, tensNorm = power_iteration(tensSquared, guessVec, 100, 1e-6)
    return np.sqrt(tensNorm), tensorArgMax


def tensor_2_norm_trials(tens, trials=500):
    """Function to calculate tensor 2-norm and argmax

    The square root of the maximum eigenvalue of the tensor squared and its argmax

    Args:
        tens (np array)
            Arbitrary 1-m tensor
        trials (int)
            Number of random trials for initial guess vector

    Returns:
        tensor_norm (float), tensor_arg_max (np array)
    """
    tens_squared = np.einsum(tensor_square_string(tens), tens, tens)
    max_norm = 0
    for i in range(trials):
        # sample uniformly randomly from the sphere with same dimension as out tensor
        guess = np.random.multivariate_normal(np.zeros(len(tens)), np.identity(len(tens)))
        guess = guess / np.linalg.norm(guess)
        tensor_arg_max, tens_norm = power_iteration(tens_squared, guess, 100, 1e-12)

        res = np.einsum("ijkl,j,k,l", tens_squared, tensor_arg_max, tensor_arg_max, tensor_arg_max)
        res = res/np.linalg.norm(res)

        if max_norm < tens_norm and np.linalg.norm(res-tensor_arg_max) < .0001:
            overall_tensor_arg_max = tensor_arg_max
            max_norm = tens_norm
    return np.sqrt(max_norm), overall_tensor_arg_max


def symmetrize_tensor(tens):
    """Symmetrize a tensor

    Args:
        tens (np array)
            Tensor

    Returns:
        symTens (np array)
    """
    dim = tens.ndim
    rangedim = range(dim)
    tensDiv = tens / math.factorial(dim)
    permutes = map(
        lambda sigma: np.moveaxis(tensDiv, rangedim, sigma),
        itertools.permutations(range(dim)),
    )
    symTens = functools.reduce(lambda x, y: x + y, permutes)
    return symTens


def get_polynomial_bound(tens):
    """Function to find a bound on the value of a sclar valued polynomial on the unit sphere

    Args:
        tens (np array)
            Tensor

    Returns:
        K (double)
            Bound on the polynomial on the unit sphere
    """
    tensOrder = tens.ndim
    # return np.sum(np.abs(tens)) * (tensOrder - 1.0)
    # print(np.sum(np.abs(tens)) * (tensOrder - 1.0))
    matrix = np.reshape(tens, (np.prod(tens.shape[:2]), np.prod(tens.shape[2:])))
    #shift = np.linalg.matrix_norm(matrix, ord=2) * (tensOrder - 1.0) # works in numpy >2.0.0
    shift = np.linalg.norm(matrix, axis=(-2, -1), ord=2) * (tensOrder - 1.0)
    # print(shift)
    return shift


def MM_iterate(stringEin, tensOrder, tens, K, vec):
    """Function to perform one step of polynomial optimization on a scalar valued polynomial on unit sphere

    Single step

    Args:
        stringEin (string)
            String to instruct einsum to perform contractions

        tensOrder (int)
            Order of the tensor

        tens (np array)
            Tensor

        K (double)
            Damping constant

        vec (np array)
            Vector

    Returns:
        vecNew (np array)

        vecNorm (float)
    """
    poly = np.einsum(stringEin, tens, *([vec] * (tensOrder - 1)))
    vecNew = vec + 1 / K * poly
    vecNorm = np.linalg.norm(vecNew)
    tensNorm = np.linalg.norm(poly)
    return vecNew / vecNorm, tensNorm


def MM_iteration(tens, vecGuess, maxIter, tol):
    """Function to perform polynomial optimization on a scalar valued polynomial on unit sphere

    Args:
        tens (np array)
            Tensor

        vec (np array)
            Vector

        maxIter (int)
            Max number of iterations to perform

        tol (float)
            Tolerance for difference and iterates

    Returns:
        eigVec (np array)

        eigValue (np array)
    """
    stringEin = power_iterate_string(tens)
    tensOrder = tens.ndim
    vec = vecGuess
    vecNorm = None
    K = get_polynomial_bound(tens)
    for i in range(maxIter):
        vecPrev = vec
        vec, vecNorm = MM_iterate(stringEin, tensOrder, tens, K, vecPrev)
        if np.linalg.norm(vec - vecPrev) < tol:
            break
    return vec, vecNorm


def MM_iteration(tens, vecGuess, K, maxIter, tol):
    """Function to perform polynomial optimization on a scalar valued polynomial on unit sphere

    Args:
        tens (np array)
            Tensor

        vec (np array)
            Vector

        K (double)
            Damping constant

        maxIter (int)
            Max number of iterations to perform

        tol (float)
            Tolerance for difference and iterates

    Returns:
        eigVec (np array)

        eigValue (np array)
    """
    stringEin = power_iterate_string(tens)
    tensOrder = tens.ndim
    vec = vecGuess
    vecNorm = None
    j=1
    for i in range(maxIter):
        j=i
        vecPrev = vec
        vec, vecNorm = MM_iterate(stringEin, tensOrder, tens, K, vecPrev)
        if np.linalg.norm(vec - vecPrev) < tol:
            break
    return vec, vecNorm


def tensor_2_norm_trials_shifted(tens, trials=500):
    """Function to calculate tensor 2-norm and argmax

    The square root of the maximum eigenvalue of the tensor squared and its argmax

    Args:
        tens (np array)
            Arbitrary 1-m tensor
        trials (int)
            Number of random trials for initial guess vector

    Returns:
        tensor_norm (float), tensor_arg_max (np array)
    """
    tens_squared = np.einsum(tensor_square_string(tens), tens, tens)
    K = get_polynomial_bound(tens_squared)
    max_norm = 0
    for i in range(trials):
        # sample uniformly randomly from the sphere with same dimension as out tensor
        guess = np.random.multivariate_normal(np.zeros(len(tens)), np.identity(len(tens)))
        guess = guess / np.linalg.norm(guess)
        tensor_arg_max, tens_norm = MM_iteration(tens_squared, guess, K, 1000, 1e-12)

        res = np.einsum("ijkl,j,k,l", tens_squared, tensor_arg_max, tensor_arg_max, tensor_arg_max)
        res = res/np.linalg.norm(res)

        if max_norm < tens_norm and np.linalg.norm(res-tensor_arg_max) < .0001:
            overall_tensor_arg_max = tensor_arg_max
            max_norm = tens_norm
    if max_norm == 0:
        raise RuntimeError
    return np.sqrt(max_norm), overall_tensor_arg_max

