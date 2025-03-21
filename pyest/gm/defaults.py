__all__ = ['default_gm']

from pyest.gm import GaussianMixture, np

__m = np.array([[10, 5.0, 0.1, 0], [15, 4.2, 0.1, 0.1]])
__w = np.array([0.5, 0.5])
__S = np.array([
    [
        [1, 0.1, 0, 0.01, ],
        [0.1, 0.2, 0.1, 0.1],
        [0., 0, 0.3, 0.01],
        [0, 0, 0, 0.5]
    ],
    [
        [0.5, 0.1, 0, 0.01, ],
        [0.1, 0.3, 0.1, 0.1],
        [0., 0, 0.3, 0.01],
        [0, 0, 0, 0.5]
    ],
])
__P = np.array([S@S.T for S in __S])


def default_gm(mean_shift=None, covariance_rotation=None):
    """ create a default 4D GM for testing, etc.

    Parameters
    ----------
    mean_shift: ndarray, optional
        (nx,) translation to apply to default GM. If mean_shift is None, the
        default GM is not translated.
    covariance_rotation: float, optional
        angle in [rad] to rotate default GM covariance matrices by. If
        covariance_rotation is None, no rotation is applied.

    Returns
    -------
    GaussianMixture
    """
    if mean_shift is not None:
        m = __m + mean_shift
    else:
        m = __m

    if covariance_rotation is not None:
        cos_a = np.cos(covariance_rotation)
        sin_a = np.sin(covariance_rotation)
        dcm = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, cos_a, -sin_a],
            [0, 0, sin_a, cos_a]
        ])
        P = np.array([dcm @ P @ dcm.T for P in __P])
    else:
        P = __P
    return GaussianMixture(__w, m, P)


if __name__ == "__main__":
    print('----Generating and plotting default Gaussian mixtures----')
    import matplotlib.pyplot as plt
    p, XX, YY = default_gm().pdf_2d(dimensions=(0, 1))
    plt.figure()
    plt.contourf(XX, YY, p)
    plt.axis('equal')
    plt.title('Default GM')

    p, XX, YY = default_gm(covariance_rotation=np.pi /
                           2).pdf_2d(dimensions=(0, 1))
    plt.figure()
    plt.contourf(XX, YY, p)
    plt.axis('equal')
    plt.title('Default GM with covariances rotated 90 deg')

    p, XX, YY = default_gm(mean_shift=np.array(
        [-10, 0, 0, 0])).pdf_2d(dimensions=(0, 1))
    plt.figure()
    plt.contourf(XX, YY, p)
    plt.axis('equal')
    plt.title('Default GM with mean shifted 10 to the left')
    plt.show()
