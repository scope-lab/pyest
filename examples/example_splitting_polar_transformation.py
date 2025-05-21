import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import pyest.gm as pygm
import pyest.gm.split as split
from pyest.filters.sigma_points import SigmaPointOptions, unscented_transform
from pyest.gm import GaussianMixture
from pyest.linalg import triangularize


# plotting functions
mpl.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})


def bounds_from_meshgrids(XX1, YY1, XX2, YY2):
    x_max = np.max(np.concatenate([XX1.ravel(), XX2.ravel()]))
    x_min = np.min(np.concatenate([XX1.ravel(), XX2.ravel()]))
    y_max = np.max(np.concatenate([YY1.ravel(), YY2.ravel()]))
    y_min = np.min(np.concatenate([YY1.ravel(), YY2.ravel()]))
    return x_min, x_max, y_min, y_max


def save_figure(example, split_method, ax, fig, w=3, h=3):
    # save title text before clearing the title
    title_text = ax.get_title()
    ax.set_title('')
    fig.set_size_inches(w=w, h=h)
    filename = example + split_method.replace(' ', '_')
    fig.savefig(filename + '.svg', bbox_inches='tight', pad_inches=0)
    ax.set_title(title_text)


def plot_split_and_transformed(p_split, py, split_method_str, example, dims=(0, 1),
                               scatter_means=True, xf_lim=None, yf_lim=None, ax_equal=False):
    num_contours = 100
    scatter_plt_args = {'marker': 'x', 'zorder': 2, 'color': 'k'}
    scatter_plt_overlay_args = {'s': 5**2, 'marker': 'x',
                                'zorder': 2.1, 'color': 'w', 'alpha': 0.9, 'linewidth': 1}
    # Plot the split density
    pp, XX, YY = p_split.pdf_2d(res=300, dimensions=dims)
    plt.figure()
    plt.contour(XX, YY, pp, num_contours)
    plt.title('Original Density, split,  ' + split_method_str, wrap=True)
    plt.colorbar()
    if scatter_means:
        plt.scatter(p_split.m[:, dims[0]],
                    p_split.m[:, dims[1]], **scatter_plt_args)
        plt.scatter(p_split.m[:, dims[0]], p_split.m[:,
                    dims[1]], **scatter_plt_overlay_args)
    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    save_figure(example, split_method_str + "before_map" +
                str(dims), plt.gca(), plt.gcf())

    # Plot the transformed split density
    pp, XX, YY = py.pdf_2d(res=300, dimensions=dims, xbnd=xf_lim, ybnd=yf_lim)
    fig, ax = plt.subplots()
    c = ax.contour(XX, YY, pp, num_contours, linewidths=0.5)
    fig.colorbar(c)
    ax.set_title('Transformed Density, ' + split_method_str, wrap=True)
    if scatter_means:
        ax.scatter(py.m[:, dims[0]], py.m[:, dims[1]], **scatter_plt_args)
        ax.scatter(py.m[:, dims[0]], py.m[:, dims[1]],
                   **scatter_plt_overlay_args)
    ax.grid()
    if xf_lim is not None:
        ax.set_xlim(xf_lim)
    if yf_lim is not None:
        ax.set_ylim(yf_lim)
    if ax_equal:
        ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('$R$')
    ax.set_ylabel(r'$\theta$')

    save_figure(example, split_method_str + '_' +
                str(dims[0]) + '_' + str(dims[1]), plt.gca(), plt.gcf())

    pp, XX, YY = py.pdf_2d(res=300, dimensions=dims)
    return pp, XX, YY


# square root EKF propagation for individual mixands
def transform_density_ekf(p_split, ny, g, G):
    my = np.zeros((len(p_split), ny))
    Sy = np.zeros((len(p_split), ny, ny))
    for i in range(len(p_split)):
        my[i] = g(p_split.m[i])
        Gval = G(*p_split.m[i])
        Sy[i] = triangularize(Gval @ p_split.Schol[i])

    wy = p_split.w.copy()
    return GaussianMixture(wy, my, Sy, cov_type='cholesky')


# density propagation example in the Cartesian to Polar transformation
def cartesian_to_polar_example():
    example = 'cart2polar'
    # Define the transformation from Cartesian to Polar coordinates
    def cartesian_to_polar(x): return [np.sqrt(
        x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])]
    # Define the transformation from Polar to Cartesian coordinates
    def polar_to_cartesian(y): return [y[0]*np.cos(y[1]), y[0]*np.sin(y[1])]
    ny = 2  # dimension of y
    nx = 2  # dimension of x
    # some limits for integration
    theta_max = 2*np.pi
    theta_min = 0

    # Define the Gaussian mixture
    weights = np.array([1])  # single component
    means = np.array([[0, 1000]])  # 2D in Cartesian coordinates
    covariances = 250**2*np.array([[[16, 0], [0, 1]]])  # covariance matrix
    # we'll use GM even though we only have a single component for generality
    p0 = GaussianMixture(weights, means, covariances)

    # ---- true density plotting ----
    # the true transformed density can be found in terms of the determinant of the inverse mapping
    # can be employed as a reference for GMM propagation
    def py_true(y): return 0 if y[0] < 0 else y[0]*p0(polar_to_cartesian(y))

    # Define the unscented transform parameters
    sigma_pt_opts = SigmaPointOptions(alpha=1e-3, beta=2, kappa=0)
    # Compute the unscented transform of the Gaussian mixture
    mean_polar, covariance_polar, Dt, sigmas, my = unscented_transform(
        p0.m[0], p0.P[0], cartesian_to_polar, sigma_pt_opts=sigma_pt_opts)

    # Create a Gaussian mixture for the transformed density
    p0_polar = GaussianMixture(
        weights, mean_polar, covariance_polar, cov_type='full')

    # Plot the original density
    pp, XX, YY = p0.pdf_2d()
    plt.figure()
    plt.contour(XX, YY, pp, 100)
    plt.title('Original Density')
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # Plot the true transformed density
    # for laziness, use the linear transformation to generate grid points
    _, XX_true, YY_true = p0_polar.pdf_2d()
    pp_true = np.array([py_true([x, y]) for x, y in zip(
        XX_true.ravel(), YY_true.ravel())]).reshape(XX_true.shape)
    fig, ax = plt.subplots()
    c = ax.contour(XX_true, YY_true, pp_true, 100, linewidths=0.5)
    ax.set_title('True Transformed Density')
    ax.set_xlabel('$R$')
    ax.set_ylabel(r'$\theta$')
    fig.colorbar(c)

    # plt.gca().set_aspect('equal', adjustable='box')
    # Grab the limits
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.grid()
    save_figure(example, 'truth', plt.gca(), plt.gcf(), w=3, h=3)

    # GMM splitting based density propagation
    # Define the Cartesian to Polar conversion in sympy
    x, y = sp.symbols('x y')
    r = sp.sqrt(x**2 + y**2)
    theta = sp.atan2(y, x)
    cartesian_to_polar_sym = sp.Matrix([r, theta])

    # Compute the symbolic Jacobian
    jacobian = cartesian_to_polar_sym.jacobian([x, y])

    # Compute the symbolic Hessian
    hessian = [sp.hessian(cartesian_to_polar_sym[i], [x, y])
               for i in range(nx)]

    # Lambdify the Jacobian and Hessian
    jacobian_func = sp.lambdify([x, y], jacobian)
    hessian_func = sp.lambdify([x, y], hessian)

    # compare the different split directions
    split_opts = pygm.GaussSplitOptions(
        L=3, lam=1e-3, recurse_depth=2, min_weight=1e-5)

    recursive_split_args = {}
    # use the same number of recursive splits for each mixand
    split_tol = -np.inf
    # settings for the SADL and ALoDT based metrics
    diff_stat_det_sigma_pt_opts = SigmaPointOptions(
        alpha=0.5)  # spread sigma points farther

    # define parameters associated with each splitting method
    # two equally performing methods
    recursive_split_args['variance'] = (split.id_variance, split_tol)
    recursive_split_args['WUSSOLC'] = (
        split.id_wussolc, hessian_func, jacobian_func, split_tol)
    # a method that does not perform as well in this non-dynamical context
    recursive_split_args['USFOS'] = (split.id_usfos, jacobian_func, split_tol)

    # plot the results for each splitting method
    for split_method, args in recursive_split_args.items():
        p_split = split.recursive_split(p0, split_opts, *args)
        py = transform_density_ekf(
            p_split, ny, cartesian_to_polar, jacobian_func)
        _, XX, YY = plot_split_and_transformed(
            p_split, py, split_method, example, xf_lim=x_lim, yf_lim=y_lim)

    plt.show()


if __name__ == '__main__':
    # run the example
    cartesian_to_polar_example()
