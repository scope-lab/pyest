import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from STMint.STMint import STMint
from diskcache import Cache

import pyest.gm as pygm
import pyest.gm.split as split
from pyest.filters.sigma_points import SigmaPointOptions
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
    labels = ['$x$', '$y$', '$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
    plt.xlabel(labels[dims[0]])
    plt.ylabel(labels[dims[1]])

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

    labels = ['$x$', '$y$', '$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
    ax.set_xlabel(labels[dims[0]])
    ax.set_ylabel(labels[dims[1]])

    save_figure(example, split_method_str + '_' +
                str(dims[0]) + '_' + str(dims[1]), plt.gca(), plt.gcf())

    pp, XX, YY = py.pdf_2d(res=300, dimensions=dims)
    return pp, XX, YY
# end plotting utilities


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


# density propagation example in a Cislunar NRHO
def cislunar_example():
    example = 'cislunar'
    # nrho ics
    mu = 1.0 / (81.30059 + 1.0)
    x0 = 1.02202151273581740824714855590570360
    z0 = -0.182096761524240501132977765539282777
    yd0 = -0.103256341062793815791764364248006121
    period = 1.5111111111111111111111111111111111111111
    transfer_time = period * 0.5

    x_0 = np.array([x0, 0, z0, 0, yd0, 0])
    ny = 6  # dimension of y
    nx = 6  # dimension of x

    weights = np.array([1])  # single component
    cov_0 = 0.00001**2 * np.identity(6) + \
        0.0001**2 * (np.diag([1, 0, 1, 0, 0, 0]))
    p0 = GaussianMixture(weights, np.array([x_0]), np.array([cov_0]))

    # nrho propagator
    integrator = STMint(preset="threeBody", preset_mult=mu,
                        variational_order=2)
    max_integrator_step = period/500.0
    int_tol = 1e-13

    # outputs x_f, STM, STT
    def flow_info(x, y, z, vx, vy, vz): return integrator.dynVar_int2(
        [0, transfer_time], [x, y, z, vx, vy, vz], rtol=int_tol, atol=int_tol, output="final"
    )
    # outputs just the hessian

    def hessian_func(x, y, z, vx, vy, vz): return integrator.dynVar_int2(
        [0, transfer_time], [x, y, z, vx, vy, vz], rtol=int_tol, atol=int_tol, output="final"
    )[2]
    # outputs just the jacobian

    def jacobian_func(x, y, z, vx, vy, vz): return integrator.dynVar_int(
        [0, transfer_time], [x, y, z, vx, vy, vz], rtol=int_tol, atol=int_tol, output="final"
    )[1]
    # outputs flow of state only

    def propagation(x_0): return integrator.dyn_int([0, transfer_time], x_0,
                                                    max_step=max_integrator_step,
                                                    t_eval=[transfer_time]).y[:, -1]

    # apply splitting methods
    split_opts = pygm.GaussSplitOptions(
        L=3, lam=1e-3, recurse_depth=3, min_weight=-np.inf)

    # Define the unscented transform parameters
    sigma_pt_opts = SigmaPointOptions(alpha=1e-3, beta=2, kappa=0)

    print("running monte carlo")
    # create/load split cache
    cislunar_mc_cache = Cache(__file__[:-3] + 'cislunar_mc_cache')
    # reference Monte Carlo (store points and pdf value at point)
    num_points = int(1e4)
    rng = np.random.default_rng(100)
    if 'samples' in cislunar_mc_cache:
        print("cache found, loading samples from cache")
        samples = cislunar_mc_cache['samples']
        final_samples = cislunar_mc_cache['final_samples']
        assert (len(samples) == num_points)
    else:
        print("cache not found, propagating samples")
        samples = rng.multivariate_normal(x_0, cov_0, num_points)
        final_samples = list(map(propagation, samples))
        cislunar_mc_cache['samples'] = samples
        cislunar_mc_cache['final_samples'] = final_samples
        print("MC propagation complete, cache saved with {} samples".format(num_points))

    idx_pairs = [(0, 1), (0, 2), (1, 2), (3, 4),
                 (4, 5), (3, 5), (0, 4), (1, 3)]
    axis_labels = ['$x$', '$y$', '$z$',
                   r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']

    # scatter plotting for Monte Carlo
    xlim = dict()
    ylim = dict()
    for idx_pair in idx_pairs:
        plt.figure()
        plt.scatter(np.array(final_samples)[:, idx_pair[0]], np.array(
            final_samples)[:, idx_pair[1]], marker='+', alpha=0.025)
        plt.xlabel(axis_labels[idx_pair[0]])
        plt.ylabel(axis_labels[idx_pair[1]])
        save_figure(example, "truth_scatter" + '_' +
                    str(idx_pair[0]) + '_' + str(idx_pair[1]), plt.gca(), plt.gcf())
        plt.figure()
        plt.hist2d(np.array(final_samples)[:, idx_pair[0]], np.array(
            final_samples)[:, idx_pair[1]], 40)
        plt.xlabel(axis_labels[idx_pair[0]])
        plt.ylabel(axis_labels[idx_pair[1]])
        # save axis limits for later
        xlim[idx_pair] = plt.gca().get_xlim()
        ylim[idx_pair] = plt.gca().get_ylim()
        save_figure(example, "truth_hist" + '_' +
                    str(idx_pair[0]) + '_' + str(idx_pair[1]), plt.gca(), plt.gcf())

    recursive_split_args = {}
    # use the same number of recursive splits for each mixand
    split_tol = -np.inf
    # settings for the SADL and ALoDT based metrics
    diff_stat_det_sigma_pt_opts = SigmaPointOptions(
        alpha=0.5)  # spread sigma points farther

    # define parameters associated with each splitting method
    recursive_split_args['variance'] = (split.id_variance, split_tol)
    recursive_split_args['USFOS'] = (split.id_usfos, jacobian_func, split_tol)
    recursive_split_args['WUSSADL'] = (
        split.id_wussadl, jacobian_func, propagation, diff_stat_det_sigma_pt_opts, split_tol)
    recursive_split_args['WUSSOLC'] = (
        split.id_wussolc, hessian_func, jacobian_func, split_tol)

    # additional splitting methods
    # uncomment these if desired
    # recursive_split_args['ALoDT'] = (split.id_max_alodt, propagation, diff_stat_det_sigma_pt_opts, split_tol)
    # recursive_split_args['FOS'] = (split.id_fos, jacobian_func, split_tol)
    # recursive_split_args['SAFOS'] = (split.id_safos, jacobian_func, split_tol)
    # recursive_split_args['USFOS'] = (split.id_usfos, jacobian_func, split_tol)
    # recursive_split_args['SOS'] = (split.id_sos, hessian_func, jacobian_func, split_tol)
    # recursive_split_args['SASOS'] = (split.id_sasos, hessian_func, split_tol)
    # recursive_split_args['WSASOS'] = (split.id_wsasos, hessian_func, jacobian_func, split_tol)
    # recursive_split_args['WUSSOS'] = (split.id_wussos, hessian_func, jacobian_func, split_tol)
    # recursive_split_args['SOLC'] = (split.id_solc, hessian_func, split_tol)
    # recursive_split_args['USSOLC'] = (split.id_ussolc, hessian_func, split_tol)
    # recursive_split_args['SADL'] = (split.id_sadl, jacobian_func, propagation, diff_stat_det_sigma_pt_opts, split_tol)

    # plot the resulting GMM densities propagated
    for split_method, args in recursive_split_args.items():
        p_split = split.recursive_split(p0, split_opts, *args)
        py = transform_density_ekf(p_split, ny, propagation, jacobian_func)

        for idx_pair in idx_pairs:
            _, XX, YY = plot_split_and_transformed(
                p_split, py, split_method, example, idx_pair, xf_lim=xlim[idx_pair], yf_lim=ylim[idx_pair])

    plt.show()


if __name__ == '__main__':
    # run the example
    cislunar_example()
