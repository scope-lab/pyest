# %%
import numpy as np
from pyest import gm, filters
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

"""
This example runs a subset of the PyEst filters on a nonlinear, discrete-time
filtering problem. It demonstrates the use of these filters, and compares them
through their estimation error and uncertainty.

Considering filtering for a nearly-coordinated turn model with range
and angle measurements
Example from "Estimation with Applications to Tracking and Navigation"
by Bar-Shalom, Li, and Kirubarajan
Sections 11.7.1-11.7.2
"""

# %% SYSTEM DEFINITION
"""
State given by x = [xi, xi_dot, eta, eta_dot, omega]
xi : x-coordinate Cartesian position
xi_dot : x-coordinate Cartesian velocity
eta : y-coordinate Cartesian position
eta_dot : y-coordinate Cartesian velocity
omega : turn rate

System dynamics are nonlinear and discrete-time
x(k) = f(x(k-1)) + M @ q(k-1)
f() : nonlinear dynamics function
q(k-1) : zero-mean white Gaussian process noise with covariance Q(k-1)
M : process noise mapping matrix

System measurements are nonlinear range and bearing angle from the origin
z(k) = h(x(k)) + v(k)
z(k) : measurement vector [rho, theta]
rho : range of target from origin
theta : bearing angle of target w.r.t. origin
h() : nonlinear measurement function
v(k) : zero-mean white Gaussian measurement noise with covariance R(k)
"""


# Discrete-time dynamics function
def f(tkm1, tk, xkm1):
    dt = tk - tkm1
    xi_km1, xi_dot_km1, eta_km1, eta_dot_km1, omega_km1 = xkm1
    if omega_km1 == 0:
        # If omega is 0, then it's just a constant velocity model
        A = np.array([
            [1, dt, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, dt, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
    else:
        # Rows of matrix
        A1 = np.array([1, np.sin(omega_km1*dt)/omega_km1, 0,
                       -(1-np.cos(omega_km1*dt)/omega_km1), 0])
        A2 = np.array([0, np.cos(omega_km1*dt), 0, -np.sin(omega_km1*dt), 0])
        A3 = np.array([0, (1-np.cos(omega_km1*dt))/omega_km1, 1,
                       np.sin(omega_km1*dt)/omega_km1, 0])
        A4 = np.array([0, np.sin(omega_km1*dt), 0, np.cos(omega_km1*dt), 0])
        A5 = np.array([0, 0, 0, 0, 1])
        A = np.array([A1, A2, A3, A4, A5])

    xk = A @ xkm1
    return xk


# Discrete-time dynamics Jacobian
def F(tkm1, tk, xkm1):
    dt = tk - tkm1
    xi_km1, xi_dot_km1, eta_km1, eta_dot_km1, omega_km1 = xkm1
    if omega_km1 == 0:
        # Must use limit of function if omega = 0
        A = np.array([
            [1, dt, 0, 0, -0.5*dt**2*eta_dot_km1],
            [0, 1, 0, 0, -dt*eta_dot_km1],
            [0, 0, 1, dt, 0.5*dt**2*xi_dot_km1],
            [0, 0, 0, 1, dt*xi_dot_km1],
            [0, 0, 0, 0, 1]
        ])
    else:
        f_omega_1 = np.cos(omega_km1*dt)*dt*xi_dot_km1/omega_km1 \
            - np.sin(omega_km1*dt)*xi_dot_km1/omega_km1**2 \
            - np.sin(omega_km1*dt)*dt*eta_dot_km1/omega_km1 \
            - (-1+np.cos(omega_km1*dt))*eta_dot_km1/omega_km1**2
        f_omega_2 = -np.sin(omega_km1*dt)*dt*xi_dot_km1 \
            - np.cos(omega_km1*dt)*dt*eta_dot_km1
        f_omega_3 = np.sin(omega_km1*dt)*dt*xi_dot_km1/omega_km1 \
            - (1-np.cos(omega_km1*dt))*xi_dot_km1/omega_km1**2 \
            + np.cos(omega_km1*dt)*dt*eta_dot_km1/omega_km1 \
            - np.sin(omega_km1*dt)*eta_dot_km1/omega_km1**2
        f_omega_4 = np.cos(omega_km1*dt)*dt*xi_dot_km1 \
            - np.sin(omega_km1*dt)*dt*eta_dot_km1

        # Rows of matrix
        A1 = np.array([1, np.sin(omega_km1)*dt/omega_km1, 0,
                       -(1-np.cos(omega_km1*dt))/omega_km1,
                       f_omega_1])
        A2 = np.array([0, np.cos(omega_km1*dt), 0, -np.sin(omega_km1*dt),
                       f_omega_2])
        A3 = np.array([0, (1-np.cos(omega_km1*dt))/omega_km1, 1,
                       np.sin(omega_km1*dt)/omega_km1, f_omega_3])
        A4 = np.array([0, np.sin(omega_km1*dt), 0, np.cos(omega_km1*dt),
                       f_omega_4])
        A5 = np.array([0, 0, 0, 0, 1])

        A = np.array([A1, A2, A3, A4, A5])

    return A


# Discrete-time process noise covariance
def Q(tkm1, tk, xkm1):
    return (tk-tkm1)*np.diag(np.array([0.01, 0.01, 0.001])**2)


# Discrete-time process noise mapping matrix
def M(tkm1, tk, xkm1):
    dt = tk - tkm1
    return np.array([
        [0.5*dt**2, 0, 0],
        [dt, 0, 0],
        [0, 0.5*dt**2, 0],
        [0, dt, 0],
        [0, 0, dt]
    ])


# Measurement function
def h(xk):
    xi, xi_dot, eta, eta_dot, omega = xk

    rho = np.sqrt(xi**2 + eta**2)
    theta = np.atan2(eta, xi)

    return np.array([rho, theta])


# Measurement Jacobian
def H(xk):
    xi, xi_dot, eta, eta_dot, omega = xk

    rho = np.sqrt(xi**2 + eta**2)
    drho_dx = np.array([xi/rho, 0, eta/rho, 0, 0])

    dtheta_dx = np.array([-eta/rho**2, 0,
                          xi/rho**2, 0, 0])

    return np.array([drho_dx, dtheta_dx])


# Measurement noise covariance
R = np.diag(np.array([100, 1*np.pi/180])**2)


# %% TRUTH PROPAGATION
t0 = 0  # Initial time
tf = 100  # Final time
dt = 1  # Time between measurements
tv = np.arange(t0, tf+dt, dt)  # Array of time values
nt = len(tv)

# Pre-allocate memory for storing truth over time
nx = 5
x_hist = np.full((nt, nx), np.nan)
x0 = np.array([25_000, 100, 10_000, 0, 10*np.pi/180])  # Initial state
x_hist[0] = x0

for k in range(1, nt):
    tkm1 = tv[k-1]
    tk = tv[k]
    xkm1 = x_hist[k-1]

    # Propagating according to dynamics model
    Mkm1 = M(tkm1, tk, xkm1)
    Qkm1 = Q(tkm1, tk, xkm1)
    xk = f(tkm1, tk, xkm1) \
        + rng.multivariate_normal(np.zeros((nx,)), Mkm1@Qkm1@Mkm1.T)

    x_hist[k] = xk

# Plotting true path of target
plt.figure()
plt.plot(x_hist[:, 0], x_hist[:, 2], color='xkcd:darkgreen')
plt.scatter(x_hist[0, 0], x_hist[0, 2], s=20, color='xkcd:black')
plt.xlabel(r"$\xi$ [m]")
plt.ylabel(r"$\eta$ [m]")
plt.title("True Trajectory")
plt.show()

# %% MEASUREMENT GENERATION

# Pre-allocate memory for storing measurements over time
nz = 2
z_hist = np.full((nt, nz), np.nan)

for k in range(1, nt):
    xk = x_hist[k]

    # Drawing random measurement according to measurement model
    zk = h(xk) + rng.multivariate_normal(np.zeros((nz,)), R)

    z_hist[k] = zk

# Plotting measurements
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Range
ax1.plot(tv[1:], z_hist[1:, 0],
         marker='x', linestyle='None', color='xkcd:red')
ax1.set_ylabel(r'Range, $\rho$')
ax1.grid()

# Bearing
ax2.plot(tv[1:], np.rad2deg(z_hist[1:, 1]),
         marker='x', linestyle='None', color='xkcd:red')
ax2.set_ylabel(r'Bearing, $\theta$ [deg]')
ax2.set_xlabel('Time [s]')
ax2.grid()
plt.suptitle("Measurement History")
plt.tight_layout()
plt.show()

# %% FILTERING PARAMETERS
# Initial conditions
P0 = np.diag(np.array([1_000, 10, 1_000, 10, 1*np.pi/180])**2)  # Initial cov
m0 = rng.multivariate_normal(x0, P0)  # initial mean

# Initial mixture (single mixand)
p0 = gm.GaussianMixture(w=np.array([1]), m=m0, cov=P0, cov_type='full')

# Splitting parameters for GM filters
# The GaussSplitOptions object stores parameters for recursive Gaussian
# splitting
split_opts = gm.GaussSplitOptions(L=3, recurse_depth=2)

"""
Storing arguments that will be passed to gm.split.recursive_split() in order
to split the Gaussian mixture for the nonlinear update function.

For more information on PyEst's split heuristics, read
Jackson Kulik and Keith A. LeGrand, "Nonlinearity and Uncertainty
Informed Moment-Matching Gaussian Mixture Splitting,"
https://arxiv.org/abs/2412.00343, 2024

USFOS requires the Jacobian and a split tolerance. The Jacobian is packaged
in the lambda function since the split heurisitics expect the Jacobian to take
each state variable as a separate parameter. The -np.inf tolerance ensures all
components are being split.

As a note, if the Jacobian is time-dependent or the time steps aren't constant,
then the split arguments need to be redefined in each filter iteration to
account for this dependence.
"""
recursive_split_args_update = (gm.split.id_usfos,
                               lambda *xk: H(np.asarray(xk)),
                               -np.inf)
recursive_split_args_predict = (gm.split.id_usfos,
                                lambda *xk: F(tv[0], tv[1], np.asarray(xk)),
                                -np.inf)

# Reduction parameters for GM filters
# K is the number of mixands the mixture is reduced to following prediction
# and update. This reduction is necessary to limit the number of mixands
# following splitting
K = 5

# UKF parameters
alpha = 0.1
beta = 2
kappa = 0

# SigmaPointOptions is a PyEst class for storing sigma point parameters
sigma_pt_opts = filters.SigmaPointOptions(
    alpha=alpha, beta=beta, kappa=kappa)

# This will store which filters are being used
filter_names = []

# These will store the estimated means and sigma bounds for each filter
mkm_hist_filters = dict()
mkp_hist_filters = dict()
Pkm_hist_filters = dict()
Pkp_hist_filters = dict()

# %% EKF FILTERING
filter_names.append("EKF")

ekf_predictor = filters.EkfdPredict(f=f, F=F, Q=Q, M=M)
ekf_updater = filters.EkfdUpdate(h=h, H=H, R=R)

# Pre-allocating memory to store mean and covariance
mkm_hist_ekf = np.full((nt, nx), np.nan)
mkm_hist_ekf[0] = m0
mkp_hist_ekf = np.full((nt, nx), np.nan)
mkp_hist_ekf[0] = m0
Pkm_hist_ekf = np.full((nt, nx, nx), np.nan)
Pkm_hist_ekf[0] = P0
Pkp_hist_ekf = np.full((nt, nx, nx), np.nan)
Pkp_hist_ekf[0] = P0

for k in range(1, nt):
    tkm1 = tv[k-1]
    tk = tv[k]
    mkm1 = mkp_hist_ekf[k-1]
    Pkm1 = Pkp_hist_ekf[k-1]
    zk = z_hist[k]

    mkm, Pkm = ekf_predictor.predict((tkm1, tk), mkm1, Pkm1)
    mkm_hist_ekf[k] = mkm
    Pkm_hist_ekf[k] = Pkm

    mkp, Pkp = ekf_updater.update(mkm, Pkm, zk)
    mkp_hist_ekf[k] = mkp
    Pkp_hist_ekf[k] = Pkp

mkm_hist_filters["EKF"] = mkm_hist_ekf
mkp_hist_filters["EKF"] = mkp_hist_ekf
Pkm_hist_filters["EKF"] = Pkm_hist_ekf
Pkp_hist_filters["EKF"] = Pkp_hist_ekf

# %% UKF FILTERING
filter_names.append("UKF")

ukf_predictor = filters.UkfPredict(
    f=lambda xkm1, tkm1, tk: f(tkm1, tk, xkm1),
    Q=lambda tkm1, tk: M(tkm1, tk, m0)@Q(tkm1, tk, m0)@M(tkm1, tk, m0).T,
    M=None,
    sigma_pt_opts=sigma_pt_opts
)
ukf_updater = filters.UkfUpdate(h=h, R=R, sigma_pt_opts=sigma_pt_opts)

# Pre-allocating memory to store mean and covariance
mkm_hist_ukf = np.full((nt, nx), np.nan)
mkm_hist_ukf[0] = m0
mkp_hist_ukf = np.full((nt, nx), np.nan)
mkp_hist_ukf[0] = m0
Pkm_hist_ukf = np.full((nt, nx, nx), np.nan)
Pkm_hist_ukf[0] = P0
Pkp_hist_ukf = np.full((nt, nx, nx), np.nan)
Pkp_hist_ukf[0] = P0

for k in range(1, nt):
    tkm1 = tv[k-1]
    tk = tv[k]
    mkm1 = mkp_hist_ukf[k-1]
    Pkm1 = Pkp_hist_ukf[k-1]
    zk = z_hist[k]

    mkm, Pkm = ukf_predictor.predict((tkm1, tk), mkm1, Pkm1)
    mkm_hist_ukf[k] = mkm
    Pkm_hist_ukf[k] = Pkm

    mkp, Pkp = ukf_updater.update(mkm, Pkm, zk)
    mkp_hist_ukf[k] = mkp
    Pkp_hist_ukf[k] = Pkp

mkm_hist_filters["UKF"] = mkm_hist_ukf
mkp_hist_filters["UKF"] = mkp_hist_ukf
Pkm_hist_filters["UKF"] = Pkm_hist_ukf
Pkp_hist_filters["UKF"] = Pkp_hist_ukf

# %% GMEKF FILTERING
filter_names.append("GMEKF")

gmekf_predictor = filters.GmekfPredict(f=f, F=F, Q=Q, M=M)
gmekf_updater = filters.GmekfUpdate(h=h, H=H, R=R)

# Creating dictionaries to store densities
pkm_hist_gmekf = dict()
pkm_hist_gmekf[0] = p0
pkp_hist_gmekf = dict()
pkp_hist_gmekf[0] = p0

for k in range(1, nt):
    tkm1 = tv[k-1]
    tk = tv[k]
    pkm1 = pkp_hist_gmekf[k-1]
    zk = z_hist[k]

    pkm1 = gm.recursive_split(pkm1, split_opts, *recursive_split_args_predict)
    pkm = gmekf_predictor.predict((tkm1, tk), pkm1)
    pkm = gm.merge_runnalls(pkm, K)
    pkm_hist_gmekf[k] = pkm

    pkm = gm.recursive_split(pkm, split_opts, *recursive_split_args_update)
    pkp = gmekf_updater.update(pkm, zk)
    pkp = gm.merge_runnalls(pkp, K)
    pkp_hist_gmekf[k] = pkp

mkm_hist_gmekf = np.array([p.mean() for _, p in pkm_hist_gmekf.items()])
mkp_hist_gmekf = np.array([p.mean() for _, p in pkp_hist_gmekf.items()])
Pkm_hist_gmekf = np.array([p.cov() for _, p in pkm_hist_gmekf.items()])
Pkp_hist_gmekf = np.array([p.cov() for _, p in pkp_hist_gmekf.items()])

mkm_hist_filters["GMEKF"] = mkm_hist_gmekf
mkp_hist_filters["GMEKF"] = mkp_hist_gmekf
Pkm_hist_filters["GMEKF"] = Pkm_hist_gmekf
Pkp_hist_filters["GMEKF"] = Pkp_hist_gmekf

# %% GMUKF FILTERING
filter_names.append("GMUKF")

gmukf_predictor = filters.GmukfPredict(
    f=lambda xkm1, tkm1, tk: f(tkm1, tk, xkm1),
    Q=lambda tkm1, tk: M(tkm1, tk, m0)@Q(tkm1, tk, m0)@M(tkm1, tk, m0).T,
    M=None,
    sigma_pt_opts=sigma_pt_opts
)
gmukf_updater = filters.GmukfUpdate(h=h, R=R, sigma_pt_opts=sigma_pt_opts)

# Creating dictionaries to store densities
pkm_hist_gmukf = dict()
pkm_hist_gmukf[0] = p0
pkp_hist_gmukf = dict()
pkp_hist_gmukf[0] = p0

for k in range(1, nt):
    tkm1 = tv[k-1]
    tk = tv[k]
    pkm1 = pkp_hist_gmukf[k-1]
    zk = z_hist[k]

    pkm1 = gm.recursive_split(pkm1, split_opts, *recursive_split_args_predict)
    pkm = gmukf_predictor.predict((tkm1, tk), pkm1)
    pkm = gm.merge_runnalls(pkm, K)
    pkm_hist_gmukf[k] = pkm

    pkm = gm.recursive_split(pkm, split_opts, *recursive_split_args_update)
    pkp = gmukf_updater.update(pkm, zk)
    pkp = gm.merge_runnalls(pkp, K)
    pkp_hist_gmukf[k] = pkp

mkm_hist_gmukf = np.array([p.mean() for _, p in pkm_hist_gmukf.items()])
mkp_hist_gmukf = np.array([p.mean() for _, p in pkp_hist_gmukf.items()])
Pkm_hist_gmukf = np.array([p.cov() for _, p in pkm_hist_gmukf.items()])
Pkp_hist_gmukf = np.array([p.cov() for _, p in pkp_hist_gmukf.items()])

mkm_hist_filters["GMUKF"] = mkm_hist_gmukf
mkp_hist_filters["GMUKF"] = mkp_hist_gmukf
Pkm_hist_filters["GMUKF"] = Pkm_hist_gmukf
Pkp_hist_filters["GMUKF"] = Pkp_hist_gmukf

# %% PLOTTING ALL FILTERS
state_names = [r"$\xi$ [m]",
               r"$\dot{\xi}$ [m/s]",
               r"$\eta$ [m]",
               r"$\dot{\eta}$ [m/s]",
               r"$\Omega$ [deg/s]"]

# Plotting estimation error and uncertainty
for filter_name in filter_names:
    # Grabbing estimates for the filter
    mkm_hist = mkm_hist_filters[filter_name]
    mkp_hist = mkp_hist_filters[filter_name]
    Pkm_hist = Pkm_hist_filters[filter_name]
    Pkp_hist = Pkp_hist_filters[filter_name]

    # Estimation error
    ekm_hist = x_hist - mkm_hist
    ekp_hist = x_hist - mkp_hist
    ek_hist_zip = np.stack(
        (ekm_hist, ekp_hist), axis=1).reshape(-1, ekm_hist.shape[1])

    # Converting bearing error from rad to deg
    ek_hist_zip[:, 4] *= 180/np.pi

    # Estimation uncertainty per state (in std. dev.)
    sigkm_hist = np.array([np.sqrt(np.diag(P)) for P in Pkm_hist])
    sigkp_hist = np.array([np.sqrt(np.diag(P)) for P in Pkp_hist])
    sigk_hist_zip = np.stack(
        (sigkm_hist, sigkp_hist), axis=1).reshape(-1, sigkm_hist.shape[1])

    # Converting bearing uncertainty from rad to deg
    sigk_hist_zip[:, 4] *= 180/np.pi

    tv_zip = np.sort(np.concatenate((tv, tv)))

    # Plotting error and uncertainty for each state
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1, figsize=(8, 8), sharex=True)

    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
        ax.plot(tv_zip, ek_hist_zip[:, i], color="xkcd:red",
                label="Estimation Error")
        ax.plot(tv_zip, 3*sigk_hist_zip[:, i], color="xkcd:blue",
                label=r"3$\sigma$ Uncertainty")
        ax.plot(tv_zip, -3*sigk_hist_zip[:, i], color="xkcd:blue")
        ax.set_xlim(np.min(tv), np.max(tv))
        ax.set_ylabel(state_names[i])

    ax1.legend(fontsize=8)
    ax5.set_xlabel("Time [s]")
    fig.suptitle(filter_name)
    fig.tight_layout()

# %%
