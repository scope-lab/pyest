# %%
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
# %%
import pyest
from pyest import gm
from pyest.sensors.defaults import default_poly_fov

# %%
p = gm.defaults.default_gm(covariance_rotation=np.pi / 6)
fov = default_poly_fov()

# %%
# compute the unnormalized posterior numerically for comparison
pp, XX, YY = p.pdf_2d(dimensions=(0, 1), res=400)
# find which points are inside fov
in_mask = fov.contains(np.vstack((XX.flatten(), YY.flatten())).T)
in_mask_mat = np.reshape(in_mask, XX.shape)
# set pdf evaluations inside fov to zero
pp[in_mask_mat] = 0

# %%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(XX, YY, pp)
plt.title("Example 1: True Posterior Density")

# %%
split_opts = gm.GaussSplitOptions(L=3, lam=1e-3, min_weight=1e-2)
p_split = gm.split_for_fov(p, fov, split_opts)

# compute the sum of the weights of components outside the FoV
comp_mask_in_fov = np.array(fov.contains(p_split.m[:, :2]))


# %%
p_oofov = gm.GaussianMixture(*p_split[~comp_mask_in_fov])

# %%
pp, xx, yy = p_oofov.pdf_2d(dimensions=(0, 1), res=400)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(xx, yy, pp, levels=150)
ax.plot(*fov.polygon.exterior.xy)
plt.title("Example 1: Split Density")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(xx, yy, pp, levels=150)
ax.plot(p_split.m[:, 0], p_split.m[:, 1], '.')
ax.plot(*fov.polygon.exterior.xy)
plt.title("Example 1: Split Density w/ Mixand Mean Locations")

# %%
# -------- Example 2: Two FoVs, pD=0.9 -------------
# create a cone FoV
r = 10
alpha = np.pi / 6
arcx = r * np.cos(np.pi / 2 + np.linspace(-alpha / 2, alpha / 2))
arcy = r * np.sin(np.pi / 2 + np.linspace(-alpha / 2, alpha / 2))
fov_verts = np.vstack(([0, 0], np.vstack((arcx, arcy)).T))
fov = pyest.sensors.PolygonalFieldOfView(fov_verts)

# create a second cone FoV
fov_rot_ang = np.pi / 8
fov_disp = np.array([3, 2])
dcm = np.array([[np.cos(fov_rot_ang), -np.sin(fov_rot_ang)],
                [np.sin(fov_rot_ang), np.cos(fov_rot_ang)]])
fov2 = pyest.sensors.PolygonalFieldOfView(fov_disp + (dcm @ fov_verts.T).T)

# %%
# create a new distribution
m = np.array([0, 5])
cov_ang = np.pi / 5
# rotate the covariance
Py = np.diag([3, 1]) ** 2
dcm = np.array([[np.cos(cov_ang), -np.sin(cov_ang)],
                [np.sin(cov_ang), np.cos(cov_ang)]])
P = dcm @ Py @ dcm.T
w = 1
p_simple = gm.GaussianMixture(w, m, P)

# %%
# plot limits
xlim = [-5, 5]
ylim = [-1, 11]

# %%
pp, xx, yy = p_simple.pdf_2d(dimensions=(0, 1), res=400, xbnd=xlim, ybnd=ylim)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(xx, yy, pp, levels=150)
ax.plot(*fov.polygon.exterior.xy, color='w')
ax.plot(*fov2.polygon.exterior.xy, color='w')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal', adjustable='box')
ax.tick_params(
    axis='both',  # changes apply to both axes
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,  # ticks along the left edge are off
    right=False,  # ticks along the right edge are off
    labelbottom=False,  # labels along the bottom edge are off
    labelleft=False)  # labels along the left edge are off
plt.savefig('FigConeFovDensity.png', dpi=300, transparent=True, bbox_inches='tight')
plt.title("Example 2: Prior Density")

# %%
# split the density along the FoV bounds
p_simple_split = gm.split_for_fov(p_simple, [fov, fov2], split_opts)

# specify a probability of detection
pD = 0.9
# update pdf as if no detection inside FoV
p_simp_upd = deepcopy(p_simple_split)
p_simp_upd.w[fov.contains(p_simp_upd.m)] *= (1 - pD)
p_simp_upd.w[fov2.contains(p_simp_upd.m)] *= (1 - pD)

# %%
pp, xx, yy = p_simp_upd.pdf_2d(dimensions=(0, 1), res=400, xbnd=xlim, ybnd=ylim)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(xx, yy, pp, levels=150)
ax.plot(*fov.polygon.exterior.xy, color='w')
ax.plot(*fov2.polygon.exterior.xy, color='w')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal', adjustable='box')
ax.tick_params(
    axis='both',  # changes apply to both axes
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    left=False,  # ticks along the left edge are off
    right=False,  # ticks along the right edge are off
    labelbottom=False,  # labels along the bottom edge are off
    labelleft=False)  # labels along the left edge are off
plt.savefig('FigConeFovDensitySplitUpdated.png', dpi=300, transparent=True, bbox_inches='tight')
plt.title("Example 2: Split and Updated Posterior, pD=0.9")
plt.show()

# %%
