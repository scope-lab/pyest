# example_2d_gm_split.py
# Written by Keith LeGrand, April 2020
import numpy as np
import matplotlib.pyplot as plt
from pyest import gm

# plot a 2D, 3 component Gaussian mixture
N = 3
# weights
w = np.array([0.4, 0.3, 0.3])
# means
m = np.array([[-0.3, 0.4], [0.1, 1.2], [0.5, 0.4]])
# covariances
P = np.tile(np.diag([0.3**2, 0.2**2]), (N, 1, 1))

# form the Gaussian mixture
gmm = gm.GaussianMixture(w, m, P)
# sample GM on 100x100 grid for plotting
p, X, Y = gmm.pdf_2d(dimensions=[0, 1], res=100)

plt.figure()
plt.contourf(X, Y, p)
# plot the sigma contours of the components
for w, m, P in gmm:
    XY = gm.sigma_contour(m, P, sig_mul=2)
    plt.plot(XY[:, 0], XY[:, 1])

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

# split the first GM component into 3 smaller components
split_options = gm.GaussSplitOptions(L=3, lam=0.001)
split_comp = gm.split_gaussian(*gmm.pop(0), split_options)
gmm += split_comp

# sample GM on 100x100 grid for plotting
p, X, Y = gmm.pdf_2d(dimensions=[0, 1], res=100)

plt.figure()
plt.contourf(X, Y, p)

# plot the sigma contours of the components
for w, m, P in gmm:
    XY = gm.sigma_contour(m, P, sig_mul=2)
    plt.plot(XY[:, 0], XY[:, 1])

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
