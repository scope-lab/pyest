import matplotlib.pyplot as plt
import numpy as np
import pyest.gm as pygm

# Initialize parameters of a GMM with 2 mixands
weights = [0.4, 0.6]
means = [[0, 0], [10, 10]]
covariances = np.array([[[16, 0], [0, 1]], [[16, 0], [0, 1]]])

# Create a GMM in PyEst
pyest_gmm = pygm.GaussianMixture(weights, means, covariances)

# Plot initial GMM
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax1.set_xlim(-10, 20)
ax1.set_ylim(-5, 15)
ax2.set_xlim(-10, 20)
ax2.set_ylim(-5, 15)

p, X, Y = pyest_gmm.pdf_2d(dimensions=[0,1], res=100)
ax1.contour(X, Y, p)
ax1.set_title('Initial GMM')

# Use PyEst split_gaussian to split the Gaussian mixture
n = 1                                                           # Select the index of the mixand to split
split_options = pygm.GaussSplitOptions(L=3, lam=0.1)            # Set number of mixands to split into and lambda parameter
split_mixand = pyest_gmm.get_comp(n)                            # Get mixand to split
split_comp = pygm.split_gaussian(*split_mixand, split_options)  # Perform split
pyest_gmm += split_comp                                         # Update PyEst GMM to contain the new split mixands
pyest_gmm.pop(n)                                                # Remove the old mixand that was just split

# Loop over GMMs and plot each mixand contour onto the axis
for i in range(len(pyest_gmm)):
    gmm = pygm.GaussianMixture(pyest_gmm[i][0], pyest_gmm[i][1], pyest_gmm[i][2])
    p, X, Y = gmm.pdf_2d(dimensions=[0,1], res=100)
    ax2.contour(X, Y, p)
ax2.set_title('Split GMM')
fig.tight_layout()
plt.show()
