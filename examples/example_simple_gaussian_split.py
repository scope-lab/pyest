import pyest.gm as gm
import matplotlib.pyplot as plt
import numpy as np

# find the optimal standard normal split solutions for different mixture
# sizes and regularization parameter values
L = 5
lam = 1e-3
p = gm.split_1d_standard_gaussian(L, lam)

# we can now plot each of the mixands by using the iterable feature of GaussianMixture
x = np.linspace(-4, 4, 100)
fig, ax = plt.subplots()
for wi, mi, Pi in p:
    ax.plot(x, wi*gm.eval_mvnpdf(x[:, np.newaxis], mi, Pi), linestyle='--')

# plot the GM approximation
ax.plot(x, p(x[:, np.newaxis]), linestyle='-')
ax.set_xlabel('x')
ax.set_ylabel('p(x)')
plt.show()
# # save figure at high resolution with no extra whitespace
# fig.savefig('univariate_split.png', dpi=400, bbox_inches='tight')
