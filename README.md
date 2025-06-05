![pyest_logo](./docs/image/pyest_logo.png)

# PyEst: Adaptive Gaussian Mixture State Estimation

[![Python application](https://github.com/scope-lab/pyest/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/scope-lab/pyest/actions/workflows/pythonapp.yml)

## Basic Usage

Import the `gm` module of PyEst as well as numpy and matplotlib
```
import numpy as np
import matplotlib.pyplot as plt
import pyest.gm as gm
```

Create a three-mixand two-dimensional Gaussian mixture:
```
# mixand means (nc,nx)
m = np.array([[0,0], [1,2], [0,-1]])
# mixand covariance matrices (nc,nx,nx)
P = np.array([[[1,0], [0,1]],
              [[2, 0.5], [0.5,3]],
              [[0.5, -0.1], [-0.1, 1]]])
# mixand weights (nc,)
w = gm.equal_weights(3)
# contruct the Gaussian mixture
p = gm.GaussianMixture(w, m, P)
```

Compute the mean and covariance of the distribution:
```
# compute and print the mean
print(p.mean())
# compute and print the covariance
print(p.cov())
```

Plot the Gaussian mixture
```
pp, XX, YY = p.pdf_2d()
fig = plt.figure()
ax = fig.add_axes(111)
ax.contourf(XX,YY,pp,100)
```

Apply a linear transformation to the mixture
```
dt = 5
F = np.array([[1, dt], [0, 1]])
my = np.array([F@m for m in p.m])
Py = np.array([F@P@F.T for P in p.P])
py = gm.GaussianMixture(p.w, my, Py)
```

Plot the transformed Gaussian mixture
```
pp, XX, YY = py.pdf_2d()
fig = plt.figure()
ax = fig.add_axes(111)
ax.contourf(XX,YY,pp,100)
plt.show()
```

## Installation

### OS X (zsh)
To install, run
```shell
pip install pyest
```

To install packages needed for running the examples, run
 ```shell
pip install 'pyest[examples]'
```

### OS X (bash), Windows (cmd prompt)
To install, run
```shell
pip install pyest
```

To install packages needed for running the examples, run
 ```shell
pip install pyest[examples]
```


## Citing this work

If you use this package in your scholarly work, please cite the following articles:

[K.A. LeGrand and S. Ferrari, “Split Happens! Imprecise and Negative Information in Gaussian Mixture Random Finite Set Filtering,” Journal of Advances in Information Fusion,  Vol 17, No. 2, December, 2022](http://keithlegrand.com/wp/wp-content/uploads/2023/05/LeGrand-2022-Split-Happens-Imprecise-and-Negative-Information-in-Gaussian-Mixture-Random-Finite-Set-Filtering.pdf)

J. Kulik and K.A. LeGrand, “Nonlinearity and Uncertainty Informed Moment-Matching Gaussian Mixture Splitting,” https://arxiv.org/abs/2412.00343


```
@article{legrand2022SplitHappensImprecise,
  title = {Split {{Happens}}! Imprecise and Negative Information in {G}aussian Mixture Random Finite Set Filtering},
  author = {LeGrand, Keith A. and Ferrari, Silvia},
  year = {2022},
  month = dec,
  journal = {Journal of Advances in Information Fusion},
  volume = {17},
  number = {2},
  eprint = {2207.11356},
  primaryclass = {cs, eess},
  pages = {78--96},
  doi = {10.48550/arXiv.2207.11356},
}
@misc{kulik2024NonlinearityUncertaintyInformed,
  title = {Nonlinearity and {{Uncertainty Informed Moment-Matching Gaussian Mixture Splitting}}},
  author = {Kulik, Jackson and LeGrand, Keith A.},
  year = {2024},
  month = nov,
  number = {arXiv:2412.00343},
  eprint = {2412.00343},
  primaryclass = {stat},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2412.00343},
  urldate = {2025-01-01},
  archiveprefix = {arXiv}
}

```

## Documentation

For more information about PyEst, please see the [documentation](https://pyest.readthedocs.io/en/latest/).
