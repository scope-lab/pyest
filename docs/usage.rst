Usage
=====

Basic Usage
----------

Import the `gm` module of PyEst as well as numpy and matplotlib:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import pyest.gm as gm

Create a three-mixand two-dimensional Gaussian mixture:

.. code-block:: python

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

Compute the mean and covariance of the distribution:

.. code-block:: python

   # compute and print the mean
   print(p.mean())
   # compute and print the covariance
   print(p.cov())

Plot the Gaussian mixture:

.. code-block:: python

   pp, XX, YY = p.pdf_2d()
   fig = plt.figure()
   ax = fig.add_axes(111)
   ax.contourf(XX,YY,pp,100)

Apply a linear transformation to the mixture:

.. code-block:: python

   dt = 5
   F = np.array([[1, dt], [0, 1]])
   my = np.array([F@m for m in p.m])
   Py = np.array([F@P@F.T for P in p.P])
   py = gm.GaussianMixture(p.w, my, Py)

Plot the transformed Gaussian mixture:

.. code-block:: python

   pp, XX, YY = py.pdf_2d()
   fig = plt.figure()
   ax = fig.add_axes(111)
   ax.contourf(XX,YY,pp,100)
   plt.show()

Advanced Usage
-------------

For more advanced usage, including nonlinear transformations and splitting methods,
please refer to the :doc:`examples`