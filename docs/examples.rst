Examples
========

This section provides examples of using PyEst for various applications.

Splitting and Plotting PyEst Gaussian Mixtures
---------------------------------------------------------------------

This example demonstrates using PyEst to split a mixand in a Gaussian mixture, and how to plot the resulting Gaussian mixture:

.. literalinclude:: ../examples/example_gm_2d_split.py
   :language: python

Gaussian Mixture Splitting for Field-of-View and Negative Information
---------------------------------------------------------------------

This example demonstrates recursive splitting for fields-of-view for incorporating negative information:

.. literalinclude:: ../examples/example_split_for_fov.py
   :language: python

Cartesian to Polar Transformation
------------------------------

This example demonstrates the transformation of a Gaussian mixture from Cartesian to polar coordinates:

.. literalinclude:: ../examples/example_splitting_polar_transformation.py
   :language: python

Cislunar Space Object Uncertainty Propagation
---------------------------------------------

This example shows how to use pyest for propagating uncertainty in the circular restricted three-body problem
(CR3BP).

.. note::
    This example utilizes a cache of precomputed Monte Carlo samples to evaluate various performance measures.
    If the cache is not available, this example will generate new samples and store them in a cache for future use.
    On first run, this may take a few minutes to build the cache. These samples are only for performance evaluation
    and not required for any of the adaptive Gaussian splitting operations.

.. literalinclude:: ../examples/example_splitting_cislunar.py
   :language: python
