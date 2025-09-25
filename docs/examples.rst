Examples
========

This section provides examples of using PyEst for various applications.

Splitting and Plotting PyEst GMMs
---------------------------------------------------------------------

This example demonstrates using PyEst to split a mixand in a GMM, and how to plot the resulting GMM:

.. literalinclude:: ../examples/example_gm_split.py
   :language: python
.. warning::
    NOTE: PyEst GMMs do NOT support GMMs with 0 mixands. To avoid this error, be sure to add a mixand before removing one from the mixture, as we did above.

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
    This example utilizes a cache of precomputed Monte Carlo samples. If the cache is not available, this example will generate new samples and store them in a cache for future use. On first run, this may take a few minutes to build the cache. 

.. literalinclude:: ../examples/example_splitting_cislunar.py
   :language: python
