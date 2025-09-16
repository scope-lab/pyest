Examples
========

This section provides examples of using PyEst for various applications.

Splitting and Plotting PyEst GMMs
---------------------------------------------------------------------

This example demonstrates using PyEst to split a mixand in a GMM, and how to plot the resulting GMM:

.. literalinclude:: ../examples/example_gm_split.py
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
(CR3BP):

.. literalinclude:: ../examples/example_splitting_cislunar.py
   :language: python
