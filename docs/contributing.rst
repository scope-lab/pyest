Contributing
===========

Installation
-----------

OS X (zsh)
~~~~~~~~~

To install everything in developer mode:

.. code-block:: shell

   pip install -e '.[test,examples]'

OS X (bash), Windows (cmd prompt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install everything in developer mode:

.. code-block:: shell

   pip install -e .[test,examples]

Testing
-------

Using tox (recommended)
~~~~~~~~~~~~~~~~~~~~~

If you don't already have tox installed, you can pip install it:

.. code-block:: shell

   pip install tox

To run unit tests:

.. code-block:: shell

   tox run

Using pytest
~~~~~~~~~~~

To run unit tests:

.. code-block:: shell

   pytest --cov=pyest --cov-report term-missing tests

To run unit tests with performance benchmarking:

.. code-block:: shell

   pytest --benchmark-save=benchmark --benchmark-compare --cov=pyest --cov-report term-missing tests

Documentation
-----------

To build the documentation:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme

Then

.. code-block:: bash

   cd docs
   make html

Pull Request Process
-----------------

1. Update the documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update the CHANGELOG.md
5. Submit a pull request

For more details, please see our :doc:`CONTRIBUTING.md <../CONTRIBUTING.md>` file.
