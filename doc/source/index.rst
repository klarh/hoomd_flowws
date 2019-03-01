Welcome to hoomd_flowws's documentation!
========================================

`hoomd_flowws <https://github.com/klarh/hoomd_flowws>`_ is an
in-development set of modules to create reusable scientific workflows
using `hoomd-blue <https://github.com/glotzerlab/hoomd-blue>`_. While
the python API of hoomd-blue holds enormous possibility for
scriptability (including making projects like this possible in the
first place), this flexibility can also lead to poorly-structured,
rigid script workflows if not carefully managed. The aim of this
project is to formulate a set of robust, modular individual components
that can be composed to perform most common workflows.

Installation
------------

Install `flowws` and `hoomd-flowws` from source on github::

  pip install git+https://github.com/klarh/flowws.git#egg=flowws
  pip install git+https://github.com/klarh/hoomd_flowws.git#egg=hoomd_flowws

Examples
--------

See the `examples` directory inside the project for usage examples.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
