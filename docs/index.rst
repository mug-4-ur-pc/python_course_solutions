Discrete Radon Transform
==============================

.. toctree::
   :maxdepth: 1

   reference

The Discrete Radon Transform to slope-offset space implementation.
The command-line interface runs Discrete Radon Transform on any .png images.


Installation
------------

To install the DRT project,
run this command in your terminal from project source folder:

.. code-block:: console

   $ pip install poetry
   $ poetry install --only main


Usage
-----

DRT's usage looks like:

.. code-block:: console

   $ poetry run semester6 [OPTIONS]

.. option:: -f <PATH>, --file <PATH>

   Path to the source image for the transformation.

.. option:: -o <PATH>, --output <PATH>

   Path where output will be written

.. option:: --full

    Transform original image and its transpose result to produce all lines finding.

.. option:: -i ["nearest", "linear", "sinc"], --interpolation ["nearest", "linear", "sinc"]
    Set an interpolation method. Linear interpolation will be used by default.

.. option:: --version

   Display the version and exit.

.. option:: --help

   Display a short usage message and exit.
