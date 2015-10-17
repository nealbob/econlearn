=========
econlearn
=========

A python machine learning toolkit for economists.

Documentation   Examples

Overview
========

``econlearn`` contains a number of supervised learning (regression), unsupervised learning and reinforcement learning algorithms, suitable for problems with large sample sizes but relatively few dimensions (large N, small D). 

``econlearn`` was developed to solve problems that arise in economics, particularly stochastic dynamic optimisation problems (Markov Decision Processes MDPs). A key feature of ``econlearn`` is the batch reinforcement learning method 'Fitted QV iteration with Tile coding, for solving MDPs by simulation (described in more detail `here <https://nealhughes.net/thesis/Thesis_main.pdf#page=200>`_).

``econlearn`` is designed to be fast. All algorithms are implemented in Cython and several make use of multi-threading. 

The methods all use one of two main 'data structures': tile coding (i.e., overlapping grids) or distance metrics.  The strength of tile coding is speed. In low dimensional problems tile coding is just about unbeatable for fitting and prediction time. With higher dimensions (say more than 10) tile coding can run into memory problems. Distance methods permit the use of continuous basis functions (i.e. radial basis functions).


Supervised learning (regression)
--------------------------------
Tile coding
Radial Basis Functions
Local quadratic regression
Distance weighted regression 

Unsupervised learning
---------------------
Sample grid (distance)
Sample grid (tile coding)
Density estimation (tile coding)
Approximate nearest neighbors (tile coding)

Reinforcement learning
----------------------
Fitted QV Iteration with Tile coding

Installation
============

This code requires installation of python with packages: ``cython``, ``numpy``, and ``scikit-learn``. A good option is to install `Anaconda <http://docs.continuum.io/anaconda/>`_.  

Next download or clone this repository. Then you need to compile the Cython modules. To do this navigate to your local ``econlearn`` directory and run the setup.py file

    python setup.py build_ext --inplace

This code has been developed and tested on linux machines only. In windows a compiler with ``openMP`` support will be required. If I get a chance I will provide more instructions on installation under Windows later.

Development
===========

``econlearn`` is a work in progress. So far only the tile coding methods are available, In time I will hopefully add more algorithms including: distance based sample grid, radial basis function regression, distance weighted regression and local quadratic regression.

While not documented, the underlying Cython modules may be of use to those wanting to access these learning methods quickly within larger applications / models. Including those wanting to implement the reinforcement learning methods in a multi-agent simulation model.
