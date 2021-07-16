Benchmark repository for TV-regularized Regression
=====================
|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of the following problem:

.. math::

    \min_{w} \frac{1}{2} \|y - \beta w\|^2_2 + \lambda \|\nabla \beta\|_1

where n (or n_samples) stands for the number of samples, p (or n_features) stands for the number of features and

.. math::

 X = [x_1^\top, \dots, x_n^\top]^\top \in \mathbb{R}^{n \times p}]
 \beta = \beta_{i+1} - \beta_i, \forall i \in {0, 1, ... len(\beta)}
 

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/hassibatej/benchmark_tv1_l1
   $ benchopt run benchmark_tv1_l1

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_tv1_l1 -s prox -d simulated --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/hassibatej/benchmark_tv1_l1/workflows/checks/badge.svg
   :target: https://github.com/hassibatej/benchmark_tv1_l1/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
