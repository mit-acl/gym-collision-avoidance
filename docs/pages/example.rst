.. _example:

Minimum working example
=======================

To simulate a 2-agent scenario:

.. parsed-literal::
    ./example.sh

This will save a plot in :code:`gym_collision_avoidance/experiments/results/example` so you can visualize the agents' trajectories.

You can use :code:`gym_collision_avoidance/experiments/src/example.py` as a starting point to write code for this environment.


.. only:: html

    .. figure:: ../_static/example.gif

Now, you can either read about the :ref:`architecture` or try some of the Use Cases, such as :ref:`use_case_compare_new_policy` or :ref:`use_case_train_rl`!

----

.. note::
    The shell script sources the right version of Python and the venv, but if you've handled this on your own, you can simply call the python file:

    .. parsed-literal::
        python gym_collision_avoidance/experiments/src/example.py