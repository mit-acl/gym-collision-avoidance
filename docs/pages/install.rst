Install
========================

Grab the code from github, initialize submodules, install dependencies and src code:

.. parsed-literal::
    git clone --recursive git@github.com:mit-acl/gym-collision-avoidance.git # If internal to MIT-ACL, use GitLab origin instead
    cd gym-collision-avoidance
    ./install.sh

You should be all set to move onto :ref:`example`!

----

Common Issues
-------------

**Issue:**

.. parsed-literal::
    RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework.

**Fix:** Add a line `backend: TkAgg` to `~/.matplotlib/matplotlibrc`.

----

**Issue:**

.. parsed-literal::
    error: Cannot compile MPI programs. Check your configuration!!!

**Fix:**

.. parsed-literal::
    brew install mpich

----

**Issue:**

.. parsed-literal::
    error with matplotlib and freetype not being found

**Fix:**

.. parsed-literal::
    brew install pkg-config

----

**Issue:**

To update the :code:`Python-RVO2` source code and re-generate the :code:`rvo2` python library, the results won't have any effect unless you remove the :code:`build` dir:

.. parsed-literal::
    # enter the venv
    cd gym-collision-avoidance/gym_collision_avoidance/envs/policies/Python-RVO2
    rm -rf build && python setup.py build && python setup.py install
