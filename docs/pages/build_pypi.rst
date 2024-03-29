Build for pypi 
=====================================

To set up local env for packaging and sending to pypi:

.. parsed-literal::
    source venv/bin/activate
    export PYTHONPATH=$PWD/venv/bin/python/dist-packages
    python -m pip install --upgrade build twine
    
To build and send a new package to pypi:

.. parsed-literal::

    source venv/bin/activate
    export PYTHONPATH=$PWD/venv/bin/python/dist-packages

    # Build the package
    python -m build

    # Upload to the test pypi server
    python -m twine upload --repository testpypi dist/*

    # Verify it works (via the testpypi server)
    python -m pip install --pre --extra-index-url https://test.pypi.org/simple/ gym-collision-avoidance==0.0.3a0

    # Upload to the real pypi server
    python -m twine upload dist/*


