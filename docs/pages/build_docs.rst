Build Docs Locally 
=====================================

To build the docs locally:

.. parsed-literal::
    source venv/bin/activate
    export PYTHONPATH=$PWD/venv/bin/python/dist-packages
    python -m pip install sphinx sphinx-rtd-theme
    cd docs
    make html
    open _build/html/index.html
