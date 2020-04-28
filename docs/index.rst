.. gym-collision-avoidance documentation master file, created by
   sphinx-quickstart on Tue Apr 21 19:23:53 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gym-collision-avoidance: Documentation
======================================

In this multiagent environment, agents try to get to their own goal location (specified at the start of each episode) by using one of many collision avoidance policies implemented.
Episodes end when agents reach their goal, collide, or timeout.
Agents can observe the environment through several types of **sensors**, act according to pre-implemented and extendable **policies**, and behave according to customizable **dynamics** models.

**Objective**: Provide a flexible codebase, reduce time spent re-implementing existing works, and establish baselines for multiagent collision avoidance problems.

.. list-table::

    * - .. figure:: _static/combo.gif
        
        Formation Control

      - .. figure:: _static/jackal_iros18.gif
        
        Policies Deployable on Hardware

    * - .. figure:: _static/a3c_20_agents.gif
        
        Many-Agent Scenarios

      - .. figure:: _static/random_2_agents.gif
        
        Random Scenarios


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   pages/install.rst
   pages/example.rst
   pages/architecture.rst
   pages/config.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Use Cases

   pages/use_cases.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Key Concepts

   pages/env.rst
   pages/agent.rst
   pages/sensors.rst
   pages/policies.rst
   pages/dynamics.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced

   pages/build_docs.rst
   pages/train_multiagent_rl.rst

Policies Implemented
====================

Learning-based:

* SA-CADRL: `Socially Aware Motion Planning with Deep Reinforcement Learning <https://arxiv.org/pdf/1703.08862.pdf>`_
* GA3C-CADRL: `Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning <https://arxiv.org/pdf/1805.01956.pdf>`_
* DRL_Long: `Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning <https://arxiv.org/abs/1709.10082>`_

Model-based:

* RVO/ORCA: `Python-RVO2 <https://github.com/sybrenstuvel/Python-RVO2>`_
* Non-Cooperative (constant velocity toward goal position)
* Static (zero velocity)

Desired Additions:

* DWA
* Social Forces
* Additional learning-based methods
* Other model-based methods
* Centralized planners
* ...


If you find this code useful, please consider citing:
=====================================================

.. parsed-literal::
   @inproceedings{Everett18_IROS,
     address = {Madrid, Spain},
     author = {Everett, Michael and Chen, Yu Fan and How, Jonathan P.},
     booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
     date-modified = {2018-10-03 06:18:08 -0400},
     month = sep,
     title = {Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning},
     year = {2018},
     url = {https://arxiv.org/pdf/1805.01956.pdf},
     bdsk-url-1 = {https://arxiv.org/pdf/1805.01956.pdf}
   }