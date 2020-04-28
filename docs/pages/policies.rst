.. _policies:

##################
Policies
##################

Base Classes
================

All agents have a Policy, which nominally converts an observation to an action.
Agents whose decision-making is completely internal to the environment have an InternalPolicy; those whose decision-making occurs at least partially externally have an ExternalPolicy.

.. autoclass:: gym_collision_avoidance.envs.policies.Policy.Policy
   :members:

.. autoclass:: gym_collision_avoidance.envs.policies.InternalPolicy.InternalPolicy
   :members:

.. autoclass:: gym_collision_avoidance.envs.policies.ExternalPolicy.ExternalPolicy
   :members:

----

.. _all_internal_policies:

Internal Policies
=================

****************
Simple Policies
****************

StaticPolicy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: gym_collision_avoidance.envs.policies.StaticPolicy.StaticPolicy
   :members:



NonCooperativePolicy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: gym_collision_avoidance.envs.policies.NonCooperativePolicy.NonCooperativePolicy
   :members:


*********************
Model-Based Policies
*********************


RVOPolicy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: gym_collision_avoidance.envs.policies.RVOPolicy.RVOPolicy
   :members:


********************
Learned Policies
********************


CADRLPolicy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: gym_collision_avoidance.envs.policies.CADRLPolicy.CADRLPolicy
   :members:


GA3CCADRLPolicy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: gym_collision_avoidance.envs.policies.GA3CCADRLPolicy.GA3CCADRLPolicy
   :members:

DRLLongPolicy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: gym_collision_avoidance.envs.policies.DRLLongPolicy.DRLLongPolicy
   :members:

----


.. _all_external_policies:

External Policies
=================

.. note::
    TODO

*********************
Still Being Trained
*********************

LearningPolicy
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gym_collision_avoidance.envs.policies.LearningPolicy.LearningPolicy
   :members:

LearningPolicyGA3C
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: gym_collision_avoidance.envs.policies.LearningPolicyGA3C.LearningPolicyGA3C
   :members:

*******************************
Pre-trained, but still external
*******************************

CARRLPolicy
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: gym_collision_avoidance.envs.policies.CARRLPolicy.CARRLPolicy
   :members:



