.. gym-collision-avoidance documentation master file, created by
   sphinx-quickstart on Tue Apr 21 19:23:53 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gym-collision-avoidance's documentation!
===================================================

.. only:: html

    .. figure:: ../misc/combo.gif

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Using the Scripts

   pages/example.rst
   pages/run_cadrl_formations.rst
   pages/run_full_test_suite.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Key Concepts

   pages/env.rst
   pages/agent.rst
   pages/sensors.rst
   pages/policies.rst
   pages/dynamics.rst

Software Architecture
=====================

There are 5 key types of objects: |Environment|, |Agent|, |Sensor|, |Policy|, and |Dynamics|.

.. figure:: ../misc/gym_arch.png

A simulation begins by creating an instance of the |Environment| : :code:`env = CollisionAvoidanceEnv()`

After initialization, the |Environment| contains a list of |Agent| s, stored at :code:`env.agents` (More on how this list gets populated, later).
Each |Agent| in that list has a |Policy|, |Dynamics|, a list of |Sensor| s, and other attributes about its state, such as its current position, velocity, radius, and goal position.

An external process moves the |environment| forward in time by calling :code:`env.step`, which triggers each |Agent| to:

- use its |Policy| to compute its next action from its current observation
- use its |Dynamics| model to compute its next state by applying the computed action
- use its |Sensor| s to compute its next observation by measuring properties of the |Environment|.

The env.step call returns useful information to the caller (not in this order):

- **next_obs**: All agents' next observations, computed by their Sensors, are stuffed into one data structure. Note: This is only useful if some |Agent| s have ExternalPolicies, otherwise the |Environment| already has access to the each |Agent|'s most recent observation and does action selection internally.
- **info**: The |Environment| will also check whether any |Agent| s have collided with one another or a static obstacle, reached their goal, or run out of time. If any of these conditions are True, that |Agent| is "done", and a dict of which agents are done/not is put into info.
- **game_over**: If all agents are "done", then that episode is "done", which is also returned by the env.step call ("game_over") [Note: the condition on when to end the episode, (e.g., wait for all agents to be done, just one, just ones with certain properties) can be adjusted]
- **reward**: In the process of checking for collisions and goal-reaching, the |Environment| also computes a scalar reward for each |Agent|; the reward list is returned by the env.step call

Connecting to an RL Training script
======================================================
Let's say the external process that initialized the |Environment| is an RL algorithm, which already has a mechanism for computing an action from an observation.

The external process can pass a dict of actions (keyed by the index of the Agent should take that action) as an argument to the env.step command.
Using a dict of actions (rather than a list/array) isolates the RL algorithm from the |Environment|, so that the RL algorithm need not know how many Agents are in the |Environment|, and the |Environment|
can handle everything related to non-RL Agents itself.

You can distinguish Agents who should follow the action dict vs. query their own |Policy| by assigning the appropriate |Policy| sub-class to each Agent.
For instance, Agents whose actions come from an external process can be given an |ExternalPolicy|.

Since the RL algorithm might not be aware of the Env-specific actions (e.g., if RL returns a discrete :math:`\texttt{raw_action}\in\{0,1,2,3\}`, it still must be converted to a vehicle speed command for this environment), the :code:`raw_action` from the :code:`actions` dict is sent through :code:`action = ExternalPolicy.convert_to_action(raw_action)`.

:code:`ExternalPolicy.convert_to_action` is a dummy pass-through, but you can create a sub-class of |ExternalPolicy| and implement the desired conversion for your RL output.

.. |Environment| replace:: :class:`~gym_collision_avoidance.envs.collision_avoidance_env.CollisionAvoidanceEnv`
.. |Agent| replace:: :class:`~gym_collision_avoidance.envs.agent.Agent`
.. |Policy| replace:: :class:`~gym_collision_avoidance.envs.policies.Policy.Policy`
.. |ExternalPolicy| replace:: :class:`~gym_collision_avoidance.envs.policies.ExternalPolicy.ExternalPolicy`
.. |Dynamics| replace:: :class:`~gym_collision_avoidance.envs.dynamics.Dynamics.Dynamics`
.. |Sensor| replace:: :class:`~gym_collision_avoidance.envs.sensors.Sensor.Sensor`
