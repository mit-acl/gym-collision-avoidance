.. _use_case_compare_new_policy:

Compare a new policy to the baselines
========================================

Background
-----------

You may have your own collision avoidance policy implemented for another simulator.
You can add a small wrapper around your policy so that it can be used by Agents in this environment.

To easily make comparisons between algorithms, this repo provides several model-based and learning-based approaches, pre-defined test scenarios, and data collection scripts.

Instructions
-------------------

#. Create a new |Policy| sub-class (can use :class:`~gym_collision_avoidance.envs.policies.NonCooperativePolicy.NonCooperativePolicy` as an example)

   * If necessary, add a submodule containing your helper methods that were written for some other environment (see Python-RVO2 as an example)
   * Implement the :code:`find_next_action` method of your new policy
   * Import your new policy at the top of :code:`test_cases.py` and add it to the :code:`policy_dict` (e.g., add :code:`'new_policy': NewPolicy`)

#. Add an element to the dict in :code:`env_utils.py` corresponding to your new policy (e.g., :code:`'new_policy_name': {'policy': 'new_policy', 'sensors': ['other_agents_states_sensor']}`)
#. In :code:`config.py`, add an element to :code:`self.POLICIES_TO_TEST` with your new policy's name, :code:`'new_policy_name'`
#. In :code:`config.py`, update :code:`self.NUM_AGENTS_TO_TEST` and :code:`self.NUM_TEST_CASES`, if desired

Run the test case script, which will run the same NUM_TEST_CASES scenarios for each policy in POLICIES_TO_TEST, for each number of agents in NUM_AGENTS_TO_TEST:

.. parsed-literal::
   ./gym_collision_avoidance/experiments/run_full_test_suite.sh

These will a :code:`.png` for each trajectory on each test cases in :code:`experiments/results/full_test_suites`, and, if desired, a pandas DataFrame with statistics about the results.

For example:

.. list-table::

    * - .. figure:: ../_static/000_CADRL_2agents.png
        
        CADRL

      - .. figure:: ../_static/000_GA3C-CADRL-10_2agents.png
        
        GA3C-CADRL

      - .. figure:: ../_static/000_RVO_2agents.png
        
        RVO

----

.. _use_case_train_rl:

Train a new RL policy
=====================================

Background
-------------------

So far, all the policies discussed in this document are pre-trained/defined, and the Environment queries those policies to compute an action from an observation.

However, RL training scripts already have a mechanism for computing an action from an observation.
Moreover, the RL algorithm will modify its action-selection rule throughout training, as the policy is updated or exploration hyperparameters change.

Therefore, the |Environment| has a mechanism to accept actions from an external process and apply those to specific agents, while still keeping the agents whose policies are pre-defined totally internal to the environment.
The external process (e.g., RL training script) can pass a dict of actions -- keyed by the index of the Agent should take that action -- as an argument to the :code:`env.step` command.

You can distinguish Agents who should follow the action dict vs. query their own |Policy| by assigning the appropriate |Policy| sub-class to each Agent.
For instance, Agents whose actions come from an external process can be given an |ExternalPolicy|.

Since the RL algorithm might not be aware of the Env-specific actions (e.g., if RL returns a discrete :math:`\texttt{raw_action}\in\{0,1,2,3\}`, it still must be converted to a vehicle speed command for this environment), the :code:`raw_action` from the :code:`actions` dict is sent through :code:`action = ExternalPolicy.convert_to_action(raw_action)`.

:code:`ExternalPolicy.convert_to_action` is a dummy pass-through, but you can create a sub-class of |ExternalPolicy| and implement the desired conversion for your RL output.

Instructions
-------------------

#. Create a new |ExternalPolicy| sub-class (can use :class:`~gym_collision_avoidance.envs.policies.LearningPolicy.LearningPolicy` as an example)

   * Implement the :code:`network_output_to_action` method of your new policy
   * Import your new policy at the top of :code:`test_cases.py` and add it to the :code:`policy_dict` (e.g., add :code:`'new_policy': NewPolicy`)

.. note::
    This is incomplete...

----

Collect a dataset of trajectories 
=====================================

Background
-----------

Collecting realistic trajectory data on dynamic agents is difficult and often time-intensive.

The two typical approaches are:

#. Set up a camera and collect video of people moving (requires post-processing to extract trajectories)
#. Set up a simulation of agents and extract their trajectories (requires realistic motion models)

Many of the packages we've experimented with that implement pedestrian motion models do not produce particularly interactive behavior, but the GA3C-CADRL, CADRL, and RVO agents in this repo typically do yield some interesting multi-agent interactions.

Thus, collecting a dataset of trajectories using this repo could help with making more realistic predictions about how agents might respond to various actions by another agent, without requiring real human data.
If nothing else, the simulated trajectories can be designed to help debug and initially test your prediction code.

Instructions
-------------

.. parsed-literal::
   ./gym_collision_avoidance/experiments/run_trajectory_dataset_creator.sh

This will store :code:`png` files of the trajectories and a :code:`.pkl` file of relevant data from the trajectories in the :code:`experiments/results/trajectory_dataset` folder.
The resulting dataset could be used to train predictive models, initialize an RL agent's policy, etc.
You can change the :code:`test_case_fn` to use different scenarios, the :code:`policies` dict to give agents different policies, etc.

----

Formation Control
========================================

Background
-----------

Say you have a good policy and want to make it spell letters or make interesting shapes, rather than just do random test cases all day.

Instructions
-------------------

Spell out CADRL, with agents starting where they ended the previous episode:

.. parsed-literal::
    ./gym_collision_avoidance/experiments/run_cadrl_formations.sh

This will save plots and animations of 10 letters (:code:`.gif` and :code:`.mp4`) format in :code:`gym_collision_avoidance/experiments/results/cadrl_formations`.

For example:

.. list-table::

    * - .. figure:: ../_static/formation_C.gif
        
        C

      - .. figure:: ../_static/formation_A.gif
        
        A

.. |Environment| replace:: :class:`~gym_collision_avoidance.envs.collision_avoidance_env.CollisionAvoidanceEnv`
.. |Agent| replace:: :class:`~gym_collision_avoidance.envs.agent.Agent`
.. |Policy| replace:: :class:`~gym_collision_avoidance.envs.policies.Policy.Policy`
.. |ExternalPolicy| replace:: :class:`~gym_collision_avoidance.envs.policies.ExternalPolicy.ExternalPolicy`
.. |Dynamics| replace:: :class:`~gym_collision_avoidance.envs.dynamics.Dynamics.Dynamics`
.. |Sensor| replace:: :class:`~gym_collision_avoidance.envs.sensors.Sensor.Sensor`
