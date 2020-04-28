.. _train_multiagent_rl:

Train a Multiagent RL Policy
============================

Training RL with multiple learning agents might look something like this:

.. parsed-literal::
    obs_for_all_agents = env.reset()
    for i in range(num_episodes):
        actions = {}
        for agent_index, obs in obs_for_all_agents:
            if obs['is_learning']:
                rl_action = model.sample(obs)
                actions[agent_index] = rl_action

        # No need to supply actions for non-learning agents

        # Run a simulation step (check for collisions, move sim agents)
        obs_for_all_agents, rewards, game_over, info = env.step([actions])

        # associate the obs_for_all_agents, rewards, and info['which_agents_done'] with one another

        # Do RL stuff with each agent's (obs, reward, action)

.. note::
    You also need to handle the fact that some agents will finish before others. Maybe send me (Michael) a message if you're trying to do this and I can give some advice.