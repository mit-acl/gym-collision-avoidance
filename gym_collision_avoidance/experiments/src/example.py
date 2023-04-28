import os

import gym
import numpy as np

gym.logger.set_level(40)
os.environ["GYM_CONFIG_CLASS"] = "Example"
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import test_cases as tc


def main():
    """
    Minimum working example:
    2 agents: 1 running external policy, 1 running GA3C-CADRL
    """

    # Create single tf session for all experiments
    import tensorflow.compat.v1 as tf

    tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("CollisionAvoidance-v0")

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__))
        + "/../../experiments/results/example/"
    )

    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = tc.get_testcase_two_agents()
    [
        agent.policy.initialize_network()
        for agent in agents
        if hasattr(agent.policy, "initialize_network")
    ]
    env.set_agents(agents)

    obs = env.reset()  # Get agents' initial observations

    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 100
    for i in range(num_steps):
        # Query the external agents' policies
        # e.g., actions[0] = external_policy(dict_obs[0])
        actions = {}
        actions[0] = np.array([1.0, 0.5])

        # Internal agents (running a pre-learned policy defined in envs/policies)
        # will automatically query their policy during env.step
        # ==> no need to supply actions for internal agents here

        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, terminated, truncated, which_agents_done = env.step(
            actions
        )

        if terminated:
            print("All agents finished!")
            break
    env.reset()

    return True


if __name__ == "__main__":
    main()
    print("Experiment over.")
