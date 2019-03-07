import os
import numpy as np
from gym_collision_avoidance.envs import collision_avoidance_env
from gym_collision_avoidance.envs import test_cases as tc

# Minimum working example

# Instantiate the environment
env = collision_avoidance_env.CollisionAvoidanceEnv()

# Set agent configuration (start/goal pos, radius, size, policy)
agents = tc.get_testcase_hololens_and_ga3c_cadrl()
env.init_agents(agents)

# Set static map of the environment (e.g. if there are static obstacles)
static_map_filename = os.path.dirname(collision_avoidance_env.__file__)+"/world_maps/002.png"
env.init_static_map(map_filename=static_map_filename)
# env.init_static_map(map_filename=None)

# Set up empty np array for agents' actions
num_actions_per_agent = 2  # speed, delta heading angle
actions = np.zeros((len(env.agents), num_actions_per_agent), dtype=np.float32)

obs = env.reset()  # Get agents' initial observations

# Alternate btwn sending actions to the environment, receiving feedback
num_steps = 50
for i in range(num_steps):

    # Query the agents' policies (e.g. using obs vector or list of agents)
    # Note: This needs to occur before updating the real agent positions!
    for agent_index, agent in enumerate(env.agents):
        actions[agent_index, :] = agent.policy.find_next_action(obs, env.agents, agent_index)

    # Update position of real agents based on real sensor data (if necessary)
    state_of_real_agents = [[-1, 0.1*i, 2+np.random.normal(0, 0.1)]]
    for state in state_of_real_agents:
        agent = env.agents[state[0]]
        agent.set_state(px=state[1], py=state[2])

    # Run a simulation step (check for collisions, move sim agents)
    obs, rewards, game_over, which_agents_done = env.step(actions)

    if game_over:
        print("All agents finished!")
        # To start a new episode...
        # env.init_agents(agents) # again with new test case
        # obs = env.reset()
        break

print("Experiment over.")
