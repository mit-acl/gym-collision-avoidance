import numpy as np
import collision_avoidance_env

### Minimum working example

# Initialize the environment
env = collision_avoidance_env.CollisionAvoidanceEnv()
# TODO: create way of initialzing an experiment
obs = env.reset() # Get agents' initial observations

# Set up agents' actions as np array
num_agents = len(env.agents)
num_actions_per_agent = 2 # speed, delta heading angle
actions = np.zeros((num_agents, num_actions_per_agent), dtype=np.float32)

# Alternate btwn sending actions to the environment, receiving feedback
num_steps = 50
for i in range(num_steps):
    # Fill in actions with something interesting (e.g. using obs)
    for agent_index, agent in enumerate(env.agents):
        actions[agent_index,:] = agent.policy.find_next_action(obs, env.agents, agent_index)
    # state_of_real_agents = None
    # for agent in env.agents:
    #   if agent.policy is "real":
    #       agent.set_state(state_of_real_agents[i])
    obs, rewards, game_over, which_agents_done = env.step(actions)
    print(obs)
    if game_over:
        print("All agents finished!")
        obs = env.reset()
        break

print("Experiment over.")
