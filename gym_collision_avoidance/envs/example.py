import numpy as np
import collision_avoidance_env

### Minimum working example

# Initialize the environment
env = collision_avoidance_env.CollisionAvoidanceEnv()
obs = env.reset() # Get agents' initial observations

# Set up agents' actions as np array
num_agents = len(env.agents)
num_actions_per_agent = 2 # speed, delta heading angle
actions = np.zeros((num_agents, num_actions_per_agent), dtype=np.float32)

# Alternate btwn sending actions to the environment, receiving feedback
num_steps = 10
for i in range(num_steps):
	# Fill in actions with something interesting (e.g. using obs)
    obs, rewards, game_over, which_agents_done = env.step(actions)
    if game_over:
    	print("All agents finished!")
    	break

print("Experiment over.")
