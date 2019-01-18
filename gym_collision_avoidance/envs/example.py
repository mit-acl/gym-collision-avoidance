import numpy as np
import collision_avoidance_env
import test_cases as tc

### Minimum working example

# Instantiate the environment
env = collision_avoidance_env.CollisionAvoidanceEnv()

# Set agent configuration (start/goal pos, radius, size, policy)
num_agents = 2
test_case_index = 0
agents = tc.get_testcase_old_and_crappy(num_agents, test_case_index)
env.init_agents(agents)

# Set up empty np array for agents' actions
num_actions_per_agent = 2 # speed, delta heading angle
actions = np.zeros((len(env.agents), num_actions_per_agent), dtype=np.float32)

obs = env.reset() # Get agents' initial observations

# Alternate btwn sending actions to the environment, receiving feedback
num_steps = 50
for i in range(num_steps):

    # Query the agents' policies (e.g. using obs vector or list of agents)
    # Note: This needs to occur before updating the real agent positions!
    for agent_index, agent in enumerate(env.agents):
        actions[agent_index,:] = agent.policy.find_next_action(obs, env.agents, agent_index)
    
    # # Update position of real agents based on real sensor data (if necessary)
    # for state in state_of_real_agents:
    #     agent = env.agents[state.id]
    #     agent.set_state(state_of_real_agents[i])
    
    # Run a simulation step (check for collisions, move sim agents)
    obs, rewards, game_over, which_agents_done = env.step(actions)

    if game_over:
        print("All agents finished!")
        # To start a new episode...
        # env.init_agents(agents) # again with new test case
        # obs = env.reset()
        break

print("Experiment over.")
