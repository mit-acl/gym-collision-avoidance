import numpy as np

from gym_collision_avoidance.envs.policies.Policy import Policy

class PPOPolicy(Policy):
    def __init__(self):
        Policy.__init__(self)

    def network_output_to_action(self, agent, network_output):
        # network_output: [speed scaling btwn 0-1, max heading angle delta btwn 0-1]
        heading = agent.max_heading_change*(2.*network_output[1] - 1.)
        speed = agent.pref_speed * network_output[0]
        actions = np.array([speed, heading])
        return actions

    def find_next_action(self, obs, agents, i):
        raise NotImplementedError
