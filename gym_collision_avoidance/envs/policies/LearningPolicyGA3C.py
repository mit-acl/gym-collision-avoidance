import numpy as np

from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.GA3C_CADRL import network

class LearningPolicyGA3C(LearningPolicy):
    def __init__(self):
        LearningPolicy.__init__(self)
        self.possible_actions = network.Actions()

    def network_output_to_action(self, agent, network_output):
        # network_output **plz confirm!**: [speed scaling btwn 0-1, max heading angle delta btwn 0-1]

        raw_action = self.possible_actions.actions[int(network_output)]
        action = np.array([agent.pref_speed*raw_action[0], raw_action[1]])
        return action