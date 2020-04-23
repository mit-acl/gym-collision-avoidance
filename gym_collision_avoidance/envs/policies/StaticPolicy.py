import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy

class StaticPolicy(Policy):
    """ For an agent who never moves, useful for confirming algorithms can avoid static objects too """
    def __init__(self):
        Policy.__init__(self, str="Static")
        pass

    def find_next_action(self, obs, agents, i):
        """ Static Agents do not move, so just set goal to current pos and action to zero. 

        Args:
            obs (dict): ignored
            agents (list): of Agent objects
            i (int): this agent's index in that list

        Returns:
            np array of shape (2,)... [spd, delta_heading] both are zero.

        """
        agents[i].goal_global_frame = agents[i].pos_global_frame
        action = np.array([0.0, 0.0])
        return action
