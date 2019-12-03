import numpy as np
from gym_collision_avoidance.envs.util import wrap

class Policy(object):
    def __init__(self, str="NoPolicy"):
        self.str = str
        self.is_still_learning = False
        self.ppo_or_learning_policy = False
        self.is_external = False

    def find_next_action(self, agents):
        raise NotImplementedError

    def near_goal_smoother(self, dist_to_goal, pref_speed, heading, raw_action):
        kp_v = 0.5
        kp_r = 1

        if dist_to_goal < 2.0:
            near_goal_action = np.empty((2,1))
            pref_speed = max(min(kp_v * (dist_to_goal-0.1), pref_speed), 0.0)
            near_goal_action[0] = min(raw_action[0], pref_speed)
            turn_amount = max(min(kp_r * (dist_to_goal-0.1), 1.0), 0.0) * raw_action[1]
            near_goal_action[1] = wrap(turn_amount + heading)
        if dist_to_goal < 0.3:
            near_goal_action = np.array([0., 0.])
        else:
            near_goal_action = raw_action

        return near_goal_action