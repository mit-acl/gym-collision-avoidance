import numpy as np
from gym_collision_avoidance.envs.util import wrap

class Policy(object):
    """ Each :class:`~gym_collision_avoidance.envs.agent.Agent` has one of these, which nominally converts an observation to an action

    :param is_still_learning: (bool) whether this policy is still being learned (i.e., weights are changing during execution)
    :param is_external: (bool) whether the Policy computes its own actions or relies on an external process to provide an action.

    """
    def __init__(self, str="NoPolicy"):
        self.str = str
        self.is_still_learning = False
        self.is_external = False

    def near_goal_smoother(self, dist_to_goal, pref_speed, heading, raw_action):
        """ Linearly ramp down speed/turning if agent is near goal, stop if close enough.

        I think this is just here for convenience, but nobody uses it? We used it on the jackal for sure.
        """

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