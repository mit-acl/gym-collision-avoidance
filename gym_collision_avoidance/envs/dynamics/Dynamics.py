import numpy as np
import math
from gym_collision_avoidance.envs.util import wrap


class Dynamics(object):
    """ Class to convert an agent's action to change in state

    There is a fundamental issue in passing the agent to the Dynamics. There should be a State object or something like that.

    :param agent: (:class:`~gym_collision_avoidance.envs.agent.Agent`) the Agent whose state should be updated

    """

    def __init__(self, agent):
        self.agent = agent
        pass

    def step(self, action, dt):
        """ Dummy method to be implemented by each Dynamics subclass
        """
        raise NotImplementedError

    def update_ego_frame(self):
        """ Update agent's heading and velocity by converting those values from the global to ego frame.

        This should be run every time :code:`step` is called (add to :code:`step`?)

        """
        # Compute heading w.r.t. ref_prll, ref_orthog coordinate axes
        self.agent.ref_prll, self.agent.ref_orth = self.agent.get_ref()
        ref_prll_angle_global_frame = np.arctan2(self.agent.ref_prll[1],
                                                 self.agent.ref_prll[0])
        self.agent.heading_ego_frame = wrap(self.agent.heading_global_frame -
                                      ref_prll_angle_global_frame)

        # Compute velocity w.r.t. ref_prll, ref_orthog coordinate axes
        cur_speed = math.sqrt(self.agent.vel_global_frame[0]**2 + self.agent.vel_global_frame[1]**2) # much faster than np.linalg.norm
        v_prll = cur_speed * np.cos(self.agent.heading_ego_frame)
        v_orthog = cur_speed * np.sin(self.agent.heading_ego_frame)
        self.agent.vel_ego_frame = np.array([v_prll, v_orthog])