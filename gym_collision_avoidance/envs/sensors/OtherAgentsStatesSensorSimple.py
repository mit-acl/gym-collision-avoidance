import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import compute_time_to_impact, vec2_l2_norm
import operator


class OtherAgentsStatesSensorSimple(Sensor):
    """ TODO: Update this doc (rest is outdated ===>) A dense matrix of relative states of other agents (e.g., their positions, vel, radii)

    :param max_num_other_agents_observed: (int) only can observe up to this many agents (the closest ones)
    :param agent_sorting_method: (str) definition of closeness in words (one of ['closest_last', 'closest_first', 'time_to_impact'])

    """
    def __init__(self, max_num_other_agents_observed=Config.MAX_NUM_OTHER_AGENTS_OBSERVED, agent_sorting_method=Config.AGENT_SORTING_METHOD):
        Sensor.__init__(self)
        self.name = 'other_agents_states'
        self.max_num_other_agents_observed = max_num_other_agents_observed
        self.agent_sorting_method = agent_sorting_method

    def sense(self, agents, agent_index, top_down_map=None):
        """ TODO: Update this doc (rest is outdated ===>). Go through each agent in the environment, and compute its relative position, vel, etc. and put into an array

        This is a denser measurement of other agents' states vs. a LaserScan or OccupancyGrid

        Args:
            agents (list): all :class:`~gym_collision_avoidance.envs.agent.Agent` in the environment
            agent_index (int): index of this agent (the one with this sensor) in :code:`agents`
            top_down_map (2D np array): binary image with 0 if that pixel is free space, 1 if occupied (not used!)

        Returns:
            other_agents_states (np array): (max_num_other_agents_observed x 7) the 7 states about each other agent, :code:`[p_parallel_ego_frame, p_orthog_ego_frame, v_parallel_ego_frame, v_orthog_ego_frame, other_agent.radius, combined_radius, dist_2_other]`

        """
        host_agent = agents[agent_index]

        other_agents_states = np.zeros((Config.MAX_NUM_OTHER_AGENTS_OBSERVED, 2))
        other_agent_count = 0
        for other_agent in agents:
            if other_agent.id == host_agent.id:
                continue

            other_obs = other_agent.pos_global_frame
            host_agent.other_agent_states[:] = other_obs
            other_agents_states[other_agent_count, :] = other_obs
            other_agent_count += 1

        host_agent.num_other_agents_observed = other_agent_count

        return other_agents_states
