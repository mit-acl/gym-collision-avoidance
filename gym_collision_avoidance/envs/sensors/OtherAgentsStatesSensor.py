import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import compute_time_to_impact, vec2_l2_norm
import operator

class OtherAgentsStatesSensor(Sensor):
    """ A dense matrix of relative states of other agents (e.g., their positions, vel, radii)

    :param max_num_other_agents_observed: (int) only can observe up to this many agents (the closest ones)
    :param agent_sorting_method: (str) definition of closeness in words (one of ['closest_last', 'closest_first', 'time_to_impact'])

    """
    def __init__(self, max_num_other_agents_observed=Config.MAX_NUM_OTHER_AGENTS_OBSERVED, agent_sorting_method=Config.AGENT_SORTING_METHOD):
        Sensor.__init__(self)
        self.name = 'other_agents_states'
        self.max_num_other_agents_observed = max_num_other_agents_observed
        self.agent_sorting_method = agent_sorting_method

    def get_clipped_sorted_inds(self, sorting_criteria):
        """ Determine the closest N agents using the desired sorting criteria

        Args:
            sorting_criteria (str): how to sort the list of agents (one of ['closest_last', 'closest_first', 'time_to_impact']). See journal paper.
    
        Returns:
            clipped_sorted_inds (list): indices of the "closest" max_num_other_agents_observed 
                agents sorted by "closeness" ("close" defined by sorting criteria),

        """

        # Grab first N agents (where N=Config.MAX_NUM_OTHER_AGENTS_OBSERVED)
        if self.agent_sorting_method in ['closest_last', 'closest_first']:
            # where "first" == closest
            sorted_sorting_criteria = sorted(sorting_criteria, key = lambda x: (x[1], x[2]))
        elif self.agent_sorting_method in ['time_to_impact']:
            # where "first" == lowest time-to-impact
            sorted_sorting_criteria = sorted(sorting_criteria, key = lambda x: (-x[3], -x[1], x[2]))
        clipped_sorting_criteria = sorted_sorting_criteria[:self.max_num_other_agents_observed]

        # Then sort those N agents by the preferred ordering scheme
        if self.agent_sorting_method == "closest_last":
            # sort by inverse distance away, then by lateral position
            sorted_dists = sorted(clipped_sorting_criteria, key = lambda x: (-x[1], x[2]))
        elif self.agent_sorting_method == "closest_first":
            # sort by distance away, then by lateral position
            sorted_dists = sorted(clipped_sorting_criteria, key = lambda x: (x[1], x[2]))
        elif self.agent_sorting_method == "time_to_impact":
            # sort by time_to_impact, break ties by distance away, then by lateral position (e.g. in case inf TTC)
            sorted_dists = sorted(clipped_sorting_criteria, key = lambda x: (-x[3], -x[1], x[2]))
        else:
            raise ValueError("Did not supply proper self.agent_sorting_method in Agent.py.")

        clipped_sorted_inds = [x[0] for x in sorted_dists]
        return clipped_sorted_inds


    def sense(self, agents, agent_index, top_down_map=None):
        """ Go through each agent in the environment, and compute its relative position, vel, etc. and put into an array

        This is a denser measurement of other agents' states vs. a LaserScan or OccupancyGrid

        Args:
            agents (list): all :class:`~gym_collision_avoidance.envs.agent.Agent` in the environment
            agent_index (int): index of this agent (the one with this sensor) in :code:`agents`
            top_down_map (2D np array): binary image with 0 if that pixel is free space, 1 if occupied (not used!)

        Returns:
            other_agents_states (np array): (max_num_other_agents_observed x 7) the 7 states about each other agent, :code:`[p_parallel_ego_frame, p_orthog_ego_frame, v_parallel_ego_frame, v_orthog_ego_frame, other_agent.radius, combined_radius, dist_2_other]`

        """
        host_agent = agents[agent_index]
        other_agent_dists = {}
        sorted_pairs = sorted(other_agent_dists.items(),
                              key=operator.itemgetter(1))

        sorting_criteria = []
        for i, other_agent in enumerate(agents):
            if other_agent.id == host_agent.id:
                continue
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - \
                host_agent.pos_global_frame
            p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_prll)
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_orth)
            dist_between_agent_centers = vec2_l2_norm(rel_pos_to_other_global_frame)
            dist_2_other = dist_between_agent_centers - host_agent.radius - other_agent.radius
            combined_radius = host_agent.radius + other_agent.radius

            if dist_between_agent_centers > Config.SENSING_HORIZON:
                # print("Agent too far away")
                continue

            if self.agent_sorting_method != "time_to_impact":
                time_to_impact = None
            else:
                time_to_impact = compute_time_to_impact(host_agent.pos_global_frame,
                                                        other_agent.pos_global_frame,
                                                        host_agent.vel_global_frame,
                                                        other_agent.vel_global_frame,
                                                        combined_radius)

            sorting_criteria.append([i, round(dist_2_other,2), p_orthog_ego_frame, time_to_impact])

        clipped_sorted_inds = self.get_clipped_sorted_inds(sorting_criteria)
        clipped_sorted_agents = [agents[i] for i in clipped_sorted_inds]

        other_agents_states = np.zeros((Config.MAX_NUM_OTHER_AGENTS_OBSERVED, 7))
        other_agent_count = 0
        for other_agent in clipped_sorted_agents:
            if other_agent.id == host_agent.id:
                continue
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - \
                host_agent.pos_global_frame
            p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame,
                                          host_agent.ref_prll)
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame,
                                        host_agent.ref_orth)
            v_parallel_ego_frame = np.dot(other_agent.vel_global_frame,
                                          host_agent.ref_prll)
            v_orthog_ego_frame = np.dot(other_agent.vel_global_frame,
                                        host_agent.ref_orth)
            dist_2_other = np.linalg.norm(rel_pos_to_other_global_frame) - \
                host_agent.radius - other_agent.radius
            combined_radius = host_agent.radius + other_agent.radius

            other_obs = np.array([p_parallel_ego_frame,
                                  p_orthog_ego_frame,
                                  v_parallel_ego_frame,
                                  v_orthog_ego_frame,
                                  other_agent.radius,
                                  combined_radius,
                                  dist_2_other])
            
            if other_agent_count == 0:
                host_agent.other_agent_states[:] = other_obs

            other_agents_states[other_agent_count,:] = other_obs
            other_agent_count += 1

        host_agent.num_other_agents_observed = other_agent_count

        return other_agents_states
