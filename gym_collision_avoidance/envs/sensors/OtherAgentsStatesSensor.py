import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import compute_time_to_impact, vec2_l2_norm
import operator

class OtherAgentsStatesSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.name = 'other_agents_states'

    def sense(self, agents, agent_index, top_down_map):
        host_agent = agents[agent_index]
        other_agent_dists = {}
        # for i, other_agent in enumerate(agents):
        #     if other_agent.id == host_agent.id:
        #         continue
            # project other elements onto the new reference frame
            # rel_pos_to_other_global_frame = other_agent.pos_global_frame - \
            #     host_agent.pos_global_frame
            # dist_between_agent_centers = np.linalg.norm(
            #         rel_pos_to_other_global_frame)
            # dist_2_other = dist_between_agent_centers - host_agent.radius - \
            #     other_agent.radius
            # if dist_between_agent_centers > Config.SENSING_HORIZON:
            #     # print "Agent too far away"
            #     continue
            # other_agent_dists[i] = dist_2_other
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

            if Config.AGENT_SORTING_METHOD != "time_to_impact":
                time_to_impact = None
            else:
                time_to_impact = compute_time_to_impact(self.pos_global_frame,
                                                        other_agent.pos_global_frame,
                                                        self.vel_global_frame,
                                                        other_agent.vel_global_frame,
                                                        combined_radius)

            sorting_criteria.append([i, round(dist_2_other,2), p_orthog_ego_frame, time_to_impact])

        if Config.AGENT_SORTING_METHOD == "closest_last":
            # sort by inverse distance away, then by lateral position
            sorted_dists = sorted(sorting_criteria, key = lambda x: (-x[1], x[2]))
        elif Config.AGENT_SORTING_METHOD == "closest_first":
            # sort by distance away, then by lateral position
            sorted_dists = sorted(sorting_criteria, key = lambda x: (x[1], x[2]))
        elif Config.AGENT_SORTING_METHOD == "time_to_impact":
            # sort by time_to_impact, break ties by distance away, then by lateral position (e.g. in case inf TTC)
            sorted_dists = sorted(sorting_criteria, key = lambda x: (-x[3], -x[1], x[2]))
        else:
            raise ValueError("Did not supply proper Config.AGENT_SORTING_METHOD in Agent.py.")


        sorted_inds = [x[0] for x in sorted_dists]
        clipped_sorted_inds = sorted_inds[-Config.MAX_NUM_OTHER_AGENTS_OBSERVED:]
        clipped_sorted_agents = [agents[i] for i in clipped_sorted_inds]

        other_agents_states = np.zeros((Config.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT, 7))
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
