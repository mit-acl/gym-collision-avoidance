import numpy as np
import os
from gym_collision_avoidance.envs.policies.Policy import Policy
from gym_collision_avoidance.envs.CADRL.scripts.multi import nn_navigation_value_multi as nn_nav
from gym_collision_avoidance.envs import util

class CADRLPolicy(Policy):
    def __init__(self):
        Policy.__init__(self)

        num_agents = 4
        file_dir = os.path.dirname(os.path.realpath(__file__)) + '/../CADRL/scripts/multi'

        # load value_net
        mode = 'rotate_constr'; passing_side = 'right'; iteration = 1300
        filename="%d_agents_policy_iter_"%num_agents + str(iteration) + ".p"
        self.value_net = nn_nav.load_NN_navigation_value(file_dir, num_agents, mode, passing_side, filename=filename, ifPrint=False)

    def find_next_action(self, obs, agents, i):
        host_agent = agents[i]
        other_agents = agents[:i]+agents[i+1:]
        agent_state = self.convert_host_state_to_cadrl_state(host_agent)
        other_agents_state = self.convert_other_states_to_cadrl_state(other_agents)
        # value = self.value_net.find_states_values(agent_state, other_agents_state)
        action = self.value_net.find_next_action(agent_state, other_agents_state)
        action[0] /= host_agent.pref_speed
        action[1] = util.wrap(action[1]-host_agent.heading_global_frame)

        return action

    def convert_host_state_to_cadrl_state(self, agent):
        # Convert this repo's state representation format into the legacy cadrl format
        # for the host agent

        # rel pos, rel vel, size
        x = agent.pos_global_frame[0]; y = agent.pos_global_frame[1]
        v_x = agent.vel_global_frame[0]; v_y = agent.vel_global_frame[1]
        radius = agent.radius; turning_dir = 0.0
        heading_angle = agent.heading_global_frame
        pref_speed = agent.pref_speed
        goal_x = agent.goal_global_frame[0]; goal_y = agent.goal_global_frame[1]
        
        agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, \
            goal_x, goal_y, radius, turning_dir])

        return agent_state

    def convert_other_states_to_cadrl_state(self, other_agents):
        # Convert this repo's state representation format into the legacy cadrl format
        # for the other agents in the environment

        if len(other_agents) > 3:
            print("CADRL ISN'T DESIGNED TO HANDLE > 4 AGENTS")

        other_agents_state = []
        for agent in other_agents:
            x = agent.pos_global_frame[0]; y = agent.pos_global_frame[1]
            v_x = agent.vel_global_frame[0]; v_y = agent.vel_global_frame[1]
            radius = agent.radius; turning_dir = 0.0
            # helper fields: # TODO: why are these here? these are hidden states
            heading_angle = np.arctan2(v_y, v_x)
            # pref_speed = np.linalg.norm(np.array([v_x, v_y]))
            # goal_x = x - 5.0; goal_y = y - 5.0 
            pref_speed = agent.pref_speed
            goal_x = agent.goal_global_frame[0]; goal_y = agent.goal_global_frame[1]

            other_agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, \
                    goal_x, goal_y, radius, turning_dir])
            other_agents_state.append(other_agent_state)
        return other_agents_state