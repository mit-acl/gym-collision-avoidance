from gym_collision_avoidance.envs.agent import Agent
import numpy as np
import time

from gym_collision_avoidance.envs.CADRL.scripts.multi import nn_navigation_value_multi as nn_nav
from gym_collision_avoidance.envs import util

class CADRLAgent(Agent):
    def __init__(self, start_x, start_y, goal_x, goal_y, radius, pref_speed, initial_heading, id):
        Agent.__init__(self, start_x, start_y, goal_x, goal_y, radius, pref_speed, initial_heading, id)

        num_agents = 4
        file_dir = '/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/scripts/multi'
        # file_dir = os.path.dirname(os.path.realpath(__file__))

        # load value_net
        # mode = 'no_constr'; passing_side = 'none'; iteration = 500
        # mode = 'no_constr'; passing_side = 'left'; iteration = 500
        # mode = 'no_constr'; passing_side = 'right'; iteration = 500
        # mode = 'rotate_constr'; passing_side = 'none'; iteration = 1000
        # mode = 'rotate_constr'; passing_side = 'left'; iteration = 500
        # mode = 'rotate_constr'; passing_side = 'right'; iteration = 600
        mode = 'rotate_constr'; passing_side = 'right'; iteration = 1300
        # mode = 'no_constr'; passing_side = 'none'
        # iteration = 1300
        filename="%d_agents_policy_iter_"%num_agents + str(iteration) + ".p"
        # filename="/%d_agents_policy_iter_"%num_agents + str(2000) + ".p"
        self.value_net = nn_nav.load_NN_navigation_value(file_dir, num_agents, mode, passing_side, filename=filename, ifPrint=False)

        self.policy_type = "CADRL"

    def find_next_action(self, agents):
        # print "[find_next_action] Agent %i:" %(self.id)
        agent_state = self.convert_state_to_cadrl_state()
        other_agents_state = self.cadrl_other_agents_state(agents)
        # print agent_state
        # print other_agents_state
        # value = self.value_net.find_states_values(agent_state, other_agents_state)
        action = self.value_net.find_next_action(agent_state, other_agents_state)
        action[0] /= self.pref_speed
        action[1] = util.wrap(action[1]-self.heading_global_frame)

        # return action, value
        return action

    def convert_state_to_cadrl_state(self):
        # Convert this repo's state representation format into the legacy cadrl format
        # for the host agent

        # rel pos, rel vel, size
        x = self.pos_global_frame[0]; y = self.pos_global_frame[1]
        v_x = self.vel_global_frame[0]; v_y = self.vel_global_frame[1]
        radius = self.radius; turning_dir = 0.0
        heading_angle = self.heading_global_frame
        pref_speed = self.pref_speed
        goal_x = self.goal_global_frame[0]; goal_y = self.goal_global_frame[1]
        
        agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, \
            goal_x, goal_y, radius, turning_dir])

        return agent_state

    def convert_cadrl_state_to_state(self, cadrl_state):
        # Convert the legacy cadrl format into this repo's state representation format 
        # for the host agent
        for agent in cadrl_state:
            print(agent)



        # rel pos, rel vel, size
        # x = self.pos_global_frame[0]; y = self.pos_global_frame[1]
        # v_x = self.vel_global_frame[0]; v_y = self.vel_global_frame[1]
        # radius = self.radius; turning_dir = 0.0
        # heading_angle = self.heading_global_frame
        # pref_speed = self.pref_speed
        # goal_x = self.goal_global_frame[0]; goal_y = self.goal_global_frame[1]
        
        # agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, \
        #     goal_x, goal_y, radius, turning_dir])

        return agent_state

    def cadrl_other_agents_state(self, agents):
        # Convert this repo's state representation format into the legacy cadrl format
        # for the other agents in the environment

        if len(agents) > 4:
            print("CADRL ISN'T DESIGNED TO HANDLE > 4 AGENTS")

        other_agents_state = []
        for agent in agents:
            if agent.id == self.id:
                continue
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