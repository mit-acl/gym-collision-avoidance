import numpy as np
import os
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import nn_navigation_value_multi as nn_nav
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import util

class CADRLPolicy(InternalPolicy):
    """ Re-purposed from: Socially Aware Motion Planning with Deep Reinforcement Learning

    Loads a pre-traned SA-CADRL 4-agent network (with no social norm preference LHS/RHS).
    Some methods to convert the gym agent representation to the numpy arrays used in the old code.

    """
    def __init__(self):
        InternalPolicy.__init__(self, str="CADRL")

        num_agents = 4
        file_dir = os.path.dirname(os.path.realpath(__file__)) + '/CADRL/scripts/multi'

        # load value_net
        # mode = 'rotate_constr'; passing_side = 'right'; iteration = 1300
        mode = 'no_constr'; passing_side = 'none'; iteration = 1000
        filename="%d_agents_policy_iter_"%num_agents + str(iteration) + ".p"
        self.value_net = nn_nav.load_NN_navigation_value(file_dir, num_agents, mode, passing_side, filename=filename, ifPrint=False)

    def find_next_action(self, obs, agents, i):
        """ Converts environment's agents representation to CADRL format, then queries NN

        Args:
            obs (dict): ignored
            agents (list): of :class:`~gym_collision_avoidance.envs.agent.Agent` objects
            i (int): index of agents list corresponding to this agent

        Returns:
            commanded [heading delta, speed]

        """
        host_agent, agent_state, other_agents_state, other_agents_actions = self.parse_agents(agents, i)
        action = self.query_and_rescale_action(host_agent, agent_state, other_agents_state, other_agents_actions)
        return action

    def find_next_action_and_value(self, obs, agents, i):
        """ Same as find_next_action but also queries value fn """
        host_agent, agent_state, other_agents_state, other_agents_actions = self.parse_agents(agents, i)
        action = self.query_and_rescale_action(host_agent, agent_state, other_agents_state, other_agents_actions)
        value = self.value_net.find_states_values(agent_state, other_agents_state)
        return action, value

    def parse_agents(self, agents, i):
        """ Convert from gym env representation of agents to CADRL's representation.

        Args:
            obs (dict): ignored
            agents (list): of :class:`~gym_collision_avoidance.envs.agent.Agent` objects
            i (int): index of agents list corresponding to this agent

        Returns:
            host_agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): this agent
            agent_state (np array): CADRL representation of this agent's state
            other_agents_state (np array): CADRL repr. of other agents' states
            other_agents_actions (np array): CADRL repr. of other agents' current actions

        """
        host_agent = agents[i]
        other_agents = agents[:i]+agents[i+1:]
        agent_state = self.convert_host_agent_to_cadrl_state(host_agent)
        other_agents_state, other_agents_actions = self.convert_other_agents_to_cadrl_state(host_agent, other_agents)
        return host_agent, agent_state, other_agents_state, other_agents_actions

    def query_and_rescale_action(self, host_agent, agent_state, other_agents_state, other_agents_actions):
        """ If there's nobody around, just go straight to goal, otherwise query DNN and make heading action an offset from current heading

        """
        if len(other_agents_state) > 0:
            action = self.value_net.find_next_action(agent_state, other_agents_state, other_agents_actions)
            # action[0] /= host_agent.pref_speed
            action[1] = util.wrap(action[1]-host_agent.heading_global_frame)
        else:
            action = np.array([1.0, -self.heading_ego_frame])
        return action

    def convert_host_agent_to_cadrl_state(self, agent):
        """ Convert this repo's state representation format into the legacy cadrl format for the host agent 

        Args:
            agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): this agent

        Returns:
            10-element (np array) describing current state

        """

        # rel pos, rel vel, size
        x = agent.pos_global_frame[0]; y = agent.pos_global_frame[1]
        v_x = agent.vel_global_frame[0]; v_y = agent.vel_global_frame[1]
        radius = agent.radius; turning_dir = agent.turning_dir
        heading_angle = agent.heading_global_frame
        pref_speed = agent.pref_speed
        goal_x = agent.goal_global_frame[0]; goal_y = agent.goal_global_frame[1]
        
        agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, \
            goal_x, goal_y, radius, turning_dir])

        return agent_state

    def convert_other_agents_to_cadrl_state(self, host_agent, other_agents):
        """ Convert this repo's state representation format into the legacy cadrl format
        for the other agents in the environment.

        Filtering other agents' velocities was crucial to replicate SA-CADRL results

        Args:
            host_agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): this agent
            other_agents (list): of all the other :class:`~gym_collision_avoidance.envs.agent.Agent` objects

        Returns:
            - (3 x 10) np array (this cadrl can handle 3 other agents), each has 10-element state vector
            - (3 x 2) np array of other agents' filtered velocities
        """        
        # if len(other_agents) > 3:
        #     print("CADRL ISN'T DESIGNED TO HANDLE > 4 AGENTS")

        # This is a hack that CADRL was not trained to handle (only trained on 4 agents)
        other_agent_dists = []
        for i, other_agent in enumerate(other_agents):
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_orth)
            dist_between_agent_centers = np.linalg.norm(rel_pos_to_other_global_frame)
            dist_2_other = dist_between_agent_centers - host_agent.radius - other_agent.radius
            if dist_between_agent_centers > Config.SENSING_HORIZON:
                # print "Agent too far away"
                continue
            other_agent_dists.append([i,round(dist_2_other,2),p_orthog_ego_frame])
        sorted_dists = sorted(other_agent_dists, key = lambda x: (-x[1], x[2]))
        sorted_inds = [x[0] for x in sorted_dists]
        clipped_sorted_inds = sorted_inds[-min(Config.MAX_NUM_OTHER_AGENTS_OBSERVED,3):]
        clipped_sorted_agents = [other_agents[i] for i in clipped_sorted_inds]

        agents = clipped_sorted_agents

        other_agents_state = []
        other_agents_actions = []
        for agent in agents:
            x = agent.pos_global_frame[0]; y = agent.pos_global_frame[1]
            v_x = agent.vel_global_frame[0]; v_y = agent.vel_global_frame[1]
            radius = agent.radius; turning_dir = agent.turning_dir
            # helper fields: # TODO: why are these here? these are hidden states - CADRL uses the raw agent states to convert to local representation internally
            heading_angle = agent.heading_global_frame
            pref_speed = agent.pref_speed
            goal_x = agent.goal_global_frame[0]; goal_y = agent.goal_global_frame[1]

            # experimental - filter velocities and pass as other_agents_actions
            # if np.shape(agent.global_state_history)[0] > 3:
            if True:
                past_vel = agent.past_global_velocities[-2:,:]
                dt_past_vec = Config.DT*np.ones((2))
                filtered_actions_theta = util.filter_vel(dt_past_vec, past_vel)
                other_agents_actions.append(filtered_actions_theta)
            else:
                other_agents_actions = None

            other_agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, \
                    goal_x, goal_y, radius, turning_dir])
            other_agents_state.append(other_agent_state)
        return other_agents_state, other_agents_actions