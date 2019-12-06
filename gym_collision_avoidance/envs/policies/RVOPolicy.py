import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.util import *
import rvo2

import matplotlib.pyplot as plt

class RVOPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="RVO")

        self.dt = Config.DT
        neighbor_dist = Config.SENSING_HORIZON
        max_neighbors = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT

        self.has_fixed_speed = False
        self.heading_noise = False

        self.max_delta_heading = np.pi/6
        
        # TODO share this parameter with environment
        time_horizon = 5.0 # NOTE: bjorn used 1.0 in training for corl19
        # Initialize RVO simulator
        self.sim = rvo2.PyRVOSimulator(timeStep=self.dt, neighborDist=neighbor_dist, 
            maxNeighbors=max_neighbors, timeHorizon=time_horizon, 
            timeHorizonObst=time_horizon, radius=0.0, 
            maxSpeed=0.0)

        self.is_init = False

    def init(self):
        state_dim = 2
        self.pos_agents = np.empty((self.n_agents, state_dim))
        self.vel_agents = np.empty((self.n_agents, state_dim))
        self.goal_agents = np.empty((self.n_agents, state_dim))
        self.pref_vel_agents = np.empty((self.n_agents, state_dim))
        self.pref_speed_agents = np.empty((self.n_agents))
        
        self.rvo_agents = [None]*self.n_agents

        # Init simulation
        for a in range(self.n_agents):
            self.rvo_agents[a] = self.sim.addAgent((0,0))
        
        # Set RVO agent's collaborativity
        # self.sim.setAgentCollabCoeff(self.rvo_agents[1], 0.5)
        #sim.setAgentCollabCoeff(a1, 0.)

        self.is_init = True

    def find_next_action(self, obs, agents, agent_index):
        # Initialize vectors on first call to infer number of agents
        if not self.is_init:
            self.n_agents = len(agents)
            self.init()

        # Share all agent positions and preferred velocities from environment with RVO simulator
        for a in range(self.n_agents):
            # Copy current agent positions, goal and preferred speeds into np arrays
            self.pos_agents[a,:] = agents[a].pos_global_frame[:]
            self.vel_agents[a,:] = agents[a].vel_global_frame[:]
            self.goal_agents[a,:] = agents[a].goal_global_frame[:]
            self.pref_speed_agents[a] = agents[a].pref_speed

            # Calculate preferred velocity
            # Assumes non RVO agents are acting like RVO agents
            self.pref_vel_agents[a,:] = self.goal_agents[a,:] - self.pos_agents[a,:]
            self.pref_vel_agents[a,:] = self.pref_speed_agents[a] / np.linalg.norm(self.pref_vel_agents[a,:]) * self.pref_vel_agents[a,:]

            # Set agent positions and velocities in RVO simulator
            self.sim.setAgentMaxSpeed(self.rvo_agents[a], agents[a].pref_speed)
            self.sim.setAgentRadius(self.rvo_agents[a], (1+5e-2)*agents[a].radius)
            self.sim.setAgentPosition(self.rvo_agents[a], tuple(self.pos_agents[a,:]))
            self.sim.setAgentVelocity(self.rvo_agents[a], tuple(self.vel_agents[a,:]))
            self.sim.setAgentPrefVelocity(self.rvo_agents[a], tuple(self.pref_vel_agents[a,:]))

        # Execute one step in the RVO simulator
        self.sim.doStep()

        # Calculate desired change of heading
        self.new_rvo_pos = self.sim.getAgentPosition(self.rvo_agents[agent_index])[:]
        deltaPos = self.new_rvo_pos - self.pos_agents[agent_index,:]
        p1 = deltaPos
        p2 = np.array([1,0]) # Angle zero is parallel to x-axis
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        new_heading_global_frame = (ang1 - ang2) % (2 * np.pi)
        delta_heading = wrap(new_heading_global_frame - agents[agent_index].heading_global_frame)
            
        # Calculate desired speed
        pref_speed = 1/self.dt * np.linalg.norm(deltaPos)

        # Limit the turning rate: stop and turn in place if exceeds
        if abs(delta_heading) > self.max_delta_heading:
            delta_heading = np.sign(delta_heading)*self.max_delta_heading
            pref_speed = 0.

        # Ignore speed
        if self.has_fixed_speed:
            pref_speed = self.max_speed

        # Add noise
        if self.heading_noise:
            delta_heading = delta_heading + np.random.normal(0,0.5)

        action = np.array([pref_speed, delta_heading])
        return action
