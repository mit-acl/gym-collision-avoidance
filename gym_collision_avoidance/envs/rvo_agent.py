from gym_collision_avoidance.envs.agent import Agent
import numpy as np

from gym_collision_avoidance.envs.util import *
from gym_collision_avoidance.envs.config import Config
import rvo2
import sys
# Executes RVO2 policy from Reciprocal Collision Avoidance for Real-Time Multi-Agent Simulation, 
# Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, and Dinesh Manocha
# Department of Computer Science, University of North Carolina at Chapel Hill
# Implementation from: https://github.com/sybrenstuvel/Python-RVO2

class RVOAgent(Agent):
    def __init__(self, start_x, start_y, goal_x, goal_y, radius, pref_speed, initial_heading, id):
        Agent.__init__(self, start_x, start_y, goal_x, goal_y, radius, pref_speed, initial_heading, id)
        self.policy_type = "RVO"

        self.dt = Config.DT
        neighbor_dist = Config.SENSING_HORIZON
        max_neighbors = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        max_speed = pref_speed
        self.has_fixed_speed = False
        # TODO share this parameter with environment
        time_horizon = 1.0#self.dt * 5
        # Initialize RVO simulator
        self.sim = rvo2.PyRVOSimulator(timeStep=self.dt, neighborDist=neighbor_dist, 
            maxNeighbors=max_neighbors, timeHorizon=time_horizon, 
            timeHorizonObst=time_horizon, radius=radius, 
            maxSpeed=max_speed)

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
        self.sim.setAgentCollabCoeff(self.rvo_agents[1], 0.5)
        #sim.setAgentCollabCoeff(a1, 0.)

        self.is_init = True

    def find_next_action(self, agents):
        # Initialize vectors on first call to infer number of agents
        if not self.is_init:
          self.n_agents = len(agents)
          self.init()

        # Share all agent positions and preferred velocities from environment with RVO simulator
        for a in range(self.n_agents):
            # TODO do all of this with the observation vector in ego coordinates...

            # Copy current agent positions, goal and preferred speeds into np arrays
            self.pos_agents[a,:] = agents[a].pos_global_frame[:]
            self.goal_agents[a,:] = agents[a].goal_global_frame[:]
            self.pref_speed_agents[a] = agents[a].pref_speed

            # Calculate preferred velocity
            # Assumes non RVO agents are acting like RVO agents
            self.pref_vel_agents[a,:] = self.goal_agents[a,:] - self.pos_agents[a,:]
            self.pref_vel_agents[a,:] = self.pref_speed_agents[a] / np.linalg.norm(self.pref_vel_agents[a,:]) * self.pref_vel_agents[a,:]

            # TODO CALCULATE OTHER AGENTS PREF VELOCITY, BASED ON THEIR CURRENT VELOCITY?

            # Set agent positions and velocities in RVO simulator
            self.sim.setAgentPosition(self.rvo_agents[a], tuple(self.pos_agents[a,:]))
            self.sim.setAgentPrefVelocity(self.rvo_agents[a], tuple(self.pref_vel_agents[a,:]))

        #print('pos', self.id, self.pos_agents[self.id,:])
        #print('goal', self.id, self.goal_agents[self.id,:])
        #print('pref speed', self.pref_speed_agents[self.id])
        #print('pref vel', self.pref_vel_agents[self.id,:])
        #print('num agents', self.sim.getNumAgents())
        
        # Execute one step in the RVO simulator
        self.sim.doStep()

        # Calculate desired change of heading
        deltaPos = self.sim.getAgentPosition(self.rvo_agents[self.id])[:] - self.pos_agents[self.id,:]
        p1 = deltaPos
        p2 = np.array([1,0]) # Angle zero is parallel to x-axis
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        new_heading_global_frame = (ang1 - ang2) % (2 * np.pi)
        delta_heading = wrap(new_heading_global_frame - self.heading_global_frame)
            
        # Calculate desired speed
        pref_speed = 1/self.dt * np.linalg.norm(deltaPos)

        action = np.array([pref_speed, delta_heading])
        return action