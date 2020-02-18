import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics

class ExternalDynamics(Dynamics):
    def __init__(self, agent):
        Dynamics.__init__(self, agent)

    def step(self, action, dt):
        return
