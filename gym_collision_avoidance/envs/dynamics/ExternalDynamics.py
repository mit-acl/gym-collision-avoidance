import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics

class ExternalDynamics(Dynamics):
    def __init__(self, agent):
        super().__init__(agent)

    def step(self, action, dt):
        return
