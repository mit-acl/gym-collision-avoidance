import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor

class OccupancyGridSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.num_beams = 5
        self.max_range = 10 # meters
        self.min_range = 0 # meters

    def sense(self, agents, agent_index, top_down_map):
        host_agent = agents[agent_index]
        raise NotImplementedError

