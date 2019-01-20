import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor

class OccupancyGridSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)

    def sense(self, agents, top_down_map):
        raise NotImplementedError
