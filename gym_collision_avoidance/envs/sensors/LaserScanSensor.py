import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs.config import Config
import matplotlib.pyplot as plt

class LaserScanSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.name = 'laserscan'
        self.num_beams = Config.LASERSCAN_LENGTH
        self.range_resolution = 0.1
        self.max_range = 6 # meters
        self.min_range = 0 # meters
        self.min_angle = -np.pi/2
        self.max_angle = np.pi/2

        self.angles = np.linspace(self.min_angle, self.max_angle, self.num_beams)
        self.ranges = np.arange(self.min_range, self.max_range, self.range_resolution)

        self.debug = False

    def sense(self, agents, agent_index, top_down_map):
        host_agent = agents[agent_index]

        if self.debug:
            lidar_map = top_down_map.map.copy()

        ego_agent_mask = top_down_map.get_agent_mask(host_agent.pos_global_frame, host_agent.radius)
        ranges = self.max_range*np.ones_like(self.angles)
        for angle_i, angle in enumerate(self.angles+host_agent.heading_global_frame):
            for r in self.ranges:
                pos = host_agent.pos_global_frame+np.array([r*np.cos(angle), r*np.sin(angle)])
                [i, j], in_map = top_down_map.world_coordinates_to_map_indices(pos)
                if self.debug and in_map:
                    lidar_map[i,j]=1
                if not in_map or ego_agent_mask[i,j]:
                    continue
                if top_down_map.map[i, j]:
                    if self.debug:
                        print("Object at {}!!".format(r))
                    ranges[angle_i] = r
                    break
        if self.debug:
            plt.imshow(lidar_map)
            plt.pause(1)
        return ranges

