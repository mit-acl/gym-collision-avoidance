import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor

class LaserScanSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.num_beams = 5
        self.range_resolution = 0.1
        self.max_range = 10 # meters
        self.min_range = 0 # meters
        self.min_angle = -np.pi/2
        self.max_angle = np.pi/2

        self.angles = np.linspace(self.min_angle, self.max_angle, self.num_beams)
        self.ranges = np.arange(self.min_range, self.max_range, self.range_resolution)

    def sense(self, agents, agent_index, top_down_map):
        host_agent = agents[agent_index]

        [pi, pj], _ = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame)
        ranges = self.max_range*np.ones_like(self.angles)
        for angle_i, angle in enumerate(self.angles+host_agent.heading_global_frame):
        	for r in self.ranges:
        		[i, j], _ = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame+np.array([r*np.cos(angle), r*np.sin(angle)]))
        		if r < host_agent.radius or i >= top_down_map.map.shape[0] or j >= top_down_map.map.shape[1]:
        			continue
        		if top_down_map.map[i, j]:
        			ranges[angle_i] = r
        			break
        print(ranges)
        return ranges

