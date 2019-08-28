import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor

import matplotlib.pyplot as plt

class OccupancyGridSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.x_width = 5
        self.y_width = 5
        self.grid_cell_size = 0.01 # currently ignored

    def sense(self, agents, agent_index, top_down_map):

        # Grab (i,j) coordinates of the upper right and lower left corner of the desired OG map, within the entire map
        host_agent = agents[agent_index]
        [map_i_high, map_j_low], _ = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame-np.array([self.x_width/2., self.y_width/2.]))
        [map_i_low, map_j_high], _ = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame+np.array([self.x_width/2., self.y_width/2.]))

        # Assume areas outside static_map should be filled with zeros
        og_map = np.zeros((int(self.y_width/top_down_map.grid_cell_size), int(self.x_width/top_down_map.grid_cell_size)), dtype=bool)

        if map_i_low >= top_down_map.map.shape[0] or map_i_high < 0 or map_j_low >= top_down_map.map.shape[1] or map_j_high < 0:
            # skip rest ==> og_map and top_down_map have no overlap
            print("*** no overlap btwn og_map and top_down_map ***")
            print("*** map dims:", map_i_low, map_i_high, map_j_low, map_j_high)
            return og_map

        # super crappy logic to handle when the OG map partially overlaps with the map 
        if map_i_low < 0:
            og_i_low = -map_i_low
            og_i_high = og_map.shape[0]
        elif map_i_high >= top_down_map.map.shape[0]:
            og_i_low = 0
            og_i_high = og_map.shape[0] - (map_i_high - top_down_map.map.shape[0])
        else:
            og_i_low = 0
            og_i_high = og_map.shape[0]

        if map_j_low < 0:
            og_j_low = -map_j_low
            og_j_high = og_map.shape[1]
        elif map_j_high >= top_down_map.map.shape[1]:
            og_j_low = 0
            og_j_high = og_map.shape[1] - (map_j_high - top_down_map.map.shape[1])
        else:
            og_j_low = 0
            og_j_high = og_map.shape[1]

        # Don't grab a map index outside the map's boundaries
        map_i_low = np.clip(map_i_low, 0, top_down_map.map.shape[0])
        map_i_high = np.clip(map_i_high, 0, top_down_map.map.shape[0])
        map_j_low = np.clip(map_j_low, 0, top_down_map.map.shape[1])
        map_j_high = np.clip(map_j_high, 0, top_down_map.map.shape[1])

        # Fill the correct OG map indices with the section of the map that has been selected
        og_map[og_i_low:og_i_high, og_j_low:og_j_high] = top_down_map.map[map_i_low:map_i_high, map_j_low:map_j_high]
        resized_og_map = self.resize(og_map)
        return resized_og_map

    def resize(self, og_map):
        resized_og_map = og_map.copy()
        return resized_og_map

if __name__ == '__main__':
    from gym_collision_avoidance.envs.Map import Map
    from gym_collision_avoidance.envs.agent import Agent
    from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
    from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics

    top_down_map = Map(x_width=10, y_width=10, grid_cell_size=0.1)
    agents = [Agent(0, 3.05, 10, 10, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamics, [], 0)]
    top_down_map.add_agents_to_map(agents)
    og = OccupancyGridSensor()
    og_map = og.sense(agents, 0, top_down_map)

    print(og_map)