import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor

import matplotlib.pyplot as plt

class OccupancyGridSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.x_width = 3
        self.y_width = 5
        self.grid_cell_size = 0.01 # currently ignored

    def sense(self, agents, agent_index, top_down_map):
        host_agent = agents[agent_index]
        gx_low, gy_low = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame-np.array([self.x_width/2., self.y_width/2.]))
        gx_high, gy_high = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame+np.array([self.x_width/2., self.y_width/2.]))

        og_map = np.zeros((int(self.x_width/top_down_map.grid_cell_size), int(self.y_width/top_down_map.grid_cell_size)), dtype=bool)

        if gx_low >= top_down_map.map.shape[0] or gx_high < 0 or gy_low >= top_down_map.map.shape[1] or gy_high < 0:
            print("*** no overlap btwn og_map and top_down_map ***")
            print("*** map dims:", gx_low, gx_high, gy_low, gy_high)

            # skip rest ==> og_map and top_down_map have NO OVERLAP
            # TODO: resize this upon creation!
            og_map = np.zeros((int(self.x_width/top_down_map.grid_cell_size), int(self.y_width/top_down_map.grid_cell_size)), dtype=bool)
            return og_map

        if gx_low < 0:
            og_x_low = -gx_low
            og_x_high = og_map.shape[0]-1
        elif gx_high >= top_down_map.map.shape[0]:
            og_x_low = 0
            og_x_high = top_down_map.map.shape[0] - og_x_low
        else:
            og_x_low = 0
            og_x_high = og_map.shape[0]



        if gy_low < 0:
            og_y_low = -gy_low
            og_y_high = og_map.shape[1]-1
        elif gy_high >= top_down_map.map.shape[1]:
            og_y_low = 0
            og_y_high = top_down_map.map.shape[1] - og_y_low
        else:
            og_y_low = 0
            og_y_high = og_map.shape[1]

        top_down_map.map[gx_low:gx_high, gy_low:gy_high] = 1
        plt.title("top down map region of interest")
        plt.imshow(top_down_map.map)
        plt.pause(1)

        print("map dims:", gx_low, gx_high, gy_low, gy_high)
        gx_low = np.clip(gx_low, 0, top_down_map.map.shape[0])
        gx_high = np.clip(gx_high, 0, top_down_map.map.shape[0])
        gy_low = np.clip(gy_low, 0, top_down_map.map.shape[1])
        gy_high = np.clip(gy_high, 0, top_down_map.map.shape[1])

        print("og dims:", og_x_low, og_x_high, og_y_low, og_y_high)
        print("map dims:", gx_low, gx_high, gy_low, gy_high)

        og_map[og_x_low:og_x_high, og_y_low:og_y_high] = top_down_map.map[gx_low:gx_high, gy_low:gy_high]
        resized_og_map = self.resize(og_map)
        return resized_og_map

    def resize(self, og_map):
        resized_og_map = og_map.copy()
        return resized_og_map