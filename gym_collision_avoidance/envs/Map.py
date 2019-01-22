import numpy as np

class Map():
    def __init__(self, x_width, y_width, grid_cell_size):
        self.x_width = x_width
        self.y_width = y_width
        self.grid_cell_size = grid_cell_size
        self.static_map = np.zeros((int(self.x_width/self.grid_cell_size),int(self.y_width/self.grid_cell_size)), dtype=bool)
        self.origin_coords = np.array([(self.x_width/2.)/self.grid_cell_size, (self.y_width/2.)/self.grid_cell_size])
        self.map = None

    def world_coordinates_to_map_indices(self, pos):
        gx = int(np.floor(self.origin_coords[0]-pos[1]/self.grid_cell_size))
        gy = int(np.floor(self.origin_coords[1]+pos[0]/self.grid_cell_size))
        grid_coords = np.array([gx, gy])
        print("pos:", pos)
        print("gx,gy:", grid_coords)
        return grid_coords

    def add_agents_to_map(self, agents):
        print('-----')
        self.map = self.static_map.copy()
        for agent in agents:
            gx, gy = self.world_coordinates_to_map_indices(agent.pos_global_frame)
            print(agent.id, agent.pos_global_frame, gx, gy)
            if gx < 0 or gy < 0 or gx > self.map.shape[0] or gy > self.map.shape[1]:
                continue # agent outside bounds of map, so don't plot it
            self.map[gx, gy] = 1
