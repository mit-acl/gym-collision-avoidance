import numpy as np

class Map():
    def __init__(self, x_width, y_width, grid_cell_size):
        self.x_width = x_width
        self.y_width = y_width
        self.grid_cell_size = grid_cell_size
        self.static_map = np.zeros((int(self.x_width/self.grid_cell_size),int(self.y_width/self.grid_cell_size)), dtype=bool)
        self.map = None

    def world_coordinates_to_map_indices(self, pos):
        print("pos:", pos)
        gx = int((pos[0]+self.x_width/2.)/self.grid_cell_size)
        gy = int((pos[1]+self.y_width/2.)/self.grid_cell_size)
        print("gx,gy:", gx, gy)
        return gx, gy

    def add_agents_to_map(self, agents):
        print("add_agents_to_map")
        self.map = self.static_map.copy()
        for agent in agents:
            gx, gy = self.world_coordinates_to_map_indices(agent.pos_global_frame)
            try:
                self.map[gx, gy] = 1
            except IndexError:
                print("agent outside of map boundaries")
        print("add_agents_to_map done.")

