import numpy as np
import imageio
import scipy.misc

class Map():
    def __init__(self, x_width, y_width, grid_cell_size, map_filename=None):
        # Set desired map parameters (regardless of actual image file dims)
        self.x_width = x_width
        self.y_width = y_width
        self.grid_cell_size = grid_cell_size

        # Load the image file corresponding to the static map, and resize according to desired specs
        dims = (int(self.x_width/self.grid_cell_size),int(self.y_width/self.grid_cell_size))
        if map_filename is None:
            self.static_map = np.zeros(dims, dtype=bool)
        else:
            self.static_map = np.invert(imageio.imread(map_filename).astype(bool))
            if self.static_map.shape != dims:
                print("Resizing map from: {} to {}".format(self.static_map.shape, dims))
                self.static_map = scipy.misc.imresize(self.static_map, dims, interp='nearest')
        
        self.origin_coords = np.array([(self.x_width/2.)/self.grid_cell_size, (self.y_width/2.)/self.grid_cell_size])
        self.map = None # This will store the current static+dynamic map at each timestep

    def world_coordinates_to_map_indices(self, pos):
        gx = int(np.floor(self.origin_coords[0]-pos[1]/self.grid_cell_size))
        gy = int(np.floor(self.origin_coords[1]+pos[0]/self.grid_cell_size))
        grid_coords = np.array([gx, gy])
        # print("pos:", pos)
        # print("gx,gy:", grid_coords)
        return grid_coords

    def add_agents_to_map(self, agents):
        # print('-----')
        self.map = self.static_map.copy()
        for agent in agents:
            gx, gy = self.world_coordinates_to_map_indices(agent.pos_global_frame)
            # print(agent.id, agent.pos_global_frame, gx, gy)
            if gx < 0 or gy < 0 or gx > self.map.shape[0] or gy > self.map.shape[1]:
                continue # agent outside bounds of map, so don't plot it
            self.map[gx, gy] = 255
