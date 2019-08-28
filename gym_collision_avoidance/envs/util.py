import numpy as np

##################
# Utils
################

# speed filter
# input: past velocity in (x,y) at time_past 
# output: filtered velocity in (speed, theta)
def filter_vel(dt_vec, agent_past_vel_xy):
    average_x = np.sum(dt_vec * agent_past_vel_xy[:,0]) / np.sum(dt_vec)
    average_y = np.sum(dt_vec * agent_past_vel_xy[:,1]) / np.sum(dt_vec)
    speeds = np.linalg.norm(agent_past_vel_xy, axis=1)
    speed = np.linalg.norm(np.array([average_x, average_y]))
    angle = np.arctan2(average_y, average_x)

    return np.array([speed, angle])

# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff

# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

def find_nearest(array,value):
    # array is a 1D np array
    # value is an scalar or 1D np array
    tiled_value = np.tile(np.expand_dims(value,axis=0).transpose(), (1,np.shape(array)[0]))
    idx = (np.abs(array-tiled_value)).argmin(axis=1)
    return array[idx], idx

def rad2deg(rad):
    return rad*180/np.pi

def rgba2rgb(rgba):
    # rgba is a list of 4 color elements btwn [0.0, 1.0]
    # or a 2d np array (num_colors, 4)
    # returns a list of rgb values between [0.0, 1.0] accounting for alpha and background color [1, 1, 1] == WHITE
    if isinstance(rgba, list):
        alpha = rgba[3]
        r = max(min((1 - alpha) * 1.0 + alpha * rgba[0],1.0),0.0)
        g = max(min((1 - alpha) * 1.0 + alpha * rgba[1],1.0),0.0)
        b = max(min((1 - alpha) * 1.0 + alpha * rgba[2],1.0),0.0)
        return [r,g,b]
    elif rgba.ndim == 2:
        alphas = rgba[:,3]
        r = np.clip((1 - alphas) * 1.0 + alphas * rgba[:,0], 0, 1)
        g = np.clip((1 - alphas) * 1.0 + alphas * rgba[:,1], 0, 1)
        b = np.clip((1 - alphas) * 1.0 + alphas * rgba[:,2], 0, 1)
        return np.vstack([r,g,b]).T

def yaw_to_quaternion(yaw):
    pitch = 0; roll = 0
    cy = np.cos(yaw * 0.5);
    sy = np.sin(yaw * 0.5);
    cp = np.cos(pitch * 0.5);
    sp = np.sin(pitch * 0.5);
    cr = np.cos(roll * 0.5);
    sr = np.sin(roll * 0.5);

    qw = cy * cp * cr + sy * sp * sr;
    qx = cy * cp * sr - sy * sp * cr;
    qy = sy * cp * sr + cy * sp * cr;
    qz = sy * cp * cr - cy * sp * sr;
    return qx, qy, qz, qw