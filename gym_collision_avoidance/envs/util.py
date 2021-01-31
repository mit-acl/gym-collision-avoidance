import numpy as np
import math
import sys
import os

##################
# Utils
################

def makedirs(directory, exist_ok=True):
    if sys.version[0] == '3':
        os.makedirs(directory, exist_ok=exist_ok)
    elif sys.version[0] == '2':
        if not os.path.exists(directory):
            os.makedirs(directory)

def l2norm(x, y):
    return math.sqrt(l2normsq(x,y))

def l2normsq(x, y):
    return (x[0]-y[0])**2 + (x[1]-y[1])**2

def compute_time_to_impact(host_pos, other_pos, host_vel, other_vel, combined_radius):
    # http://www.ambrsoft.com/TrigoCalc/Circles2/CirclePoint/CirclePointDistance.htm
    v_rel = host_vel - other_vel
    coll_cone_vec1, coll_cone_vec2 = tangent_vecs_from_external_pt(host_pos[0],
                                                                        host_pos[1],
                                                                        other_pos[0],
                                                                        other_pos[1],
                                                                        combined_radius)

    if coll_cone_vec1 is None:
        # collision already occurred ==> collision cone isn't meaningful anymore
        return 0.0
    else: 
        # check if v_rel btwn coll_cone_vecs
        # (B btwn A, C): https://stackoverflow.com/questions/13640931/how-to-determine-if-a-vector-is-between-two-other-vectors)

        if (np.cross(coll_cone_vec1, v_rel) * np.cross(coll_cone_vec1, coll_cone_vec2) >= 0 and 
            np.cross(coll_cone_vec2, v_rel) * np.cross(coll_cone_vec2, coll_cone_vec1) >= 0):
            # quadratic eqn for soln to line from host agent pos along v_rel vector to collision circle
            # circle: (x-a)**2 + (y-b)**2 = r**2
            # line: y = v1/v0 *(x-px) + py
            # solve for x: (x-a)**2 + ((v1/v0)*(x-px)+py-a)**2 = r**2
            v0, v1 = v_rel
            if abs(v0) < 1e-5 and abs(v1) < 1e-5:
                # agents aren't moving toward each other ==> inf TTC
                return np.inf

            px, py = host_pos
            a, b = other_pos
            r = combined_radius
            if abs(v0) < 1e-5: # vertical v_rel (solve for y, x known)
                print("[warning] v0=0, and not yet handled")
                x1 = x2 = px
                A = 1
                B = -2*b
                C = b**2+(px-a)**2-r**2
                y1 = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
                y2 = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
            else: # non-vertical v_rel (solve for x)
                A = 1+(v1/v0)**2
                B = -2*a + 2*(v1/v0)*(py-b-(v1/v0)*px)
                C = a**2 - r**2 + ((v1/v0)*px - (py-b))**2

                det = B**2 - 4*A*C
                if det == 0:
                    print("[warning] det == 0, so only one tangent pt")
                elif det < 0:
                    print("[warning] det < 0, so no tangent pts...")

                x1 = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
                x2 = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
                y1 = (v1/v0)*(x1-px) + py
                y2 = (v1/v0)*(x2-px) + py

            d1 = np.linalg.norm([x1-px, y1-py])
            d2 = np.linalg.norm([x2-px, y2-py])
            d = min(d1, d2)
            spd = np.linalg.norm(v_rel)
            return d / spd 
        else:
            return np.inf

def tangent_vecs_from_external_pt(xp, yp, a, b, r):
    # http://www.ambrsoft.com/TrigoCalc/Circles2/CirclePoint/CirclePointDistance.htm
    # (xp, yp) is coords of pt outside of circle
    # (x-a)**2 + (y-b)**2 = r**2 is defn of circle

    sq_dist_to_perimeter = (xp-a)**2 + (yp-b)**2 - r**2
    if sq_dist_to_perimeter < 0:
        # print("sq_dist_to_perimeter < 0 ==> agent center is already within coll zone??")
        return None, None

    sqrt_term = np.sqrt((xp-a)**2 + (yp-b)**2 - r**2)
    xnum1 = r**2 * (xp-a)
    xnum2 = r*(yp-b)*sqrt_term
    
    ynum1 = r**2 * (yp-b)
    ynum2 = r*(xp-a)*sqrt_term

    den = (xp-a)**2 + (yp-b)**2

    # pt1, pt2 are the tangent pts on the circle perimeter
    pt1 = np.array([(xnum1 + xnum2)/den + a, (ynum1 - ynum2)/den + b])
    pt2 = np.array([(xnum1 - xnum2)/den + a, (ynum1 + ynum2)/den + b])

    # vec1, vec2 are the vecs from (xp,yp) to the tangent pts on the circle perimeter
    vec1 = pt1 - np.array([xp, yp])
    vec2 = pt2 - np.array([xp, yp])
    
    return vec1, vec2

def vec2_l2_norm(vec):
    # return np.linalg.norm(vec)
    return math.sqrt(vec2_l2_norm_squared(vec))

def vec2_l2_norm_squared(vec):
    return vec[0]**2 + vec[1]**2

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