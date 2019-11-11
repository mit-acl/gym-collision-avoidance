'''

There are a lot of ways to define a test case.
At the end of the day, you just need to provide a list of Agent objects to the environment.

- For simple testing of a particular configuration, consider defining a function like `get_testcase_two_agents`.
- If you want some final position configuration, consider something like `formation`.
- For a large, static test suite, consider creating a pickle file of [start_x, start_y, goal_x, goal_y, radius, pref_speed] tuples and use our code to convert that into a list of Agents, as in `preset_testCases`.

After defining a test case function that returns a list of Agents, you can select that test case fn in the evaluation code, e.g. 

'''

import numpy as np
from gym_collision_avoidance.envs.agent import Agent

from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
from gym_collision_avoidance.envs.policies.DRLLongPolicy import DRLLongPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxTurnRate import UnicycleDynamicsMaxTurnRate
from gym_collision_avoidance.envs.dynamics.ExternalDynamics import ExternalDynamics
from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
from gym_collision_avoidance.envs.config import Config

import os
import pickle

from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import gen_rand_testcases as tc

test_case_filename = "{dir}/test_cases/{pref_speed_string}{num_agents}_agents_{num_test_cases}_cases.p"

def get_testcase_two_agents():
    goal_x = 3
    goal_y = 3
    agents = [
        Agent(-goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.5, LearningPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
        Agent(goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1)
        ]
    return agents

def get_testcase_two_agents_laserscanners():
    goal_x = 3
    goal_y = 3
    agents = [
        Agent(-goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.5, PPOPolicy, UnicycleDynamics, [LaserScanSensor], 0),
        Agent(goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, 0.5, PPOPolicy, UnicycleDynamics, [LaserScanSensor], 1)
        ]
    return agents

def get_testcase_random(num_agents=None, side_length=None, speed_bnds=None, radius_bnds=None, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[]):
    if num_agents is None:
        num_agents = np.random.randint(2, Config.MAX_NUM_AGENTS_IN_ENVIRONMENT+1)

    if side_length is None:
        side_length = 4

    if speed_bnds is None:
        speed_bnds = [0.5, 1.5]

    if radius_bnds is None:
        radius_bnds = [0.2, 0.8]

    cadrl_test_case = tc.generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds)

    agents = cadrl_test_case_to_agents(cadrl_test_case,
        agents_policy=agents_policy,
        agents_dynamics=agents_dynamics,
        agents_sensors=agents_sensors)
    return agents

def get_testcase_2agents_swap(test_case_index, num_test_cases=10, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[]):
    pref_speed = 1.0
    radius = 0.5
    goal_x = 3
    goal_y = 0

    total_delta = 3.
    offset = (test_case_index - num_test_cases/2.) / (num_test_cases/total_delta)

    agents = [
        Agent(-goal_x, -goal_y, goal_x, goal_y+offset, radius, pref_speed, None, agents_policy, agents_dynamics, agents_sensors, 0),
        Agent(goal_x, goal_y, -goal_x, -goal_y, radius, pref_speed, None, agents_policy, agents_dynamics, agents_sensors, 1)
        ]
    return agents

def get_testcase_easy():

    num_agents = 2
    side_length = 2
    speed_bnds = [0.5, 1.5]
    radius_bnds = [0.2, 0.8]

    test_case = tc.generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds)

    agents = cadrl_test_case_to_agents(test_case)
    return agents

def get_testcase_fixed_initial_conditions(agents):
    new_agents = []
    for agent in agents:
        goal_x, goal_y = get_new_goal(agent.pos_global_frame)
        new_agent = Agent(agent.pos_global_frame[0], agent.pos_global_frame[1], goal_x, goal_y, agent.radius, agent.pref_speed, agent.heading_global_frame, agent.policy.__class__, agent.dynamics_model.__class__, [], agent.id)
        new_agents.append(new_agent)
    return new_agents

def get_testcase_fixed_initial_conditions_for_non_ppo(agents):
    new_agents = []
    for agent in agents:
        if agent.policy.str == "PPO":
            start_x, start_y = get_new_start_pos()
        else:
            start_x, start_y = agent.pos_global_frame
        goal_x, goal_y = get_new_goal(agent.pos_global_frame)
        new_agent = Agent(start_x, start_y, goal_x, goal_y, agent.radius, agent.pref_speed, agent.heading_global_frame, agent.policy.__class__, agent.dynamics_model.__class__, [], agent.id)
        new_agents.append(new_agent)
    return new_agents

def get_new_goal(pos):
    bounds = np.array([[-5, 5], [-5, 5]])
    dist_from_pos_threshold = 4.
    far_from_pos = False
    while not far_from_pos:
        gx, gy = np.random.uniform(bounds[:,0], bounds[:,1])
        far_from_pos = np.linalg.norm(pos - np.array([gx, gy])) >= dist_from_pos_threshold
    return gx, gy

def small_test_suite(num_agents, test_case_index, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[], vpref_constraint=False, radius_bnds=None):
    cadrl_test_case = preset_testCases(num_agents)[test_case_index]
    agents = cadrl_test_case_to_agents(cadrl_test_case, agents_policy=agents_policy, agents_dynamics=agents_dynamics, agents_sensors=agents_sensors)
    return agents

def full_test_suite(num_agents, test_case_index, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[], vpref_constraint=False, radius_bounds=None):
    cadrl_test_case = preset_testCases(num_agents, full_test_suite=True, vpref_constraint=vpref_constraint, radius_bounds=radius_bounds)[test_case_index]
    agents = cadrl_test_case_to_agents(cadrl_test_case, agents_policy=agents_policy, agents_dynamics=agents_dynamics, agents_sensors=agents_sensors)
    return agents

def formation(agents, letter, num_agents=6, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[OtherAgentsStatesSensor]):
    formations = {
        'A': 2*np.array([
              [-1.5, 0.0], # A
              [1.5, 0.0], 
              [0.75, 1.5],
              [-0.75, 1.5],
              [0.0, 1.5], 
              [0.0, 3.0]
            ]),
        'C': 2*np.array([
              [0.0, 0.0], # C
              [-0.5, 1.0], 
              [-0.5, 2.0],
              [0.0, 3.0],
              [1.5, 0.0], 
              [1.5, 3.0]
              ]),
        'L': 2*np.array([
            [0.0, 0.0], # L
            [0.0, 1.0], 
            [0.0, 2.0],
            [0.0, 3.0],
            [0.75, 0.0], 
            [1.5, 0.0]
            ]),
        'D': 2*np.array([
            [0.0, 0.0],
            [0.0, 1.5], 
            [0.0, 3.0],
            [1.5, 1.5], 
            [1.2, 2.5],
            [1.2, 0.5],
            ]),
        'R': 2*np.array([
            [0.0, 0.0],
            [0.0, 1.5], 
            [0.0, 3.0],
            [1.3, 2.8], 
            [1.2, 1.7],
            [1.7, 0.0],
            ]),
    }

    agent_inds = np.arange(num_agents)
    np.random.shuffle(agent_inds)

    new_agents = []
    for agent in agents:
        start_x, start_y = agent.pos_global_frame
        goal_x, goal_y = formations[letter][agent_inds[agent.id]]
        new_agent = Agent(start_x, start_y, goal_x, goal_y, agent.radius, agent.pref_speed, agent.heading_global_frame, agents_policy, agents_dynamics, agents_sensors, agent.id)
        new_agents.append(new_agent)
    return new_agents

def cadrl_test_case_to_agents(test_case, agents_policy=LearningPolicy, agents_dynamics=UnicycleDynamics, agents_sensors=[]):
    ###############################
    # This function accepts a test_case in legacy cadrl format and converts it
    # into our new list of Agent objects. The legacy cadrl format is a list of
    # [start_x, start_y, goal_x, goal_y, pref_speed, radius] for each agent.
    ###############################

    agents = []
    policies = [NonCooperativePolicy, LearningPolicy, StaticPolicy]
    if Config.EVALUATE_MODE or Config.PLAY_MODE:
        agent_policy_list = [agents_policy for _ in range(np.shape(test_case)[0])]
    else:
        # Random mix of agents following various policies
        # agent_policy_list = np.random.choice(policies,
        #                                      np.shape(test_case)[0],
        #                                      p=[0.05, 0.9, 0.05])
        agent_policy_list = np.random.choice(policies,
                                             np.shape(test_case)[0],
                                             p=[0.0, 1.0, 0.0])

        # Make sure at least one agent is following PPO
        #  (otherwise waste of time...)
        if LearningPolicy not in agent_policy_list:
            random_agent_id = np.random.randint(len(agent_policy_list))
            agent_policy_list[random_agent_id] = LearningPolicy

    agent_dynamics_list = [agents_dynamics for _ in range(np.shape(test_case)[0])]
    agent_sensors_list = [agents_sensors for _ in range(np.shape(test_case)[0])]

    for i, agent in enumerate(test_case):
        px = agent[0]
        py = agent[1]
        gx = agent[2]
        gy = agent[3]
        pref_speed = agent[4]
        radius = agent[5]
        if Config.EVALUATE_MODE:
            # initial heading is pointed toward the goal
            vec_to_goal = np.array([gx, gy]) - np.array([px, py])
            heading = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        else:
            heading = np.random.uniform(-np.pi, np.pi)

        agents.append(Agent(px, py, gx, gy, radius, pref_speed, heading, agent_policy_list[i], agent_dynamics_list[i], agent_sensors_list[i], i))
    return agents

def preset_testCases(num_agents, full_test_suite=False, vpref_constraint=False, radius_bounds=None):
    if full_test_suite:
        num_test_cases = 500

        if vpref_constraint:
            pref_speed_string = 'vpref1.0_r{}-{}/'.format(radius_bounds[0], radius_bounds[1])
        else:
            pref_speed_string = ''

        filename = test_case_filename.format(
                num_agents=num_agents, num_test_cases=num_test_cases, pref_speed_string=pref_speed_string,
                dir=os.path.dirname(os.path.realpath(__file__)))
        test_cases = pickle.load(open(filename, "rb"), encoding='latin1')

    else:
        if num_agents == 1:
            test_cases = []
            # fixed speed and radius
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [3.0/1.4, -3.0/1.4, -3.0/1.4, 3.0/1.4, 1.0, 0.3]
                ]))

        elif num_agents == 2:
            test_cases = []
            # fixed speed and radius
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0/1.4, -3.0/1.4, -3.0/1.4, 3.0/1.4, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-2.0, -1.5, 2.0, 1.5, 1.0, 0.5],
                [-2.0, 1.5, 2.0, -1.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [0.0, -3.0, 0.0, 3.0, 1.0, 0.5]
                ]))
            # variable speed and radius
            test_cases.append(np.array([
                [-2.5, 0.0, 2.5, 0.0, 1.0, 0.3],
                [2.5, 0.0, -2.5, 0.0, 0.8, 0.4]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 0.6, 0.5],
                [3.0/1.4, -3.0/1.4, -3.0/1.4, 3.0/1.4, 1.0, 0.4]
                ]))
            test_cases.append(np.array([
                [-2.0, 0.0, 2.0, 0.0, 0.9, 0.35],
                [2.0, 0.0, -2.0, 0.0, 0.85, 0.45]
                ]))
            test_cases.append(np.array([
                [-4.0, 0.0, 4.0, 0.0, 1.0, 0.4],
                [-2.0, 0.0, 2.0, 0.0, 0.5, 0.4]
                ]))

        elif num_agents == 3 or num_agents == 4:
            test_cases = []
            # hardcoded to be 3 agents for now
            d = 3.0
            l1 = d*np.cos(np.pi/6)
            l2 = d*np.sin(np.pi/6)
            test_cases.append(np.array([
                [0.0, d, 0.0, -d, 1.0, 0.5],
                [l1, -l2, -l1, l2, 1.0, 0.5],
                [-l1, -l2, l1, l2, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, 1.5, 1.0, 0.5]
                ]))
            # hardcoded to be 4 agents for now
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.3],
                [3.0, -1.5, -3.0, -1.5, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.3],
                [3.0, -3.0, -3.0, -3.0, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [0.0, -3.0, 0.0, 3.0, 1.0, 0.5],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.5],
                [0.0, 3.0, 0.0, -3.0, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-2.0, -1.5, 2.0, 1.5, 1.0, 0.5],
                [-2.0, 1.5, 2.0, -1.5, 1.0, 0.5],
                [-2.0, -4.0, 2.0, -4.0, 0.9, 0.35],
                [2.0, -4.0, -2.0, -4.0, 0.85, 0.45]
                ]))
            test_cases.append(np.array([
                [-4.0, 0.0, 4.0, 0.0, 1.0, 0.4],
                [-2.0, 0.0, 2.0, 0.0, 0.5, 0.4],
                [-4.0, -4.0, 4.0, -4.0, 1.0, 0.4],
                [-2.0, -4.0, 2.0, -4.0, 0.5, 0.4]
                ]))

        elif num_agents == 5:
            test_cases = []

            radius = 4
            tc = gen_circle_test_case(num_agents, radius)
            test_cases.append(tc)

            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, 3.0, 3.0, 3.0, 1.0, 0.5],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.5]
                ]))

        elif num_agents == 6:
            test_cases = []

            radius = 5
            tc = gen_circle_test_case(num_agents, radius)
            test_cases.append(tc)

            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, 3.0, 3.0, 3.0, 1.0, 0.5],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.5],
                [-3.0, -4.5, 3.0, -4.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, 0.7, 3.0, 0.7, 1.0, 0.3],
                [3.0, 0.7, -3.0, 0.7, 1.0, 0.3],
                [-3.0, -0.7, 3.0, -0.7, 1.0, 0.3],
                [3.0, -0.7, -3.0, -0.7, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, 1.0, 3.0, 1.0, 1.0, 0.3],
                [3.0, 1.0, -3.0, 1.0, 1.0, 0.3],
                [-3.0, -1.0, 3.0, -1.0, 1.0, 0.3],
                [3.0, -1.0, -3.0, -1.0, 1.0, 0.3]
                ]))

        elif num_agents == 10:
            test_cases = []

            radius = 5
            tc = gen_circle_test_case(num_agents, radius)
            test_cases.append(tc)

        elif num_agents == 20:
            test_cases = []

            radius = 10
            tc = gen_circle_test_case(num_agents, radius)
            test_cases.append(tc)

        else:
            print("[preset_testCases in Collision_Avoidance.py]\
                    invalid num_agents")
            assert(0)
    return test_cases


def gen_circle_test_case(num_agents, radius):
    tc = np.zeros((num_agents, 6))
    for i in range(num_agents):
        tc[i, 4] = 1.0
        tc[i, 5] = 0.5
        theta_start = (2*np.pi/num_agents)*i
        theta_end = theta_start + np.pi
        tc[i, 0] = radius*np.cos(theta_start)
        tc[i, 1] = radius*np.sin(theta_start)
        tc[i, 2] = radius*np.cos(theta_end)
        tc[i, 3] = radius*np.sin(theta_end)
    return tc

def get_testcase_hololens_and_ga3c_cadrl():
    goal_x1 = 3
    goal_y1 = 3
    goal_x2 = 2
    goal_y2 = 5
    agents = [
              Agent(-goal_x1, goal_y1, goal_x1, -goal_y1, 0.5, 1.0, 0.5, ExternalPolicy, ExternalDynamics, [], 0), # hololens
              Agent(goal_x1, goal_y1, goal_x1, -goal_y1, 0.5, 1.0, 0.5, ExternalPolicy, ExternalDynamics, [], 1), # real robot
              Agent(-goal_x1+np.random.uniform(-3,3), -goal_y1+np.random.uniform(-1,1), goal_x1, goal_y1, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 2)
              ]
              # Agent(goal_x1, goal_y1, -goal_x1, -goal_y1, 0.5, 2.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 1),
              # Agent(-goal_x2, -goal_y2, goal_x2, goal_y2, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 2),
              # Agent(goal_x2, goal_y2, -goal_x2, -goal_y2, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 3),
              # Agent(-goal_x2, goal_y2, goal_x2, -goal_y2, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamicsMaxTurnRate, [], 4),
              # Agent(-goal_x1, goal_y1, goal_x1, -goal_y1, 0.5, 1.0, 0.5, ExternalPolicy, ExternalDynamics, [], 5)]
    return agents

def get_testcase_hololens_and_cadrl():
    goal_x = 3
    goal_y = 3
    agents = [Agent(-goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.5, GA3CCADRLPolicy, UnicycleDynamics, [OccupancyGridSensor, LaserScanSensor], 0),
              Agent(goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, 0.5, CADRLPolicy, UnicycleDynamics, [], 1),
              Agent(-goal_x, goal_y, goal_x, -goal_y, 0.5, 1.0, 0.5, ExternalPolicy, ExternalDynamics, [], 2)]
    return agents

if __name__ == '__main__':
    np.random.seed(0)
    # speed_bnds = [0.5, 1.5]
    speed_bnds = [1.0, 1.0]
    # radius_bnds = [0.2, 0.8]
    radius_bnds = [0.1, 0.1]
    num_agents = 4
    side_length = 4

    num_test_cases = 500
    test_cases = []

    for i in range(num_test_cases):
        test_case = tc.generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds)
        test_cases.append(test_case)

    if speed_bnds == [1., 1.]:
        pref_speed_string = 'vpref1.0_r{}-{}/'.format(radius_bnds[0], radius_bnds[1])
    else:
        pref_speed_string = ''

    filename = test_case_filename.format(
                num_agents=num_agents, num_test_cases=num_test_cases, pref_speed_string=pref_speed_string,
                dir=os.path.dirname(os.path.realpath(__file__)))

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    pickle.dump(test_cases, open(filename, "wb"))


