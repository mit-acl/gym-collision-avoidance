import numpy as np
from gym_collision_avoidance.envs.agent import Agent

from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.PPOPolicy import PPOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxTurnRate import UnicycleDynamicsMaxTurnRate
from gym_collision_avoidance.envs.dynamics.ExternalDynamics import ExternalDynamics
from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.config import Config

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

def get_testcase_old_and_crappy(num_agents, index):
    cadrl_test_case = preset_testCases(num_agents)[index]
    agents = cadrl_test_case_to_agents(cadrl_test_case)
    return agents

def cadrl_test_case_to_agents(test_case, alg='PPO'):
    ###############################
    # This function accepts a test_case in legacy cadrl format and converts it
    # into our new list of Agent objects. The legacy cadrl format is a list of
    # [start_x, start_y, goal_x, goal_y, pref_speed, radius] for each agent.
    ###############################

    agents = []
    # policies = [NonCooperativePolicy, StaticPolicy]
    # dynamics_models = [UnicycleDynamics, ExternalDynamics]
    if Config.EVALUATE_MODE:
        agent_policy_list = [ExternalPolicy for _ in range(np.shape(test_case)[0])]
        agent_dynamics_list = [ExternalDynamics for _ in range(np.shape(test_case)[0])]
        # agent_dynamics_list = [UnicycleDynamics for _ in range(np.shape(test_case)[0])]
        # agent_policy_list = [CADRLPolicy for _ in range(np.shape(test_case)[0])]
        # agent_policy_list = [NonCooperativePolicy for _ in range(np.shape(test_case)[0])]
        # agent_policy_list = [StaticPolicy for _ in range(np.shape(test_case)[0])]
    else:
        # Random mix of agents following various policies
        agent_policy_list = np.random.choice(policies,
                                             np.shape(test_case)[0],
                                             p=[0.5, 0.5])
        # if 0 not in agent_policy_list:
        #     # Make sure at least one agent is following PPO
        #     #  (otherwise waste of time...)
        #     random_agent_id = np.random.randint(len(agent_policy_list))
        #     agent_policy_list[random_agent_id] = 0
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

        agents.append(Agent(px, py, gx, gy, radius, pref_speed, heading, agent_policy_list[i], agent_dynamics_list[i], i))
    return agents


def preset_testCases(test_case_num_agents, full_test_suite=False):
    if full_test_suite:
        num_test_cases = 500
        test_cases = pickle.load(open(
            "/home/mfe/ford_ws/src/2017-avrl/src/environment/\
            Collision-Avoidance/test_cases/%s_agents_%i_cases_short.p"
            % (test_case_num_agents, num_test_cases), "rb"))
    else:

        if test_case_num_agents == 2:
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

        elif test_case_num_agents == 3 or test_case_num_agents == 4:
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

        elif test_case_num_agents == 5:
            test_cases = []

            radius = 4
            tc = gen_circle_test_case(test_case_num_agents, radius)
            test_cases.append(tc)

            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, 3.0, 3.0, 3.0, 1.0, 0.5],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.5]
                ]))

        elif test_case_num_agents == 6:
            test_cases = []

            radius = 5
            tc = gen_circle_test_case(test_case_num_agents, radius)
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

        elif test_case_num_agents == 10:
            test_cases = []

            radius = 7
            tc = gen_circle_test_case(test_case_num_agents, radius)
            test_cases.append(tc)

        elif test_case_num_agents == 20:
            test_cases = []

            radius = 10
            tc = gen_circle_test_case(test_case_num_agents, radius)
            test_cases.append(tc)

        else:
            print("[preset_testCases in Collision_Avoidance.py]\
                    invalid test_case_num_agents")
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


'''
###############################
        # This function initializes the self.agents list.
        #
        # Outputs
        # - self.agents: list of Agent objects that have been initialized
        # - self.which_agents_running_ppo: list of T/F values for each agent in self.agents
        ###############################

        # Agents
        # easy_test_cases = True
        easy_test_cases = False
        random_test_cases = True
        if easy_test_cases:
            goal_x = 3
            goal_y = 3
            self.agents = np.array([
                Agent(goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.5, 0),
                Agent(-goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, 0.5, 1)])
        else:
            if self.evaluate:
                print("self.test_case_index:", self.test_case_index)
                self.test_case_index = 1
                # self.test_case_index += 1
                self.test_case_num_agents = 2
                # self.full_test_suite = True
                self.full_test_suite = False
                test_cases = \
                    preset_testCases(self.test_case_num_agents,
                                     full_test_suite=self.full_test_suite)
                test_case = test_cases[self.test_case_index]

            elif random_test_cases:
                num_agents = 2
                # num_agents = np.random.randint(2, Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
                side_length = 4
                #  side_length = np.random.uniform(4, 8)
                speed_bnds = [0.5, 1.5]
                radius_bnds = [0.2, 0.8]

                test_case = \
                    tc.generate_rand_test_case_multi(num_agents,
                                                     side_length,
                                                     speed_bnds,
                                                     radius_bnds,
                                                     is_end_near_bnd=False,
                                                     is_static=False)

            self.agents = self._cadrl_test_case_to_agents(test_case, alg=alg)
        self.which_agents_running_ppo = \
            [agent.id for agent in self.agents if isinstance(agent.policy, type(PPOPolicy))]
        self.num_agents_running_ppo = len(self.which_agents_running_ppo)

'''