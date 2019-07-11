import gym
import os
import numpy as np
import pickle

from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.wrappers import FlattenDictWrapper, MultiagentFlattenDictWrapper
import gym_collision_avoidance.envs.test_cases as tc 
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import bench, logger

from gym_collision_avoidance.envs.policies.PPOCADRLPolicy import PPOCADRLPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy

Config.EVALUATE_MODE = True
Config.PLOT_EPISODES = True
Config.ANIMATE_EPISODES = False
Config.DT = 0.1

# record_pickle_files = True
record_pickle_files = False


# test_case_fn = tc.small_test_suite
test_case_fn = tc.full_test_suite
test_case_args = {}
policies = [CADRLPolicy]
# policies = [GA3CCADRLPolicy]
# policies = [PPOCADRLPolicy]
# policies = [PPOCADRLPolicy, RVOPolicy, CADRLPolicy]
num_agents_to_test = [2]
# num_agents_to_test = [2,3,4]
# num_test_cases = 8
num_test_cases = Config.NUM_TEST_CASES

def run_episode(env, one_env):
    score = 0
    done = False
    while not done:
        obs, rew, done, info = env.step([])
        score += rew[0]

    # After end of episode, compute statistics about the agents
    agents = one_env.prev_episode_agents
    time_to_goal = np.array([a.t for a in agents])
    extra_time_to_goal = np.array([a.t - a.straight_line_time_to_reach_goal for a in agents])
    collision = np.array(
        np.any([a.in_collision for a in agents])).tolist()
    all_at_goal = np.array(
        np.all([a.is_at_goal for a in agents])).tolist()
    any_stuck = np.array(
        np.any([not a.in_collision and not a.is_at_goal for a in agents])).tolist()

    return time_to_goal, extra_time_to_goal, collision, all_at_goal, any_stuck

def create_env():
    num_envs = 1
    ncpu = 1
    def make_env():
        env = gym.make("CollisionAvoidance-v0")
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        env = MultiagentFlattenDictWrapper(env, dict_keys=Config.STATES_IN_OBS, max_num_agents=Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
        return env
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    unwrapped_envs = [e.unwrapped for e in env.envs]
    one_env = unwrapped_envs[0]
    return env, one_env

def store_stats(stats, policy, test_case, times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck):
    stats[policy][test_case] = {}
    stats[policy][test_case]['times_to_goal'] = times_to_goal
    stats[policy][test_case]['extra_times_to_goal'] = extra_times_to_goal
    stats[policy][test_case]['mean_extra_time_to_goal'] = np.mean(extra_times_to_goal)
    stats[policy][test_case]['total_time_to_goal'] = np.sum(times_to_goal)
    stats[policy][test_case]['collision'] = collision
    stats[policy][test_case]['all_at_goal'] = all_at_goal
    if not collision: stats[policy]['non_collision_inds'].append(test_case)
    if all_at_goal: stats[policy]['all_at_goal_inds'].append(test_case)
    if not collision and not all_at_goal: stats[policy]['stuck_inds'].append(test_case)
    return stats


import tensorflow as tf
tf.Session().__enter__()
env, one_env = create_env()

for num_agents in num_agents_to_test:
    test_case_args['num_agents'] = num_agents
    stats = {}
    for policy in policies:
        stats[policy] = {}
        stats[policy]['non_collision_inds'] = []
        stats[policy]['all_at_goal_inds'] = []
        stats[policy]['stuck_inds'] = []
    for test_case in range(num_test_cases):
        test_case_args['test_case_index'] = test_case
        for policy in policies:
            print('-------')
            test_case_args['agents_policy'] = policy
            agents = test_case_fn(**test_case_args)
            for agent in agents:
                if isinstance(agent.policy, PPOCADRLPolicy):
                    agent.policy.env = env
                    agent.policy.initialize_network()
            one_env.set_agents(agents)
            one_env.test_case_index = test_case
            init_obs = env.reset()

            times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck = run_episode(env, one_env)

            stats = store_stats(stats, policy, test_case, times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck)
            
            print("Test Case:", test_case)
            if collision:
                print("*******Collision*********")
            if not collision and not all_at_goal:
                print("*******Stuck*********")

            print("Agents Time to goal:", times_to_goal)
            print("Agents Extra Times to goal:", extra_times_to_goal)
            print("Total time to goal (all agents):", np.sum(times_to_goal))
    one_env.reset()
    if record_pickle_files:
        for policy in policies:
            file_dir = os.path.dirname(os.path.realpath(__file__)) + '/../logs/test_case_stats/'
            file_dir += '{num_agents}_agents/'.format(num_agents=test_case_args['num_agents'])
            os.makedirs(file_dir, exist_ok=True)
            fname = file_dir+policy.__name__+'.p'
            pickle.dump(stats[policy], open(fname,'wb'))
            print('dumped {}'.format(fname))
# print('---------------------')
# print("Total time_to_goal: {0:.2f}".format(total_time_to_goal))
# print('---------------------')
        

print("Experiment over.")
