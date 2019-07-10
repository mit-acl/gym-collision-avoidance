import gym
import os
import numpy as np

from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.wrappers import FlattenDictWrapper, MultiagentFlattenDictWrapper
import gym_collision_avoidance.envs.test_cases as tc 
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import bench, logger

from gym_collision_avoidance.envs.policies.PPOCADRLPolicy import PPOCADRLPolicy

Config.EVALUATE_MODE = True
# Config.EVALUATE_MODE = False
Config.PLAY_MODE = not Config.EVALUATE_MODE
if Config.EVALUATE_MODE:
    Config.PLOT_EPISODES = True
    # Config.ANIMATE_EPISODES = False
    Config.ANIMATE_EPISODES = True
    Config.DT = 0.1
elif Config.PLAY_MODE:
    Config.PLOT_EPISODES = True
    Config.ANIMATE_EPISODES = True

test_case_fn = tc.get_testcase_old_and_crappy
num_agents = 2
num_test_cases = Config.NUM_TEST_CASES

import tensorflow as tf
tf.Session().__enter__()

def run_episode(env, init_obs):
    obs = init_obs
    score = 0
    done = False
    while not done:
        obs, rew, done, info = env.step([])
        score += rew[0]
    return score, obs

num_envs = 1
ncpu = 1
def make_env():
    env = gym.make("CollisionAvoidance-v0")
    env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    env = MultiagentFlattenDictWrapper(env, dict_keys=Config.STATES_IN_OBS, max_num_agents=Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
    # static_map_path = "~/code/gyms/gym-collision-avoidance/gym_collision_avoidance/envs/world_maps/{}.png"
    # maps = [static_map_path.format(i) for i in ['000']]
    # # maps = [static_map_path.format(i) for i in ['004', '005', '006']]
    # # maps = [static_map_path.format(i) for i in ['000', '003', '004', '005', '006']]
    # env.unwrapped.set_static_map(map_filename=maps)
    return env
env = DummyVecEnv([make_env for _ in range(num_envs)])

unwrapped_envs = [e.unwrapped for e in env.envs]
one_env = unwrapped_envs[0]
total_time_to_goal = 0.0
for ep in range(num_test_cases):
    print('-------')
    print("Episode:", ep)
    agents = test_case_fn(num_agents, ep)
    policies = {}
    for agent in agents:
        if isinstance(agent.policy, PPOCADRLPolicy):
            agent.policy.env = env
            agent.policy.initialize_network()
    one_env.set_agents(agents)
    one_env.test_case_index = ep
    init_obs = env.reset()

    score, init_obs = run_episode(env, init_obs)
    agents = one_env.prev_episode_agents
    time_to_goal = np.array([a.t for a in agents])
    collision = np.array(
        np.any([a.in_collision for a in agents])).tolist()
    all_at_goal = np.array(
        np.all([a.is_at_goal for a in agents])).tolist()
    any_stuck = np.array(
        np.any([not a.in_collision and not a.is_at_goal for a in agents])).tolist()
    total_time_to_goal += np.sum(time_to_goal)

    print("score:", score)
    print("time to goal:", time_to_goal)
    print("at goal: {}".format([a.is_at_goal for a in agents]))
    print("combined time to goal: {0:.1f}".format(np.sum(time_to_goal)))
    print("collision:", collision)
    print("any_stuck:", any_stuck)
one_env.reset()
print('---------------------')
print("Total time_to_goal: {0:.2f}".format(total_time_to_goal))
print('---------------------')
        

print("Experiment over.")
