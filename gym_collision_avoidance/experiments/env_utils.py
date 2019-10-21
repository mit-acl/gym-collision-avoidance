import gym
import numpy as np
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.wrappers import FlattenDictWrapper, MultiagentFlattenDictWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import bench, logger

def create_env():
    import tensorflow as tf
    tf.Session().__enter__()
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

    return time_to_goal, extra_time_to_goal, collision, all_at_goal, any_stuck, agents
