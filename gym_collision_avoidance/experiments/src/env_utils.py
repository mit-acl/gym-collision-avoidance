import gym
gym.logger.set_level(40)
import numpy as np
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.wrappers import FlattenDictWrapper, MultiagentFlattenDictWrapper, MultiagentDummyVecEnv, MultiagentDictToMultiagentArrayWrapper
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def create_env():
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()
    num_envs = 1
    ncpu = 1
    
    def make_env():
        env = gym.make("CollisionAvoidance-v0")

        # The env provides a dict observation by default. Most RL code
        # doesn't handle dict observations, so these wrappers convert to arrays
        if Config.TRAIN_SINGLE_AGENT:
            # only return observations of a single agent
            env = FlattenDictWrapper(env, dict_keys=Config.STATES_IN_OBS)
        else:
            # Convert the dict into an np array, shape=(max_num_agents, num_states_per_agent)
            env = MultiagentDictToMultiagentArrayWrapper(env, dict_keys=Config.STATES_IN_OBS, max_num_agents=Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
            # Convert the dict into a flat np array, shape=(max_num_agents*num_states_per_agent)
            # env = MultiagentFlattenDictWrapper(env, dict_keys=Config.STATES_IN_OBS, max_num_agents=Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
        
        return env
    
    # To be prepared for training on multiple instances of the env at once
    if Config.TRAIN_SINGLE_AGENT:
        env = DummyVecEnv([make_env for _ in range(num_envs)])
    else:
        env = MultiagentDummyVecEnv([make_env for _ in range(num_envs)])
    unwrapped_envs = [e.unwrapped for e in env.envs]
    
    # Set env id for each env
    for i, e in enumerate(unwrapped_envs):
        e.id = i
    
    one_env = unwrapped_envs[0]
    return env, one_env

def run_episode(env, one_env):
    total_reward = 0
    step = 0
    done = False
    while not done:
        obs, rew, done, info = env.step([None])
        total_reward += rew[0]
        step += 1

    # After end of episode, store some statistics about the environment
    # Some stats apply to every gym env...
    generic_episode_stats = {
        'total_reward': total_reward,
        'steps': step,
    }

    agents = one_env.agents
    # agents = one_env.prev_episode_agents
    time_to_goal = np.array([a.t for a in agents])
    extra_time_to_goal = np.array([a.t - a.straight_line_time_to_reach_goal for a in agents])
    collision = np.array(
        np.any([a.in_collision for a in agents])).tolist()
    all_at_goal = np.array(
        np.all([a.is_at_goal for a in agents])).tolist()
    any_stuck = np.array(
        np.any([not a.in_collision and not a.is_at_goal for a in agents])).tolist()
    outcome = "collision" if collision else "all_at_goal" if all_at_goal else "stuck"
    specific_episode_stats = {
        'num_agents': len(agents),
        'time_to_goal': time_to_goal,
        'total_time_to_goal': np.sum(time_to_goal),
        'extra_time_to_goal': extra_time_to_goal,
        'collision': collision,
        'all_at_goal': all_at_goal,
        'any_stuck': any_stuck,
        'outcome': outcome,
        'policies': [agent.policy.str for agent in agents],
    }

    # Merge all stats into a single dict
    episode_stats = {**generic_episode_stats, **specific_episode_stats}

    env.reset()

    return episode_stats, agents

def store_stats(df, hyperparameters, episode_stats):
    # Add a new row to the pandas DataFrame (a table of results, where each row is an episode)
    # that contains the hyperparams and stats from that episode, for logging purposes
    df_columns = {**hyperparameters, **episode_stats}
    df = df.append(df_columns, ignore_index=True)
    return df


policies = {

    'GA3C-CADRL-10-WS-4-1': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-184-72-212-132.compute-1.amazonaws.com/wandb/run-20200403_144424-3eoowzko/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-10-WS-4-2': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-34-228-142-219.compute-1.amazonaws.com/wandb/run-20200403_144424-eozu6syw/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-10-WS-4-3': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-54-146-99-195.compute-1.amazonaws.com/wandb/run-20200403_144424-22s6pbwt/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-10-WS-4-4': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-54-227-24-219.compute-1.amazonaws.com/wandb/run-20200403_144424-2f8r4ydk/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-10-WS-4-5': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-54-242-32-57.compute-1.amazonaws.com/wandb/run-20200403_144424-i41jmnda/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },


    'GA3C-CADRL-4-WS-4-1': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-184-72-212-132.compute-1.amazonaws.com/wandb/run-20200402_210747-dt4uwai3/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-4-WS-4-2': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-34-228-142-219.compute-1.amazonaws.com/wandb/run-20200402_210747-cvcfrsqt/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-4-WS-4-3': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-54-146-99-195.compute-1.amazonaws.com/wandb/run-20200402_210747-1rmgsf1f/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-4-WS-4-4': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-54-227-24-219.compute-1.amazonaws.com/wandb/run-20200402_210747-2unxv49c/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-4-WS-4-5': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_ec2-54-242-32-57.compute-1.amazonaws.com/wandb/run-20200402_210747-2hjygfa8/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },


    'GA3C-CADRL-10-LSTM-1': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-107-21-169-18.compute-1.amazonaws.com/wandb/run-20200403_144352-24y2fdt1/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 19,
            },
        },
    'GA3C-CADRL-10-LSTM-2': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-107-22-158-27.compute-1.amazonaws.com/wandb/run-20200403_144352-degz8bdo/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 19,
            },
        },
    'GA3C-CADRL-10-LSTM-3': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-34-228-80-228.compute-1.amazonaws.com/wandb/run-20200403_144352-38r4hkya/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 19,
            },
        },
    'GA3C-CADRL-10-LSTM-4': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-54-226-118-56.compute-1.amazonaws.com/wandb/run-20200403_144352-2wxsxlws/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 19,
            },
        },
    'GA3C-CADRL-10-LSTM-5': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-54-226-192-14.compute-1.amazonaws.com/wandb/run-20200403_144352-13bui0x5/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 19,
            },
        },

    'GA3C-CADRL-4-LSTM-1': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-107-21-169-18.compute-1.amazonaws.com/wandb/run-20200402_205112-3dz5k5pp/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-4-LSTM-2': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-107-22-158-27.compute-1.amazonaws.com/wandb/run-20200402_205111-1kglu4km/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-4-LSTM-3': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-34-228-80-228.compute-1.amazonaws.com/wandb/run-20200402_205112-16352wzy/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-4-LSTM-4': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-54-226-118-56.compute-1.amazonaws.com/wandb/run-20200402_205112-25eq7fer/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },
    'GA3C-CADRL-4-LSTM-5': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/rnn_order_ec2-54-226-192-14.compute-1.amazonaws.com/wandb/run-20200402_205111-2pka2zpr/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 3,
            },
        },



    'GA3C-CADRL-4-WS-6-1': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_6_ec2-35-174-113-253.compute-1.amazonaws.com/wandb/run-20200412_151445-icfzrvij/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 5,
            },
        },
    'GA3C-CADRL-4-WS-6-2': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_6_ec2-3-88-36-89.compute-1.amazonaws.com/wandb/run-20200412_151445-2iv4i4nj/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 5,
            },
        },
    'GA3C-CADRL-4-WS-6-3': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_6_ec2-54-161-193-150.compute-1.amazonaws.com/wandb/run-20200412_151445-91kuvs98/checkpoints/',
        'checkpt_name': 'network_01490001',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 5,
            },
        },
    'GA3C-CADRL-4-WS-6-4': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_6_ec2-54-243-3-91.compute-1.amazonaws.com/wandb/run-20200412_051045-crf4k6on/checkpoints/',
        'checkpt_name': 'network_01490001',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 5,
            },
        },
    'GA3C-CADRL-10-WS-6-1': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_6_ec2-35-174-113-253.compute-1.amazonaws.com/wandb/run-20200413_023855-3mbmr1nc/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 5,
            },
        },
    'GA3C-CADRL-10-WS-6-2': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_6_ec2-3-88-36-89.compute-1.amazonaws.com/wandb/run-20200413_023855-3pxw2ixl/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 5,
            },
        },
    'GA3C-CADRL-10-WS-6-3': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_6_ec2-54-161-193-150.compute-1.amazonaws.com/wandb/run-20200413_023939-sgw8r5gx/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 5,
            },
        },
    'GA3C-CADRL-10-WS-6-4': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_6_ec2-54-243-3-91.compute-1.amazonaws.com/wandb/run-20200412_163307-1yz34rae/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 5,
            },
        },

    'GA3C-CADRL-4-WS-8-1': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_8_ec2-18-212-168-204.compute-1.amazonaws.com/wandb/run-20200412_151345-1luyhexf/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 7,
            },
        },
    'GA3C-CADRL-4-WS-8-2': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_8_ec2-34-203-243-164.compute-1.amazonaws.com/wandb/run-20200412_151345-2j2jvgjv/checkpoints/',
        'checkpt_name': 'network_01490001',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 7,
            },
        },
    'GA3C-CADRL-4-WS-8-3': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_8_ec2-3-92-177-119.compute-1.amazonaws.com/wandb/run-20200412_051045-3oza4dxf/checkpoints/',
        'checkpt_name': 'network_01490002',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 7,
            },
        },
    'GA3C-CADRL-4-WS-8-4': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_8_ec2-54-211-142-133.compute-1.amazonaws.com/wandb/run-20200412_151345-3ql9fhpf/checkpoints/',
        'checkpt_name': 'network_01490000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 7,
            },
        },
    'GA3C-CADRL-10-WS-8-1': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_8_ec2-18-212-168-204.compute-1.amazonaws.com/wandb/run-20200413_024321-3m0g6fei/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 7,
            },
        },
    'GA3C-CADRL-10-WS-8-2': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_8_ec2-34-203-243-164.compute-1.amazonaws.com/wandb/run-20200413_024321-1qog6ten/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 7,
            },
        },
    'GA3C-CADRL-10-WS-8-3': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_8_ec2-3-92-177-119.compute-1.amazonaws.com/wandb/run-20200412_163307-1yslbfru/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 7,
            },
        },
    'GA3C-CADRL-10-WS-8-4': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': '/home/mfe/ijrr_cadrl_results/multiple_seeds/ws_order_8_ec2-54-211-142-133.compute-1.amazonaws.com/wandb/run-20200413_024321-1i8errn0/checkpoints/',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 7,
            },
        },

    'GA3C-CADRL-10': {
        'policy': 'GA3C_CADRL',
        'checkpt_dir': 'IROS18',
        'checkpt_name': 'network_01900000',
        'sensors': ['other_agents_states'],
        'sensor_args': {
            'agent_sorting_method': 'closest_last',
            'max_num_other_agents_observed': 19,
            },
        },

    # 'GA3C-CADRL-4-LSTM': {
    #     'policy': 'GA3C_CADRL',
    #     'checkpt_dir': "run-20190727_015942-jzuhlntn",
    #     'checkpt_name': 'network_01490000',
    #     'sensors': ['other_agents_states'],
    #     },
    'CADRL': {
        'policy': 'CADRL',
        'sensors': ['other_agents_states'],
        },
    'RVO': {
        'policy': 'RVO',
        'sensors': ['other_agents_states'],
        },
    'DRL-Long': {
        'policy': 'drllong',
        'checkpt_name': 'stage2.pth',
        'sensors': ['other_agents_states', 'laserscan']
        },
    }