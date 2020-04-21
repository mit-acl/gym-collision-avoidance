import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import time

os.environ['GYM_CONFIG_CLASS'] = 'FullTestSuite'
from gym_collision_avoidance.envs import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env, store_stats, policies

# from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
# from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor

# def main():
#     np.random.seed(0)
#     env, one_env = create_env()
#     for num_agents in num_agents_to_test:
#         for test_case in range(num_test_cases):

def reset_env(env, one_env, test_case_fn, test_case_args, test_case, num_agents, policies, policy, prev_agents):
    # if start_from_last_configuration:
    #     if test_case == 0:
    #         agents = tc.small_test_suite(num_agents=num_agents, test_case_index=0)
    #     test_case_args['agents'] = agents
    #     test_case_args['letter'] = letters[test_case % len(letters)]
    # else:
    test_case_args['num_agents'] = num_agents
    test_case_args['test_case_index'] = test_case
    one_env.plot_policy_name = policy
    test_case_args['policies'] = policies[policy]['policy']
    if 'sensors' in policies[policy]:
        test_case_args['agents_sensors'] = policies[policy]['sensors']
    else:
        test_case_args['agents_sensors'] = []
    test_case_args['prev_agents'] = prev_agents
    agents = test_case_fn(**test_case_args)
    if prev_agents is None:
        for agent in agents:
            if 'checkpt_name' in policies[policy]:
                agent.policy.env = env
                agent.policy.initialize_network(**policies[policy])
            if 'sensor_args' in policies[policy]:
                for sensor in agent.sensors:
                    sensor.set_args(policies[policy]['sensor_args'])
    one_env.set_agents(agents)
    init_obs = env.reset()
    one_env.test_case_index = test_case
    return init_obs

# def reset_env2(env, one_env, test_case_fn, test_case_args, test_case, num_agents, policies, policy, prev_agents):
#     test_case_fn = tc.get_testcase_crazy
#     # test_case_fn = tc.get_testcase_random

#     # test_case_args['policy_to_ensure'] = policies[policy]['policy']
#     # test_case_args['policies'] = ['noncoop', policies[policy]['policy'], 'static']
#     # test_case_args['policy_distr'] = [0.05, 0.9, 0.05]
#     # test_case_args['speed_bnds'] = [0.5, 2.0]
#     # test_case_args['radius_bnds'] = [0.2, 0.8]
#     # test_case_args['side_length'] = [
#     #         {'num_agents': [0,5], 'side_length': [4,5]}, 
#     #         # {'num_agents': [5,np.inf], 'side_length': [6,8]},
#     #         {'num_agents': [5,np.inf], 'side_length': [2,2]},
#     #         ]

#     test_case_args['policy'] = policies[policy]['policy']
#     # test_case_args['num_agents'] = num_agents
#     # test_case_args['test_case_index'] = test_case
#     one_env.plot_policy_name = policy
#     # if 'sensors' in policies[policy]:
#     #     test_case_args['agents_sensors'] = policies[policy]['sensors']
#     # else:
#     #     test_case_args['agents_sensors'] = []
#     # test_case_args['prev_agents'] = prev_agents
#     agents = test_case_fn(**test_case_args)
#     if prev_agents is None:
#         for agent in agents:
#             if 'checkpt_name' in policies[policy] and hasattr(agent.policy, "initialize_network"):
#                 agent.policy.env = env
#                 agent.policy.initialize_network(**policies[policy])
#             if 'sensor_args' in policies[policy]:
#                 for sensor in agent.sensors:
#                     sensor.set_args(policies[policy]['sensor_args'])
#     one_env.set_agents(agents)
#     init_obs = env.reset()
#     one_env.test_case_index = test_case
#     return init_obs

# def get_hyperparam_dict():
#     hyperparameters = {
#         'test_case': test_case,
#         first_hyperparamter_name: first_hyperparameter,
#         'epsilon_scalar_robustness': epsilon_scalar_robustness,
#         'seed': seed,
#         'LPFULL': LPFULL
#     }
#     return hyperparameters

def main():
    np.random.seed(0)

    test_case_fn = tc.full_test_suite
    test_case_args = {}

    if Config.FIXED_RADIUS_AND_VPREF:
        radius_bounds = [0.5, 0.5]
        test_case_args['vpref_constraint'] = True
        test_case_args['radius_bounds'] = radius_bounds
        vpref1_str = 'vpref1.0_r{}-{}/'.format(radius_bounds[0], radius_bounds[1])
    else:
        vpref1_str = ''

    env, one_env = create_env()

    print("Running {test_cases} test cases for {num_agents} for policies: {policies}".format(
        test_cases=Config.NUM_TEST_CASES,
        num_agents=Config.NUM_AGENTS_TO_TEST,
        policies=Config.POLICIES_TO_TEST,
        ))
    with tqdm(total=len(Config.NUM_AGENTS_TO_TEST)*len(Config.POLICIES_TO_TEST)*Config.NUM_TEST_CASES) as pbar:
        for num_agents in Config.NUM_AGENTS_TO_TEST:
            one_env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/../results/full_test_suites/{vpref1_str}{num_agents}_agents/figs/'.format(vpref1_str=vpref1_str, num_agents=num_agents))
            for policy in Config.POLICIES_TO_TEST:
                np.random.seed(0)
                prev_agents = None
                df = pd.DataFrame()
                for test_case in range(Config.NUM_TEST_CASES):
                    ##### Actually run the episode ##########
                    _ = reset_env(env, one_env, test_case_fn, test_case_args, test_case, num_agents, policies, policy, prev_agents)
                    episode_stats, prev_agents = run_episode(env, one_env)
                    df = store_stats(df, {'test_case': test_case, 'policy_id': policy}, episode_stats)
                    ########################################
                    pbar.update(1)

                if Config.RECORD_PICKLE_FILES:
                    file_dir = os.path.dirname(os.path.realpath(__file__)) + '/../results/full_test_suites/{vpref1_str}'.format(vpref1_str=vpref1_str)
                    file_dir += '{num_agents}_agents/stats/'.format(num_agents=num_agents)
                    os.makedirs(file_dir, exist_ok=True)
                    log_filename = file_dir+'/stats_{}.p'.format(policy)
                    # log_filename = file_dir+'/stats_{}_{}.p'.format(policy, now.strftime("%m_%d_%Y__%H_%M_%S"))
                    df.to_pickle(log_filename)

    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")