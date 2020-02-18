import gym
import os
import numpy as np
import pickle

from gym_collision_avoidance.envs.config import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env, store_stats

# from gym_collision_avoidance.envs.policies.PPOCADRLPolicy import PPOCADRLPolicy
# from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor


np.random.seed(0)

Config.EVALUATE_MODE = True
Config.SAVE_EPISODE_PLOTS = True
Config.SHOW_EPISODE_PLOTS = False
Config.ANIMATE_EPISODES = False
start_from_last_configuration = False
Config.DT = 0.1

# record_pickle_files = True
record_pickle_files = False

test_case_fn = tc.small_test_suite
# test_case_fn = tc.full_test_suite
policies = {
            'GA3C-CADRL-10': {
                'policy': GA3CCADRLPolicy,
                'checkpt_dir': 'IROS18',
                'checkpt_name': 'network_01900000',
                'sensors': [OtherAgentsStatesSensor]
                },
            'GA3C-CADRL-10-AWS': {
                'policy': GA3CCADRLPolicy,
                'checkpt_dir': 'run-20190727_192048-qedrf08y',
                'checkpt_name': 'network_01900000',
                'sensors': [OtherAgentsStatesSensor]
                },
            'GA3C-CADRL-4-AWS': {
                'policy': GA3CCADRLPolicy,
                'checkpt_dir': "run-20190727_015942-jzuhlntn",
                'checkpt_name': 'network_01490000',
                'sensors': [OtherAgentsStatesSensor]
                },
            'CADRL': {
                'policy': CADRLPolicy,
                'sensors': [OtherAgentsStatesSensor]
                },
            # 'RVO': {
            #     'policy': RVOPolicy,
            #     },
            }

num_agents_to_test = [6]
num_test_cases = 8
test_case_args = {}
Config.PLOT_CIRCLES_ALONG_TRAJ = True

Config.NUM_TEST_CASES = num_test_cases

print("Creating environment.")
env, one_env = create_env()
print("Created environment.")

for num_agents in num_agents_to_test:

    one_env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/../results/small_test_suites/{num_agents}_agents/figs/'.format(num_agents=num_agents))

    test_case_args['num_agents'] = num_agents
    stats = {}
    for policy in policies:
        stats[policy] = {}
        stats[policy]['non_collision_inds'] = []
        stats[policy]['all_at_goal_inds'] = []
        stats[policy]['stuck_inds'] = []
    for test_case in range(num_test_cases):
        if start_from_last_configuration:
            if test_case == 0:
                agents = tc.small_test_suite(num_agents=num_agents, test_case_index=0)
            test_case_args['agents'] = agents
            test_case_args['letter'] = letters[test_case % len(letters)]
        else:
            test_case_args['test_case_index'] = test_case
        for policy in policies:
            print('-------')
            one_env.plot_policy_name = policy
            policy_class = policies[policy]['policy']
            test_case_args['agents_policy'] = policy_class
            test_case_args['agents_sensors'] = policies[policy]['sensors']
            agents = test_case_fn(**test_case_args)
            for agent in agents:
                if 'checkpt_name' in policies[policy]:
                    agent.policy.env = env
                    agent.policy.initialize_network(**policies[policy])
            one_env.set_agents(agents)
            init_obs = env.reset()
            one_env.test_case_index = test_case

            times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck, agents = run_episode(env, one_env)

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
            file_dir = os.path.dirname(os.path.realpath(__file__)) + '/results/full_test_suites/'
            file_dir += '{num_agents}_agents/stats/'.format(num_agents=num_agents)
            os.makedirs(file_dir, exist_ok=True)
            fname = file_dir+policy+'.p'
            pickle.dump(stats[policy], open(fname,'wb'))
            print('dumped {}'.format(fname))
# print('---------------------')
# print("Total time_to_goal: {0:.2f}".format(total_time_to_goal))
# print('---------------------')
        

print("Experiment over.")
