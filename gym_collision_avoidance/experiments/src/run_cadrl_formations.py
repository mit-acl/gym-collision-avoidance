import os
import numpy as np
os.environ['GYM_CONFIG_CLASS'] = 'Formations'
from gym_collision_avoidance.envs import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env, store_stats, policies

def reset_env(env, one_env, test_case_fn, test_case_args, test_case, num_agents, policies, policy, prev_agents=None, start_from_last_configuration=True):
    if prev_agents is None:
        prev_agents = tc.small_test_suite(num_agents=num_agents, test_case_index=0, policies=policies[policy]['policy'], agents_sensors=policies[policy]['sensors'])
        for agent in prev_agents:
            if 'checkpt_name' in policies[policy]:
                agent.policy.env = env
                agent.policy.initialize_network(**policies[policy])
            if 'sensor_args' in policies[policy]:
                for sensor in agent.sensors:
                    sensor.set_args(policies[policy]['sensor_args'])

    test_case_args['agents'] = prev_agents
    test_case_args['letter'] = Config.LETTERS[test_case % len(Config.LETTERS)]
    one_env.plot_policy_name = policy
    agents = test_case_fn(**test_case_args)
    one_env.set_agents(agents)
    init_obs = env.reset()
    one_env.test_case_index = test_case
    return init_obs

def main():
    np.random.seed(0)

    test_case_fn = tc.formation
    test_case_args = {}

    env, one_env = create_env()

    one_env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + '/../results/cadrl_formations/')

    for num_agents in Config.NUM_AGENTS_TO_TEST:
        for policy in Config.POLICIES_TO_TEST:
            np.random.seed(0)
            prev_agents = None
            for test_case in range(Config.NUM_TEST_CASES):
                _ = reset_env(env, one_env, test_case_fn, test_case_args, test_case, num_agents, policies, policy, prev_agents)
                episode_stats, prev_agents = run_episode(env, one_env)

    return True

if __name__ == '__main__':
    main()
    print("Experiment over.")