import os
import numpy as np
import pickle
from tqdm import tqdm

from gym_collision_avoidance.envs.config import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env

from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy

np.random.seed(0)

Config.EVALUATE_MODE = True
Config.SAVE_EPISODE_PLOTS = True
Config.SHOW_EPISODE_PLOTS = False
Config.DT = 0.1
start_from_last_configuration = False

results_subdir = 'trajectory_dataset'

# test_case_fn = tc.get_testcase_2agents_swap
test_case_fn = tc.get_testcase_random
policies = {
            'RVO': {
                'policy': RVOPolicy,
                },
            # 'GA3C-CADRL-10': {
            #     'policy': GA3CCADRLPolicy,
            #     'checkpt_dir': 'IROS18',
            #     'checkpt_name': 'network_01900000'
            #     },
            }

num_agents_to_test = [2]
num_test_cases = 500
test_case_args = {}
Config.PLOT_CIRCLES_ALONG_TRAJ = True
Config.NUM_TEST_CASES = num_test_cases

def add_traj(agents, trajs, dt, traj_i, max_ts):
    agent_i = 0
    other_agent_i = (agent_i + 1) % 2
    agent = agents[agent_i]
    other_agent = agents[other_agent_i]
    max_t = int(max_ts[agent_i])
    future_plan_horizon_secs = 3.0
    future_plan_horizon_steps = int(future_plan_horizon_secs / dt)

    for t in range(max_t):
        robot_linear_speed = agent.global_state_history[t, 9]
        robot_angular_speed = agent.global_state_history[t, 10] / dt

        t_horizon = min(max_t, t+future_plan_horizon_steps)
        future_linear_speeds = agent.global_state_history[t:t_horizon, 9]
        future_angular_speeds = agent.global_state_history[t:t_horizon, 10] / dt
        predicted_cmd = np.dstack([future_linear_speeds, future_angular_speeds])

        future_positions = agent.global_state_history[t:t_horizon, 1:3]

        d = {
            'control_command': np.array([
                robot_linear_speed,
                robot_angular_speed
                ]),
            'predicted_cmd': predicted_cmd,
            'future_positions': future_positions,
            'pedestrian_state': {
                'position': np.array([
                    other_agent.global_state_history[t, 1],
                    other_agent.global_state_history[t, 2],
                    ]),
                'velocity': np.array([
                    other_agent.global_state_history[t, 7],
                    other_agent.global_state_history[t, 8],
                    ])
            },
            'robot_state': np.array([
                agent.global_state_history[t, 1],
                agent.global_state_history[t, 2],
                agent.global_state_history[t, 10],
                ]),
            'goal_position': np.array([
                agent.goal_global_frame[0],
                agent.goal_global_frame[1],
                ])
        }
        trajs[traj_i].append(d)

#     global_state = np.array([self.t,
#                                  self.pos_global_frame[0],
#                                  self.pos_global_frame[1],
#                                  self.goal_global_frame[0],
#                                  self.goal_global_frame[1],
#                                  self.radius,
#                                  self.pref_speed,
#                                  self.vel_global_frame[0],
#                                  self.vel_global_frame[1],
#                                  self.speed_global_frame,
#                                  self.heading_global_frame])

    return trajs


def main():
    env, one_env = create_env()
    dt = one_env.dt_nominal
    file_dir_template = os.path.dirname(os.path.realpath(__file__)) + '/../results/{results_subdir}/{num_agents}_agents'

    trajs = [[] for _ in range(num_test_cases)]

    for num_agents in num_agents_to_test:

        file_dir = file_dir_template.format(num_agents=num_agents, results_subdir=results_subdir)
        plot_save_dir = file_dir + '/figs/'
        os.makedirs(plot_save_dir, exist_ok=True)
        one_env.plot_save_dir = plot_save_dir

        test_case_args['num_agents'] = num_agents
        test_case_args['side_length'] = 7
        for test_case in tqdm(range(num_test_cases)):
            # test_case_args['test_case_index'] = test_case
            # test_case_args['num_test_cases'] = num_test_cases
            for policy in policies:
                one_env.plot_policy_name = policy
                policy_class = policies[policy]['policy']
                test_case_args['agents_policy'] = policy_class
                agents = test_case_fn(**test_case_args)
                for agent in agents:
                    if 'checkpt_name' in policies[policy]:
                        agent.policy.env = env
                        agent.policy.initialize_network(**policies[policy])
                one_env.set_agents(agents)
                one_env.test_case_index = test_case
                init_obs = env.reset()

                times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck, agents = run_episode(env, one_env)

                max_ts = [t / dt for t in times_to_goal]
                trajs = add_traj(agents, trajs, dt, test_case, max_ts)

        # print(trajs)
                
        one_env.reset()

        pkl_dir = file_dir + '/trajs/'
        os.makedirs(pkl_dir, exist_ok=True)
        fname = pkl_dir+policy+'.pkl'
        pickle.dump(trajs, open(fname,'wb'))
        print('dumped {}'.format(fname))

    print("Experiment over.")

if __name__ == '__main__':
    main()