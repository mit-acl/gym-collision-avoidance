import os
import numpy as np
import pickle
from tqdm import tqdm

os.environ['GYM_CONFIG_CLASS'] = 'CollectRegressionDataset'
from gym_collision_avoidance.envs import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env

np.random.seed(0)

def fill(env, one_env, num_datapts=10):
    obs = env.reset()
    # print(obs)
    # assert(0)
    STATES = np.empty((num_datapts, obs[0].shape[-1]-1))
    ACTIONS = np.empty((num_datapts, 2))
    VALUES = np.empty((num_datapts, 1))
    ind = 0
    with tqdm(total=num_datapts) as pbar:
        while True:
            obs = env.reset()
            game_over = False
            while not game_over:
                for agent_ind, agent in enumerate(one_env.agents):
                    action, value = agent.policy.find_next_action_and_value([], one_env.agents, agent_ind)
                    STATES[ind, :] = obs[0][0, agent_ind, 1:]
                    ACTIONS[ind, :] = action
                    VALUES[ind, :] = value
                    ind += 1
                    pbar.update(1)
                    if ind == num_datapts:
                        return STATES, ACTIONS, VALUES
                obs, rewards, game_over, info = env.step([{}])

def main():
    filename_template = os.path.dirname(os.path.realpath(__file__)) + '/../../datasets/regression/{num_agents}_agents_{dataset_name}_cadrl_dataset_action_value_{mode}.p'
    env, one_env = create_env()
    modes = [
        {
            'mode': 'train',
            'num_datapts': 100000,
        },
        {
            'mode': 'test',
            'num_datapts': 20000,
        },
    ]
    for mode in modes:
        STATES, ACTIONS, VALUES = fill(env, one_env, num_datapts=mode['num_datapts'])
        filename = filename_template.format(mode=mode['mode'], dataset_name=Config.DATASET_NAME, num_agents=Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
        file_dir = os.path.dirname(filename)
        os.makedirs(file_dir, exist_ok=True)

        with open(filename, "wb") as f:
            pickle.dump([STATES,ACTIONS,VALUES], f)

    print("Files written.")

if __name__ == '__main__':
    main()