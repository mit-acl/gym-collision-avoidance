import unittest
import os
from importlib import reload, import_module
import gym_collision_avoidance.envs

EPS = 1e-6

def setup(module_name):
    import sys
    modules = list(sys.modules.keys())
    for module in modules:
        if 'gym_collision_avoidance.' in module:
            del sys.modules[module]
    import_module(module_name)
    reload(gym_collision_avoidance.envs)

class TestSum(unittest.TestCase):

    def test_example_script(self):
        setup("gym_collision_avoidance.experiments.src.example")
        main = gym_collision_avoidance.experiments.src.example.main
        # Check that code runs without error
        self.assertTrue(main())
        # Check that plot was generated
        plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/example/000_learning_2agents.png'
        self.assertTrue(os.path.isfile(plot_filename))

    def test_run_cadrl_formations(self):
        setup("gym_collision_avoidance.experiments.src.run_cadrl_formations")
        main = gym_collision_avoidance.experiments.src.run_cadrl_formations.main
        # Check that code runs without error
        self.assertTrue(main())

        ### Check that all plots were generated
        # full episode plots
        for i in range(2):
            plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/cadrl_formations/{}_GA3C-CADRL-10_6agents.png'.format(str(i).zfill(3))
            self.assertTrue(os.path.isfile(plot_filename))
        # animations
        for i in range(2):
            plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/cadrl_formations/animations/{}_GA3C-CADRL-10_6agents.gif'.format(str(i).zfill(3))
            self.assertTrue(os.path.isfile(plot_filename))
            plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/cadrl_formations/animations/{}_GA3C-CADRL-10_6agents.mp4'.format(str(i).zfill(3))

    def test_run_full_test_suite(self):
        setup("gym_collision_avoidance.experiments.src.run_full_test_suite")
        main = gym_collision_avoidance.experiments.src.run_full_test_suite.main
        # Check that code runs without error
        self.assertTrue(main())
        
        ### Check that all plots were generated
        # full episode plots
        for num_agents in [2,3,4]:
            for policy in ['GA3C-CADRL-10', 'CADRL', 'RVO']:
                for i in range(3):
                    plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/full_test_suites/{num_agents}_agents/figs/{tc}_{policy}_{num_agents}agents.png'.format(tc=str(i).zfill(3), policy=policy, num_agents=num_agents)
                    self.assertTrue(os.path.isfile(plot_filename))
                # pickle_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/full_test_suites/{num_agents}_agents/stats/{policy}.p'.format(policy=policy, num_agents=num_agents)
                # self.assertTrue(os.path.isfile(pickle_filename))

if __name__ == '__main__':
    unittest.main()