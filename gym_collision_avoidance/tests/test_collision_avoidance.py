import unittest
import os

EPS = 1e-6

class TestSum(unittest.TestCase):

    def test_example_script(self):
        from gym_collision_avoidance.experiments.src.example import main
        # Check that code runs without error
        self.assertTrue(main())
        # Check that plot was generated
        plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/example/000_learning_2agents.png'
        self.assertTrue(os.path.isfile(plot_filename))


if __name__ == '__main__':
    unittest.main()