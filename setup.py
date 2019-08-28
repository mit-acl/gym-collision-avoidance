from setuptools import setup

setup(name='gym_collision_avoidance',
      version='0.0.1',
      install_requires=['gym>=0.2.3',
          'matplotlib==2.1.2', # 2.1.2 to support python2.7 -___-
          'shapely'
          ]
)

# setup(name='gym_collision_avoidance',
#       version='0.0.1',
#       install_requires=['gym>=0.2.3'],
#       packages=['','cadrl_multi', 'cadrl_nn', 'cadrl_test_data'],
#       package_dir={'cadrl_multi':'gym_collision_avoidance/envs/CADRL/scripts/multi', 'cadrl_nn':'gym_collision_avoidance/envs/CADRL/scripts/neural_networks', 'cadrl_test_data':'gym_collision_avoidance/envs/CADRL/scripts/neural_networks/test_data'}
# )
