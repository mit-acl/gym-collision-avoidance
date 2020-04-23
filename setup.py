from setuptools import setup

setup(
    name='gym_collision_avoidance',
    version='1.0.0',
    description='Simulation environment for collision avoidance',
    url='https://github.com/mit-acl/gym-collision-avoidance',
    author='Michael Everett, Yu Fan Chen, Jonathan P. How, MIT',  # Optional
    keywords='robotics planning gym rl',  # Optional
    python_requires='>=3.0, <4',
    install_requires=[
        'tensorflow==1.15.2',
        'Pillow',
        'PyOpenGL',
        'pyyaml',
        'matplotlib>=3.0.0',
        'shapely',
        'pytz',
        'imageio==2.4.1',
        'gym',
        'moviepy',
        'pandas',
    ],
)