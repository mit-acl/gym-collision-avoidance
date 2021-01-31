from setuptools import setup, find_packages
from io import open

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='gym_collision_avoidance',
    version='0.0.2',
    description='Simulation environment for collision avoidance',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mit-acl/gym-collision-avoidance',
    author='Michael Everett, Yu Fan Chen, Jonathan P. How, MIT',  # Optional
    keywords='robotics planning gym rl',  # Optional
    python_requires='<3.8',
    install_requires=[
        'tensorflow==1.15',
        'Pillow',
        'PyOpenGL',
        'pyyaml',
        'matplotlib',
        'pytz',
        'imageio==2.4.1',
        'gym',
        'moviepy',
        'pandas',
        'opencv-python<=4.2.0.32', # solely for stable_baselines (needs to be this version or lower for python2.7)
        'stable_baselines<2.4.1', # needs to be this version or lower for python2.7 (setup.py uses subprocess in a python3 way)
    ],
    packages=find_packages(),
    include_package_data=True,
)