from io import open

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gym_collision_avoidance",
    version="0.0.3",
    description="Simulation environment for collision avoidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mit-acl/gym-collision-avoidance",
    author="Michael Everett, Yu Fan Chen, Jonathan P. How, MIT",  # Optional
    keywords="robotics planning gym rl",  # Optional
    install_requires=[
        "tensorflow",
        "Pillow",
        "PyOpenGL",
        "pyyaml",
        "matplotlib",
        "pytz",
        "imageio==2.4.1",
        "gym",
        "moviepy",
        "pandas",
    ],
    packages=find_packages(),
    package_data={
        "": [
            "envs/policies/CADRL/**/*.p",
            "envs/policies/GA3C_CADRL/checkpoints/**/network*",
        ]
    },
    include_package_data=True,
)
