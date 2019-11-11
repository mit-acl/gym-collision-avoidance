# gym-collision-avoidance

<img src="misc/000_GA3C-CADRL-10_6agents.gif" width="500" alt="Agents spelling ``C''">

This is a multiagent domain featuring continuous state and action spaces.

Agents try to get to their own goal location (chosen at the start of each episode) by using one of many collision avoidance policies implemented.
Episodes end when agents reach their goal or collide.
Agents can observe the environment through one of many types of sensors, including one which provides raw state info about other agents.
Reward is given to agents when arriving at the goal.

**Objective**: Provide a flexible codebase, reduce time spent re-implementing existing works, and establish baselines for multiagent collision avoidance problems.

### Install

Grab the code from github, initialize submodules, install dependencies and src code
```bash
git clone --recursive git@github.com:mit-acl/gym-collision-avoidance.git
./install.sh
```

### Minimum working example

To simulate a 2-agent scenario:
```bash
./example.sh
```

This will save a plot in `experiments/results/example` so you can visualize the agents' trajectories.

You can use `example.py` as a starting point to write code for this environment.

---

### Further experiments

#### To replicate experiments in submitted IJRR paper:

Formations (spelling out CADRL):
```bash
./experiments/run_cadrl_formations.sh
```

Full test suite:
```bash
./experiments/run_full_test_suite.sh
```

---

### Common Issues

#### On OSX

```
RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework.
```

Add a line `backend: TkAgg` to `~/.matplotlib/matplotlibrc`.


```
error: Cannot compile MPI programs. Check your configuration!!!
```

Do:
```bash
brew install mpich
```

```
error with matplotlib and freetype not being found
```

Do:
```bash
brew install pkg-config
```

---

### Policies Implemented

Learning-based:
- SA-CADRL: [Socially Aware Motion Planning with Deep Reinforcement Learning
](https://arxiv.org/pdf/1703.08862.pdf)
- GA3C-CADRL: [Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning](https://arxiv.org/pdf/1805.01956.pdf)
- DRL_Long: [Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning](https://arxiv.org/abs/1709.10082)

Classical:
- RVO/ORCA: [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)
- Non-Cooperative (constant velocity toward goal position)
- Static (zero velocity)

Desired Additions:
- DWA
- Social Forces
- Additional learning-based methods
- Centralized planners

---

### TODOs:
- Get DRLLong working by pointing to correct commit on mfe's fork (cuda in python)
- Confirm sensors work
- Get GA3C-CADRL to actually use the dict obs?
- Clean up README
- Fix permissions on all files

---

### If you find this code useful, please consider citing:

M. Everett, Y. Chen, and J. P. How, "Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018

Paper: https://arxiv.org/abs/1805.01956
Video: https://www.youtube.com/watch?v=XHoXkWLhwYQ

Bibtex:
```
@inproceedings{Everett18_IROS,
  address = {Madrid, Spain},
  author = {Everett, Michael and Chen, Yu Fan and How, Jonathan P.},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  date-modified = {2018-10-03 06:18:08 -0400},
  month = sep,
  title = {Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning},
  year = {2018},
  url = {https://arxiv.org/pdf/1805.01956.pdf},
  bdsk-url-1 = {https://arxiv.org/pdf/1805.01956.pdf}
}
```
