# gym-collision-avoidance

TODOS:
- Set config option to use array obs instead of dict by default
- Get the test suite code working
- Get the small test suite code working
- Download on a fresh computer and test
- Confirm RVO works
- Confirm DRLLong works
- Confirm CADRL works
- Confirm example.py works
- Confirm sensors work
- Clean up README
- Fix permissions on all files

The [Collision Avoidance](https://bitbucket.org/mfe7/gym-collision-avoidance) is a multiagent
domain featuring continuous state and action spaces.

Currently, agents are all running the same policy, and try to get to their own
goal location, which is chosen at the start of each episode.

Episodes end when all agents reach their goals, collide with one another, or timeout
(take too long to get to goal)

Reward is given to agents when arriving at the goal.

### To replicate experiments in submitted IJRR paper:

```bash
git clone --recursive <url>
cd gym-collision-avoidance/gym_collision_avoidance
./install.sh
```

#### Virtualenv style (easier?)

Formations (spelling out CADRL):
```bash
./experiments/run_cadrl_formations.sh
```

#### Docker style

Start docker and move to correct dir
```
docker_run_cadrl_openai
cd ~/code/gyms/gym-collision-avoidance/gym_collision_avoidance/experiments
```

Formations (spelling out CADRL):
```
python3 run_cadrl_formations.py
```

500 test cases per number of agents, for each algorithm:
```
python3 run_full_test_suite.py
python3 process_full_test_suite_pickles.py
```


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

