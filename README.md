# gym-collision-avoidance

The [Collision Avoidance](https://bitbucket.org/mfe7/gym-collision-avoidance) is a multiagent
domain featuring continuous state and action spaces.

Currently, agents are all running the same policy, and try to get to their own
goal location, which is chosen at the start of each episode.

Episodes end when all agents reach their goals, collide with one another, or timeout
(take too long to get to goal)

Reward is given to agents when arriving at the goal.

### To replicate experiments in submitted IJRR paper:

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
