# gym-collision-avoidance

The [Collision Avoidance](https://bitbucket.org/mfe7/gym-collision-avoidance) is a multiagent
domain featuring continuous state and action spaces.

Currently, agents are all running the same policy, and try to get to their own
goal location, which is chosen at the start of each episode.

Episodes end when all agents reach their goals, collide with one another, or timeout
(take too long to get to goal)

Reward is given to agents when arriving at the goal.

# Installation

First, install openai's gym. Then install this custom environment.

```bash
cd gym-collision-avoidance
pip install -e .
```
