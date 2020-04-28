import numpy as np

from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy

class LearningPolicy(ExternalPolicy):
    """ An RL policy that is still being trained or otherwise fed actions from an external script, but still needs to convert the external actions to this env's format
    """
    def __init__(self):
        ExternalPolicy.__init__(self, str="learning")
        self.is_still_learning = True
        self.ppo_or_learning_policy = True

    def external_action_to_action(self, agent, external_action):
        """ Convert the external_action into an action for this environment using properties about the agent.

        For instance, RL network might have continuous outputs between [0-1], which could be 
        scaled by this method to correspond to a speed between [0, pref_speed],
        without the RL network needing to know the agent's preferred speed.

        Args:
            agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): the agent who has this policy
            external_action (int, array, ...): what the learning system returned for an action

        Returns:
            [speed, heading_change] command

        """

        # external_action: [speed scaling btwn 0-1, max heading angle delta btwn 0-1]
        heading_change = agent.max_heading_change*(2.*external_action[1] - 1.)
        speed = agent.pref_speed * external_action[0]
        actions = np.array([speed, heading_change])
        return actions