class Sensor(object):
    """ Each :class:`~gym_collision_avoidance.envs.agent.Agent` has a list of these, which compute a measurement about the environment/other Agents

    """
    def __init__(self):
        pass

    def sense(self, agents, agent_index, top_down_map):
        """ Dummy method to be re-implemented by each Sensor subclass
        """
        raise NotImplementedError

    def set_args(self, args):
        """ Update several class attributes (in dict format) of the Sensor object
        
        Args:
            args (dict): {'arg_name1': new_value1, ...} sets :code:`self.arg_name1 = new_value1`, etc. 

        """
        # Supply a dict of arg key value pairs
        for arg, value in args.items():
            # print("Setting self.{} to {}".format(arg, value))
            setattr(self, arg, value)
