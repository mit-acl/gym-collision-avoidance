class Sensor(object):
    def __init__(self):
        pass

    def sense(self, agents, agent_index, top_down_map):
        raise NotImplementedError

    def set_args(self, args):
        # Supply a dict of arg key value pairs
        for arg, value in args.items():
            # print("Setting self.{} to {}".format(arg, value))
            setattr(self, arg, value)
