

class Policy(object):
    def __init__(self, str="NoPolicy"):
    	self.str = str

    def find_next_action(self, agents):
        raise NotImplementedError
