

class Policy(object):
    def __init__(self, str="NoPolicy"):
    	self.str = str
    	self.is_still_learning = False

    def find_next_action(self, agents):
        raise NotImplementedError
