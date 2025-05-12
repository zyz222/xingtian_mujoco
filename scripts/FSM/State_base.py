class State:
    def __init__(self, dynamics):
        self.dynamics = dynamics

    def execute(self, q, v,a,tau, user_command):
        raise NotImplementedError("Must implement in subclass.")
    
    def state_enter(self, q, v):
        self.q = q
        self.v = v


    def state_exit(self):
        pass