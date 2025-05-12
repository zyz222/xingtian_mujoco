from FSM.FSM_stand import StandState
from FSM.FSM_move_wheel import MoveState_Wheel
from FSM.FSM_freestand import FreeStandState
from FSM.FSM_passive import passive

class StateMachine:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.states = {
            "passive":passive(dynamics),
            "stand": StandState(dynamics),
            "free_stand": FreeStandState(dynamics),
            "move": MoveState_Wheel(dynamics),
            # "walk": WalkState(dynamics),
            # "trot": TrotState(dynamics),
            "idle": StandState(dynamics)  # Default to Stand for idle
        }
        self.current_state = self.states["passive"]
        self.previous_state = None
    def change_state(self, state_name, q, v, a, joint_torque, wheel_force):
        print(f"Attempting to change state to: {state_name}")

        # Only change state if the requested state is different from the current state
        if state_name != self.current_state.__class__.__name__.lower():
            print(f"State is changing from {self.current_state.__class__.__name__} to {state_name}")

            # Exit the current state
            if self.previous_state is not None:
                self.previous_state.state_exit()  # Call exit on previous state
            
            # Update previous state
            self.previous_state = self.current_state
            
            # Change to the new state
            self.current_state = self.states[state_name]
            print(f"Entering new state: {state_name}")

            # Call enter on the new state
            self.current_state.state_enter(q, v)
        else:
            print(f"Already in state: {state_name}, no state change required.")


        

    def enter(self, q, v):
        self.current_state.state_enter(q, v)
    def update(self, q, v, a, tau, user_command):
        return self.current_state.execute(q, v, a,tau, user_command)
    
    def exit(self):
        self.current_state.state_exit()
