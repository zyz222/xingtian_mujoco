from FSM.State_base import State
import numpy as np

class FreeStandState(State):
    def execute(self, q, v,a, user_command):
        kp, kd = 120.0, 1.2
        target_pos = np.array([-2.18, 1.14,0, -2.18, 1.14,0, -2.18, 1.14,0, -2.18, 1.14,0])
        current_q = q[-12:]
        # target_q = np.zeros(19)
        # target_q[-12:] = target_pos
        
        current_v = v[-12:]      #18维度
   

        tau_full = np.zeros(12)
        tau_full = kp * (target_pos - current_q) - kd * (current_v)
        tau_final = tau_full[-12:]
        return np.clip(tau_final, -30, 30)
    def state_enter(self, q, v):

        pass
    def state_exit(self):

        pass
        