from FSM.State_base import State
import numpy as np

class MoveState_Wheel(State):
    def execute(self, q, v, user_command):
        tau_full = np.zeros(13)
        wheel_torque = user_command.get("wheel_torque", 2.0)

        tau_full[3] = wheel_torque    # LF_WHEEL
        tau_full[6] = wheel_torque    # LR_WHEEL
        tau_full[9] = wheel_torque    # RF_WHEEL
        tau_full[12] = wheel_torque   # RR_WHEEL

        return np.clip(tau_full, -30, 30)
