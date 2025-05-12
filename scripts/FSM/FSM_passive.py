from FSM.State_base import State
import numpy as np

class passive(State):
  

    def execute(self, q, v, a, tau, user_command):
     
        tau_final = np.ones(12) * 0.1
        # print(tau_final)
        # 返回计算后的关节扭矩并进行限制
        return np.clip(tau_final, -30, 30)

    def state_enter(self, q, v):
        # 进入状态时初始化插值
        pass

    def state_exit(self):
        # 目前状态退出时不需要特别处理
        pass