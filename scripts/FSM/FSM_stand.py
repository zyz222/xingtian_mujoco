from FSM.State_base import State
import numpy as np

class StandState(State):
    def __init__(self, dynamics):
        super().__init__(dynamics)
        self.interpolation_step = 0  # 跟踪插值步数
        self.initial_pos = None      # 状态切换时的初始关节位置



    def execute(self, q, v, a, tau, user_command):
        kp, kd = 120.0, 1.2
        
        # 目标位置（假设目标是12个自由度的关节位置）
        target_pos = np.array([-2.18, 1.14, q[9], -2.18, 1.14, q[12], -2.18, 1.14, q[15], -2.18, 1.14, q[18]])
        
        # 当前的位置和速度
        current_q = q[-12:]
        current_v = v[-12:]

        # 如果插值还没有开始，初始化插值的初始位置
        if self.initial_pos is None:
            self.initial_pos = current_q.copy()

        total_steps = 500  # 设置插值的总步数
        if self.interpolation_step < total_steps:
            # 计算当前插值进度（alpha），根据步数逐渐接近目标
            alpha = self.interpolation_step / total_steps
            self.interpolation_step += 1
        else:
            alpha = 1.0  # 当插值完成后，设置alpha为1

        # 使用线性插值来逐渐接近目标位置
        interpolated_target = (1 - alpha) * self.initial_pos + alpha * target_pos
        print(f"self.interpolation_step: {self.interpolation_step}")
        print(f"Alpha: {alpha}")
        print(f"Interpolated Target: {interpolated_target}")
        # 使用PD控制器计算关节扭矩
        tau_full = kp * (interpolated_target - current_q) - kd * current_v
        tau_final = tau_full[-12:]
        # print(tau_final)
        # 返回计算后的关节扭矩并进行限制
        return np.clip(tau_final, -30, 30)

    def state_enter(self, q, v):
        # 进入状态时初始化插值
        if self.initial_pos is None:
            self.initial_pos = q[-12:].copy()  # 将当前关节位置作为起始位置
        self.interpolation_step = 0  # 重置插值步数
        print("Entering StandState, initializing interpolation")
    def state_exit(self):
        # 目前状态退出时不需要特别处理
        pass