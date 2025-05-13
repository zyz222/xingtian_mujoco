from FSM.State_base import State
import numpy as np

class StandState(State):
    def __init__(self, dynamics):
        super().__init__(dynamics)
        self.interpolation_step = 0  # 跟踪插值步数
        self.initial_pos = None      # 状态切换时的初始关节位置


    # 最后一项是期望轨迹
    def execute(self, q, v, a, tau, user_command):
        # kp, kd = 120.0, 1.2
        kp, kd = 20.0, 0.8

        # 目标关节位置插值目标（示例针对四条腿，每腿3自由度）
        target_pos = np.array([-2.18, 1.14, q[9], -2.18, 1.14, q[12], -2.18, 1.14, q[15], -2.18, 1.14, q[18]])

        current_q = q[-12:]  # 末12个关节角度
        current_v = v[-12:]

        # 初始化插值起点
        if self.initial_pos is None:
            self.initial_pos = current_q.copy()

        total_steps = 500
        if self.interpolation_step < total_steps:
            # 计算当前插值进度（alpha），根据步数逐渐接近目标
            alpha = self.interpolation_step / total_steps
            self.interpolation_step += 1
        else:
            alpha = 1.0  # 当插值完成后，设置alpha为1
        interpolated_target = (1 - alpha) * self.initial_pos + alpha * target_pos


      
        # # 验证动力学模型：计算逆动力学所需扭矩
        a[6:] = interpolated_target  # 这里 a[6:] 应该是期望的关节加速度，如果插值用于位置，a[6:] 直接置 0 更合理
        expected_tau = self.dynamics.inverse_dynamics(q, v, a)
        print(f"=== Inverse Dynamics τ (full):\n{expected_tau[-12:]}\n")
        # self.run_diagnostics(q,v,a,tau)

        if hasattr(self.dynamics, 'foot_frames'):
            feet = self.dynamics.foot_frames()
        else:
            raise ValueError("无法自动获取 feet 名称，请手动指定。")

        if not feet:
            raise ValueError("足端 frames 列表为空，请检查 foot_frames() 方法或手动指定 feet 名称。")

        print(f"[信息] 使用的足端 frames: {feet}")

        forces = self.dynamics.foot_forces_from_torques(tau, feet, frame="foot", q=q)
        
        # 通过足端力反推扭矩
        tau_computed = self.dynamics.joint_torques_from_forces(forces, frame="foot", q=q)
        
        # # 误差分析
        error_vector = tau - tau_computed[6:]
        error_norm = np.linalg.norm(error_vector)
        relative_error = error_norm / (np.linalg.norm(tau) + 1e-8)
        
        print(f"[验证] τ 重建 L2 范数误差: {error_norm:.6e}")
        print(f"[验证] τ 相对误差: {relative_error:.6%}")
        print(f"[详细] 扭矩误差向量:\n{error_vector}")

        if relative_error > 0.05:
            print("⚠️ [警告] τ 重建误差超过 5%，请检查 Jacobian 或坐标变换！")
        else:
            print("✅ [通过] τ 重建误差在合理范围内。")



        # PD 控制器计算关节扭矩（只对主动关节控制）
        torque = kp * (interpolated_target - current_q) - kd * current_v - expected_tau[-12:]
        
        tau_final = np.clip(torque, -30, 30)
        # print("=== Final Computed Torque (τ):\n", tau_final)

        return tau_final


    def state_enter(self, q, v):
        # 进入状态时初始化插值
        self.initial_pos = q[-12:].copy()  # 将当前关节位置作为起始位置
        self.interpolation_step = 0  # 重置插值步数
        print("Entering StandState, initializing interpolation")
    def state_exit(self):
        # 目前状态退出时不需要特别处理
        pass

    # ===============================
    # 测试方法（直接调用进行全流程验证）
    # ===============================
    def run_diagnostics(self, q, v,a,tau):
        print("\n=== [Diagnostics Start] ===\n")
        # self.state_enter(q, v)

        # # 基础属性检查
        # print("[模型属性]")
        # print(f"自由度 DOF: {self.dynamics.model.nv}")
        # print(f"足端 Frames: {self.dynamics.foot_frames()}\n")

        # # 正向运动学验证
        #   # 正运动学 - 获取四足位置（机体坐标系）
        # foot_names = self.dynamics.foot_frames()
        # foot_positions = [self.dynamics.foot_position(f, 'body', q) for f in foot_names]

        # print("====== Foot Positions (Body Frame) ======")
        # for name, pos in zip(foot_names, foot_positions):
        #     print(f"{name}: {pos}")

        # # # 逆运动学验证：以第1个足端位置为目标，重新解算关节角度
        # foot = foot_names[0]
        # try:
        #     q_solution = self.dynamics.foot_inverse_kinematics(foot, foot_positions[0])
        #     print("=== Inverse Kinematics Solution Found ===")
        #     print(q_solution[-12:])                                       #前几个是body_q
        # except RuntimeError as e:
        #     print("!!! Inverse Kinematics Failed:", e)
        #     q_solution = q.copy()  # 防止崩溃，保持原始配置
        # print("=== body_q ===\n")
        # print(q[:7])

        # # 动力学矩阵验证 
        # print("\n[动力学矩阵验证]")
        # M = self.dynamics.mass_matrix(q)
        # C = self.dynamics.coriolis_matrix(q, v)
        # g = self.dynamics.gravity_vector(q)
        # print(f"M shape: {M.shape}, C shape: {C.shape}, g shape: {g.shape}")
        # tau_left = M @ a + C @ v + g
        # # 右边：通过逆动力学直接计算
        # tau_right = self.dynamics.inverse_dynamics(q, v, a)
        # # tau_ = np.zeros(18)
        # # tau_[-12:] = tau
        # tau_right[-12:] = -tau
        # # 比较误差
        # error = np.linalg.norm(tau_left - tau_right)
        # print(f"Inverse Dynamics Consistency Error: {error:.6e}")
        # if hasattr(self.dynamics, 'foot_frames'):
        #     feet = self.dynamics.foot_frames()
        # else:
        #     raise ValueError("无法自动获取 feet 名称，请手动指定。")

        # if not feet:
        #     raise ValueError("足端 frames 列表为空，请检查 foot_frames() 方法或手动指定 feet 名称。")

        # print(f"[信息] 使用的足端 frames: {feet}")

        # forces = self.dynamics.foot_forces_from_torques(tau, feet, frame="foot", q=q)
        # print(f"[INFO] 获取到的足端力: {forces}")
        # F_stack = np.hstack([forces[f] for f in feet])  # (6 * num_feet, )

        # JTs = [self.dynamics.foot_jacobian(f, q, "foot").T for f in feet]
        # JT_stack = np.hstack(JTs)  # (18, 6 * num_feet)

        # # 提取腿部/轮子的关节自由度部分（排除浮动基座的6自由度）
        # actuated_joint_idx = np.arange(6, 18)  # 或者根据实际情况调整

        # JT_stack_leg = JT_stack[actuated_joint_idx, :]  # (12, N)
        # tau_leg = tau  # (12, )

        # # 验证维度是否匹配
        # assert JT_stack_leg.shape[1] == F_stack.shape[0], "Jacobian 列数与足端力向量维度不匹配！"

        # # 计算重建的 τ
        # tau_reconstructed_leg = JT_stack_leg[-12:,:] @ F_stack

        # # 误差分析
        # error_vector = tau_leg - tau_reconstructed_leg
        # error_norm = np.linalg.norm(error_vector)
        # max_error = np.max(np.abs(error_vector))
        # relative_error = error_norm / (np.linalg.norm(tau_leg) + 1e-8)

        # print(f"[验证] τ 重建误差 L2范数: {error_norm:.6e}")
        # print(f"[验证] τ 最大单元误差: {max_error:.6e}")
        # print(f"[验证] τ 相对误差: {relative_error:.6%}")

        # if relative_error > 0.05:
        #     print("⚠️  [警告] τ 重建误差超过 5%，请检查 Jacobian 计算或足端力估算！")
        # else:
        #     print("✅  [通过] τ 重建误差在可接受范围内。")

        #___________________到这里是没问题的_____________________
        # # 力 ↔ 扭矩 映射验证
        # if hasattr(self.dynamics, 'foot_frames'):
        #     feet = self.dynamics.foot_frames()
        # else:
        #     raise ValueError("无法自动获取 feet 名称，请手动指定。")

        # if not feet:
        #     raise ValueError("足端 frames 列表为空，请检查 foot_frames() 方法或手动指定 feet 名称。")

        # print(f"[信息] 使用的足端 frames: {feet}")

        # forces = self.dynamics.foot_forces_from_torques(tau, feet, frame="foot", q=q)
        
        # # 通过足端力反推扭矩
        # tau_computed = self.dynamics.joint_torques_from_forces(forces, frame="foot", q=q)
        
        # # # 误差分析
        # error_vector = tau - tau_computed[6:]
        # error_norm = np.linalg.norm(error_vector)
        # relative_error = error_norm / (np.linalg.norm(tau) + 1e-8)
        
        # print(f"[验证] τ 重建 L2 范数误差: {error_norm:.6e}")
        # print(f"[验证] τ 相对误差: {relative_error:.6%}")
        # print(f"[详细] 扭矩误差向量:\n{error_vector}")

        # if relative_error > 0.05:
        #     print("⚠️ [警告] τ 重建误差超过 5%，请检查 Jacobian 或坐标变换！")
        # else:
        #     print("✅ [通过] τ 重建误差在合理范围内。")
