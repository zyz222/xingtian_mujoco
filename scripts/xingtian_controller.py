import zmq
import json
import numpy as np
import threading
import time
import sys
import os
from queue import Queue
import termios
import tty

# -----------------------------------------------------------------------------
# 自己写的动力学封装
# -----------------------------------------------------------------------------
from xingtian_robot import FloatingBaseDynamics

# ★ 绝对路径更稳，也可以换成 ROS_PACKAGE_PATH
URDF_PATH = "xingtian_sym_model/urdf/xingtian_fixed.urdf"
MESH_DIRS = ["xingtian_sym_model/meshes"]

# -----------------------------------------------------------------------------
# 控制器主类
# -----------------------------------------------------------------------------
class MujocoController:
    """通过 ZMQ 与 Mujoco-Sim 交换状态 / 力矩，并在本地跑 Pinocchio 动力学。"""
    def __init__(self, state_sub_port: int = 5556, cmd_pub_port: int = 5555):
        # ---------- ZMQ ----------
        self.context = zmq.Context()
        self.state_sub_socket = self.context.socket(zmq.SUB)
        self.state_sub_socket.connect(f"tcp://localhost:{state_sub_port}")
        self.state_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.cmd_pub_socket = self.context.socket(zmq.PUB)
        self.cmd_pub_socket.connect(f"tcp://localhost:{cmd_pub_port}")

        # ---------- Pinocchio / 浮动基模型 ----------
        self.robot = FloatingBaseDynamics(
            URDF_PATH,
            package_dirs=MESH_DIRS,
            visual=False,
        )
        # 足端 frame 名按 URDF link 名
        self.feet = ["LF_wheel", "LR_wheel", "RF_wheel", "RR_wheel"]

        # ---------- 线程 ----------
        self.running = False
        self.state_thread = threading.Thread(target=self._state_and_control_loop, daemon=True)
        self.cmd_thread = threading.Thread(target=self._keyboard_command_loop, daemon=True)
        # self.status_thread = threading.Thread(target=self._status_display_loop, daemon=True)

        # ---------- 共享状态量 ----------
        self.mode_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.mode = 0          # 0: passive, 1: stand, 2: move, 3: idle
        self.user_ctrl_sign = 0.0
        self.last_mode = -1

        # 来自模拟器的传感器缓存（初始化为 0）
        self.joint_pos   = np.zeros(8)
        self.joint_vel   = np.zeros(8)
        self.wheel_pos   = np.zeros(4)
        self.wheel_vel   = np.zeros(4)
        self.body_pos    = np.zeros(3)
        self.body_quat   = np.array([1,0,0,0])    #w x y z
        self.body_linvel = np.zeros(3)
        self.body_angvel = np.zeros(3)
        self.body_acc    = np.zeros(3)
        self.body_acc_w   = np.zeros(3)
        self.joint_torque= np.zeros(8)
        self.wheel_torque= np.zeros(4)
        self.wheel_force = np.zeros(12)  # 4 × (fx,fy,fz)


        # 用于插值站立姿态
        self.interpolation_step = 0
        self.initial_pos = self.joint_pos.copy()
        #计算发布的力矩
        self.joint_tau = np.zeros(8)
        self.wheel_tau = np.zeros(4)
    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        self.running = True
        self.state_thread.start()
        self.cmd_thread.start()
        print("Controller started")

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.context.term()
        print("Controller stopping…")

    # ------------------------------------------------------------------
    # ZMQ 线程：收状态 → 算力矩 → 发送
    # ------------------------------------------------------------------
    def _state_and_control_loop(self):
        while self.running:
            state = self.get_state(timeout=10)
            if state is None:
                torque_cmd = self._compute_control_command(None)
            else:
                torque_cmd = self._compute_control_command(state)

            self.send_control_command(torque_cmd)
    # ------------------------------------------------------------------
    # 键盘线程：模式切换
    # ------------------------------------------------------------------
    def _keyboard_command_loop(self):
        print("Modes: [1] Stand, [2] Move, [3] Idle, [Q] quit")
        def get_key():
            fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd); ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            return ch
        while self.running:
            try:
                cmd = get_key().lower()
                with self.mode_lock:
                    if cmd == '1':
                        self.mode = 1
                    elif cmd == '2':
                        self.mode = 2
                    elif cmd == '3':
                        self.mode = 3
                    elif cmd == '0':
                        self.mode = 0
                    elif cmd == 'w':
                        self.user_ctrl_sign += 0.1
                    elif cmd == 's':
                        self.user_ctrl_sign -= 0.1
                    elif cmd == 'a':
                        self.user_ctrl_sign -= 0.1
                    elif cmd == 'd':
                        self.user_ctrl_sign += 0.1
                    elif cmd == 'q':
                        print("Exiting...")
                        self.stop()
                        sys.exit(0)
                    else:
                        print("Unknown key pressed.")
            except Exception as e:
                print(f"Keyboard input error: {e}")
                break

    # ------------------------------------------------------------------
    # 计算控制命令（核心）
    # ------------------------------------------------------------------
    def _compute_control_command(self, state):
        # -------- 1) 把模拟器状态写入本地缓存 --------
        if state:
            self.joint_pos   = np.asarray(state['joint_pos']).flatten()
            self.joint_vel   = np.asarray(state['joint_vel']).flatten()
            self.wheel_pos   = np.asarray(state['wheel_pos']).flatten()
            self.wheel_vel   = np.asarray(state['wheel_vel']).flatten()
            self.body_pos    = np.asarray(state['body_pos']).flatten()
            self.body_quat   = np.asarray(state['body_quat']).flatten()
            self.body_linvel = np.asarray(state['body_linvel']).flatten()
            self.body_angvel = np.asarray(state['body_angvel']).flatten()
            self.body_acc   = np.asarray(state['imu']['acc']).flatten()
            self.body_acc_w = np.asarray(state['imu']['gyro']).flatten()
            self.joint_torque = np.array(state['joint_torque']).flatten()
            self.wheel_torque = np.array(state['wheel_torque']).flatten()
            self.wheel_force = np.array(state['wheel_force']).flatten()
        # ------------------------------------------------------------------
        # 把模拟器状态同步到 Pinocchio
        # 模拟器返回顺序:
        #   joint_pos  = [LF_hip, LF_knee, LR_hip, LR_knee, RR_hip, RR_knee, RF_hip, RF_knee]
        #   wheel_pos  = [LF_wheel, LR_wheel, RR_wheel, RF_wheel]
        # Pinocchio 顺序:
        #   free-flyer (7)  +  LF(hip,knee,wheel)  +  LR(...)  +  RF(...)  +  RR(...)
        #   即  q[ 7:10] = LF, q[10:13] = LR, q[13:16] = RF, q[16:19] = RR
        # self.joint_pos[::2] -=2.18
        # self.joint_pos[1::2] += 1.14         #水平向前为0度，后腿是向后
        q = np.zeros(self.robot.model.nq)
        v = np.zeros(self.robot.model.nv)
        a = np.zeros(self.robot.model.nv)
        # --- free‑flyer ---
        q[:3]  = self.body_pos                  # x y z
        q[3:7] = self.body_quat[[1, 2, 3, 0]]  # 正确转换顺序 wxyz → xyzw
        v[:6]  = np.hstack([self.body_linvel,    # vx vy vz
                             self.body_angvel]) # wx wy wz
        a[:6]  = np.hstack([self.body_acc,    # ax ay az
                              self.body_acc_w])
        # --- 关节位置 --------------------------------------------------
        # LF
        q[7]  = self.joint_pos[0]   # LF_hip
        q[8]  = self.joint_pos[1]   # LF_knee
        q[9]  = self.wheel_pos[0]   # LF_wheel
        # LR
        q[10] = self.joint_pos[2]   # LR_hip
        q[11] = self.joint_pos[3]   # LR_knee
        q[12] = self.wheel_pos[1]   # LR_wheel
        # RF
        q[13] = self.joint_pos[6]   # RF_hip
        q[14] = self.joint_pos[7]   # RF_knee
        q[15] = self.wheel_pos[3]   # RF_wheel
        # RR
        q[16] = self.joint_pos[4]   # RR_hip
        q[17] = self.joint_pos[5]   # RR_knee
        q[18] = self.wheel_pos[2]   # RR_wheel

        # --- 关节速度 --------------------------------------------------
        v[6]  = self.joint_vel[0]   # LF_hip_vel
        v[7]  = self.joint_vel[1]   # LF_knee_vel
        v[8]  = self.wheel_vel[0]   # LF_wheel_vel
        v[9]  = self.joint_vel[2]   # LR_hip_vel
        v[10] = self.joint_vel[3]   # LR_knee_vel
        v[11] = self.wheel_vel[1]   # LR_wheel_vel
        v[12] = self.joint_vel[6]   # RF_hip_vel
        v[13] = self.joint_vel[7]   # RF_knee_vel
        v[14] = self.wheel_vel[3]   # RF_wheel_vel
        v[15] = self.joint_vel[4]   # RR_hip_vel
        v[16] = self.joint_vel[5]   # RR_knee_vel
        v[17] = self.wheel_vel[2]   # RR_wheel_vel

        # 同步到动力学模型
        self.robot.set_state(q, v)
        # 打印每个足端在 "world" 和 "body" 下的位置
        # for foot in ['LF_wheel', 'LR_wheel', 'RF_wheel', 'RR_wheel']:
        #     pos_world = self.robot.frame_pose(foot, q).translation
        #     pos_body  = self.robot.foot_position(foot, 'body', q)
        #     print(f"{foot}: world = {pos_world.round(3)}, body = {pos_body.round(3)}")

        # 打印足端位置（body frame）
        foot_pos_body = {f: self.robot.foot_position(f, 'body') for f in self.feet}
        with self.print_lock:
            print("Feet (body frame):", {k: p.round(3).tolist() for k, p in foot_pos_body.items()})

        # -------- 2) 选择模式控制策略 --------
        with self.mode_lock:
            mode = self.mode
        self.joint_tau = np.zeros(8)
        self.wheel_tau = np.zeros(4)
        if mode == 0:     #test 零位
            target_pos = np.zeros(8)
             # 如果模式切换了，重新插值初始化
            if self.last_mode != 0:
                self.interpolation_step = 0
                self.initial_pos = self.joint_pos.copy()
            kp = 120.0
            kd = 1.2
            total_steps = 500
            if self.interpolation_step < total_steps:
                alpha = self.interpolation_step / total_steps
                self.interpolation_step += 1
            else:
                alpha = 1.0
            interpolated_target = (1 - alpha) * self.initial_pos + alpha * target_pos
            self.joint_tau = kp * (interpolated_target - self.joint_pos) - kd * self.joint_vel
        elif mode == 1:  # Stand
            # 如果模式切换了，重新插值初始化
            if self.last_mode != 1:
                self.interpolation_step = 0
                self.initial_pos = self.joint_pos.copy()
            target_pos = np.array([-2.18, 1.14, -2.18, 1.14, -2.18, 1.14, -2.18, 1.14])
            kp = 120.0
            kd = 1.2
            total_steps = 500
            if self.interpolation_step < total_steps:
                alpha = self.interpolation_step / total_steps
                self.interpolation_step += 1
            else:
                alpha = 1.0
            interpolated_target = (1 - alpha) * self.initial_pos + alpha * target_pos
            self.joint_tau = kp * (interpolated_target - self.joint_pos) - kd * self.joint_vel
            self.wheel_tau = np.ones(4) * self.user_ctrl_sign
            self.wheel_tau[0] *= -1
            self.wheel_tau[3] *= -1
        elif mode == 2:  # Move (给轮子一个前驱扭矩)
            self.wheel_tau = np.ones(4)*2.0
        elif mode == 3:
            pass  # Idle null torques

        self.last_mode = mode

        # -------- 3) 把腿关节力矩映射为 Pinocchio 的 12×1 tau -------- LF LR RR RF   前轴都朝右，后轴朝左
        tau_full = np.zeros(self.robot.model.nv)
        tau_full[6:] = np.concatenate([self.joint_tau, self.wheel_tau])

        return np.clip(tau_full[6:], -30, 30)  # 仅发送 12 个执行器力矩

    # ------------------------------------------------------------------
    # ZMQ 帮手
    # ------------------------------------------------------------------
    def get_state(self, timeout=5):
        poller = zmq.Poller(); poller.register(self.state_sub_socket, zmq.POLLIN)
        evts = dict(poller.poll(timeout))
        if self.state_sub_socket in evts:
            return self.state_sub_socket.recv_json()
        return None

    def send_control_command(self, cmd):
        self.cmd_pub_socket.send_json({'control_cmd': cmd.tolist()})

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ctrl = MujocoController()
    ctrl.start()
    try:
        while ctrl.running:
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()
