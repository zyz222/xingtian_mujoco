
import zmq
import numpy as np
import threading
import time

#-----------------------------------------------------------------------------
# 自己写的动力学封装
# -----------------------------------------------------------------------------
from xingtian_robot import FloatingBaseDynamics

# 绝对路径更稳，也可以换成 ROS_PACKAGE_PATH
URDF_PATH = "xingtian_sym_model/urdf/xingtian_sym_model.urdf"
MESH_DIRS = ["xingtian_sym_model/meshes"]

# -----------------------------------------------------------------------------
# 控制器主类
# -----------------------------------------------------------------------------
class MujocoController:
    """通过 ZMQ 与 Mujoco-Sim 交换状态 / 力矩，并在本地跑 Pinocchio 动力学。"""
    def __init__(self, status_queue, state_sub_port: int = 5556, cmd_pub_port: int = 5555):
        # ---------- ZMQ ----------
        self.context = zmq.Context()
        self.state_sub_socket = self.context.socket(zmq.SUB)
        self.state_sub_socket.connect(f"tcp://localhost:{state_sub_port}")
        self.state_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.cmd_pub_socket = self.context.socket(zmq.PUB)
        self.cmd_pub_socket.connect(f"tcp://localhost:{cmd_pub_port}")

        # ---------- Pinocchio / 浮动基模型 ----------
        self.robot = FloatingBaseDynamics(URDF_PATH,package_dirs=MESH_DIRS,visual=False)
        # ---------- 线程 ----------
        # self.running = False
        self.status_queue = status_queue
        self.stop_event = threading.Event()
        # ---------- 线程安全锁 ----------
        self._state_lock = threading.Lock()

        self._init_sim_state_buffers()
    def _init_sim_state_buffers(self):
        # 来自模拟器的传感器缓存（初始化为 0）
        with self._state_lock:
            self.joint_pos   = np.zeros(12)
            self.joint_vel   = np.zeros(12)
            self.joint_torque= np.zeros(12)
            self.body_pos    = np.zeros(3)
            self.body_quat   = np.array([1,0,0,0])    #w x y z
            self.body_linvel = np.zeros(3)
            self.body_angvel = np.zeros(3)
            self.body_acc    = np.zeros(3)
            self.body_acc_w   = np.zeros(3)
            self.wheel_force = np.zeros(12)  # 4 × (fx,fy,fz)
            self.q = np.zeros(self.robot.model.nq)      
            self.v = np.zeros(self.robot.model.nv)
            self.a = np.zeros(self.robot.model.nv)
    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        """启动状态监听线程"""
        self.listener_thread = threading.Thread(
            target=self._state_listener_loop,
            daemon=True
        )
        self.listener_thread.start()
        return True

    def stop(self):
        self.stop_event.set()
        # time.sleep(0.1)
        self.context.term()
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=0.5)
        print("Controller stopping…")

    # ------------------------------------------------------------------
    # ZMQ 线程：收状态 → 算力矩 → 发送
    # ------------------------------------------------------------------
    def _state_listener_loop(self):
        """状态监听主循环（改进版）"""
        poller = zmq.Poller()
        poller.register(self.state_sub_socket, zmq.POLLIN)
        
        while not self.stop_event.is_set():
            try:
                events = dict(poller.poll(timeout=100))  # 100ms超时
                if self.state_sub_socket in events:
                    state = self.state_sub_socket.recv_json()
                    self._update_sim_state(state)
                    
                    # 将关键状态放入队列（非阻塞）
                    if not self.status_queue.full():
                        self.status_queue.put({
                            'q': self.q.copy(),
                            'v': self.v.copy(),
                            'a': self.a.copy(),
                            'tau': self.joint_torque.copy(),
                            'wheel_force': self.wheel_force.copy()
                        })
            except zmq.ZMQError as e:
                print(f"[ZMQ ERROR] {e}")
                break
    def _update_sim_state(self, state):
        """更新来自仿真的最新状态。"""
        with self._state_lock:
            self.joint_pos = np.asarray(state['joint_pos']).flatten()
            self.joint_vel = np.asarray(state['joint_vel']).flatten()
            self.joint_torque = np.array(state['joint_torque']).flatten()
            self.body_pos = np.asarray(state['body_pos']).flatten()
            self.body_quat = np.asarray(state['body_quat']).flatten()
            self.body_linvel = np.asarray(state['body_linvel']).flatten()
            self.body_angvel = np.asarray(state['body_angvel']).flatten()
            self.body_acc   = np.asarray(state['imu']['acc']).flatten()
            self.body_acc_w = np.asarray(state['imu']['gyro']).flatten()
            self.wheel_force = np.array(state['wheel_force']).flatten()
            self.get_current_q_v()    #将q v a更新
            self.robot.set_robot_state(self.q, self.v)
    # ------------------------------------------------------------------
    # 计算控制命令（核心）
    # ------------------------------------------------------------------
    def get_current_q_v(self):
        # -------- 1) 把模拟器状态写入本地缓存 --------
        # --- free‑flyer ---
        self.q[:3]  = self.body_pos                  # x y z
        self.q[3:7] = self.body_quat[[1, 2, 3, 0]]  # 正确转换顺序 wxyz → xyzw
        self.v[:6]  = np.hstack([self.body_linvel,    # vx vy vz
                             self.body_angvel]) # wx wy wz
        self.a[:6]  = np.hstack([self.body_acc,    # ax ay az
                              self.body_acc_w])
        # --- 关节位置 --------------------------------------------------
        # --- 关节位置 --------------------------------------------------
        for i in range(4):
            q_start = 7 + i*3
            joint_start = i*3
            self.q[q_start : q_start+3] = self.joint_pos[joint_start : joint_start+3]

        # --- 关节速度 --------------------------------------------------
        for i in range(4):
            v_start = 6 + i*3
            vel_start = i*3
            self.v[v_start : v_start+3] = self.joint_vel[vel_start : vel_start+3]
        # # LF
        # 关节顺序表 (含浮动基座):
        # Index 0: root_joint
        # Index 1: LF_hip_joint
        # Index 2: LF_knee_joint
        # Index 3: LF_wheel_joint
        # Index 4: LR_hip_joint
        # Index 5: LR_knee_joint
        # Index 6: LR_wheel_joint
        # Index 7: RF_hip_joint
        # Index 8: RF_knee_joint
        # Index 9: RF_wheel_joint
        # Index 10: RR_hip_joint
        # Index 11: RR_knee_joint
        # Index 12: RR_wheel_joint
        # 同步到动力学模型
        # self.robot.set_robot_state(q, v)
        # print(f"self.robot.inverse_dynamics:{self.robot.inverse_dynamics(q, v, a)[6:]}\n")   #这个也没问题！！
        # return q,v
    # ------------------------------------------------------------------
    # ZMQ 帮手
    # ------------------------------------------------------------------
    def send_control_command(self, cmd):
        """发送控制指令（线程安全）"""
        try:
            with self._state_lock:
                self.cmd_pub_socket.send_json({
                    'control_cmd': cmd.tolist()
                })
        except zmq.ZMQError as e:
            print(f"[ERROR] Command send failed: {e}")


