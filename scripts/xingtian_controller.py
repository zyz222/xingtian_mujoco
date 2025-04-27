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
class MujocoController:
    def __init__(self, state_sub_port=5556, cmd_pub_port=5555):
        self.context = zmq.Context()

        self.state_sub_socket = self.context.socket(zmq.SUB)
        self.state_sub_socket.connect(f"tcp://localhost:{state_sub_port}")
        self.state_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.cmd_pub_socket = self.context.socket(zmq.PUB)
        self.cmd_pub_socket.connect(f"tcp://localhost:{cmd_pub_port}")

        self.running = False

        self.state_thread = threading.Thread(target=self._state_and_control_loop)
        self.cmd_thread = threading.Thread(target=self._keyboard_command_loop)
        self.status_thread = threading.Thread(target=self._status_display_loop)

        self.state_thread.daemon = True
        self.cmd_thread.daemon = True
        self.status_thread.daemon = True

        self.mode_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.max_width = 14

        self.mode = 0  # 0: passive, 1: stand, 2: move, 3: idle
        self.user_ctrl_sign = 0.0  # forward/lateral torque adjustment

        self.last_flip_time = time.time()
        self.sign = 1

        self.joint_pos = np.zeros(8)
        self.joint_vel = np.zeros(8)
        self.wheel_pos = np.zeros(4)
        self.wheel_vel = np.zeros(4)
        self.body_pos = np.zeros(3)
        self.body_quat = np.zeros(4)
        self.body_linvel = np.zeros(3)
        self.body_angvel = np.zeros(3)
        self.body_acc = np.zeros(3)
        self.joint_torque = np.zeros(8)
        self.wheel_torque = np.zeros(4)
        self.wheel_force = np.zeros(12)
    def start(self):
        self.running = True
        self.state_thread.start()
        self.cmd_thread.start()
        # self.status_thread.start()
        print("Controller started")

    def stop(self):
        self.running = False
        time.sleep(0.1)
        print("Controller stopping...")

    def _state_and_control_loop(self):
        while self.running:
            try:
                state = self.get_state(timeout=10)
                if state is None:
                    torque_cmd = self._compute_control_command(None)
                else:
                    torque_cmd = self._compute_control_command(state)

                self.send_control_command(torque_cmd)

            except Exception as e:
                print(f"Control loop error: {e}")
                break

    def _keyboard_command_loop(self):
        print("==============================")
        print("Modes: [1] Stand, [2] Move, [3] Idle")
        print("Use [W/S] to increase/decrease forward torque")
        print("Use [A/D] to adjust lateral torque")
        print("Press [Q] to quit")
        print("==============================")
        print("Listening for keyboard input...")

        def get_key():
            if sys.platform == 'win32':
                return msvcrt.getch().decode('utf-8')
            else:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return key

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
                        self.user_ctrl_sign += 0.5
                    elif cmd == 's':
                        self.user_ctrl_sign -= 0.5
                    elif cmd == 'a':
                        self.user_ctrl_sign -= 0.5
                    elif cmd == 'd':
                        self.user_ctrl_sign += 0.5
                    elif cmd == 'q':
                        print("Exiting...")
                        self.stop()
                        sys.exit(0)
                    else:
                        print("Unknown key pressed.")
            except Exception as e:
                print(f"Keyboard input error: {e}")
                break

    def _status_display_loop(self):
        while self.running:
            try:
                self._clear_console()
                print("==============================")
                print(self._get_status())
                print("==============================")
                time.sleep(0.5)
            except Exception as e:
                print(f"Status display error: {e}")
                break

    def _compute_control_command(self, state):
        with self.mode_lock:
            mode = self.mode
            user_ctrl_sign = self.user_ctrl_sign

        if state:
            self.joint_pos = np.array(state['joint_pos']).flatten()
            self.joint_vel = np.array(state['joint_vel']).flatten()
            self.wheel_pos = np.array(state['wheel_pos']).flatten()
            self.wheel_vel = np.array(state['wheel_vel']).flatten()
            self.body_pos = np.array(state['body_pos']).flatten()
            self.body_quat = np.array(state['body_quat']).flatten()
            self.body_linvel = np.array(state['body_linvel']).flatten()
            self.body_angvel = np.array(state['body_angvel']).flatten()
            self.body_acc = np.array(state['imu']['acc']).flatten()
            self.joint_torque = np.array(state['joint_torque']).flatten()
            self.wheel_torque = np.array(state['wheel_torque']).flatten()
            self.wheel_force = np.array(state['wheel_force']).flatten()
        else:
            self.joint_pos = np.zeros(8)
            self.joint_vel = np.zeros(8)
            self.wheel_pos = np.zeros(4)
            self.wheel_vel = np.zeros(4)
            self.body_pos = np.zeros(3)
            self.body_quat = np.zeros(4)
            self.body_linvel = np.zeros(3)
            self.body_angvel = np.zeros(3)
            self.body_acc = np.zeros(3)
            self.joint_torque = np.zeros(8)
            self.wheel_torque = np.zeros(4)
            self.wheel_force = np.zeros(12)
        self.joint_pos[::2] -=2.18
        self.joint_pos[1::2] += 1.14
        # 初始化记录
        if not hasattr(self, 'interpolation_step'):
            self.interpolation_step = 0
            self.initial_pos = self.joint_pos.copy()
            self.last_mode = -1

        if mode == 1:  # test
            now = time.time()
            # if now - self.last_flip_time > 0.01:
            self.sign *= -1
            self.joint_torque = np.ones(8) * 0.5 * self.sign
            self.wheel_torque = np.ones(4) * 1.5 * self.sign
            self.last_flip_time = now
      
        elif mode == 2:  # stand_up
            # <!--髋关节与水平方向夹角-2.18  膝关节夹角1.14   -->  -2.18四条腿都往前伸长
            # 如果模式切换了，重新插值初始化
            if self.last_mode != 2:
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

            self.joint_torque = kp * (interpolated_target - self.joint_pos) - kd * self.joint_vel
            self.wheel_torque = np.zeros(4) * user_ctrl_sign

        elif mode == 3:  # wheel_model
            self.joint_torque = np.zeros(8)
            self.wheel_torque = np.zeros(4)

        elif mode == 0:  # troting
            self.joint_torque = np.zeros(8) 
            self.wheel_torque = np.zeros(4) 

        self.last_mode = mode
        total_torque = np.concatenate([self.joint_torque, self.wheel_torque])
        total_torque = np.clip(total_torque, -50.0, 50.0)
        return total_torque

    def get_state(self, timeout=1):
        poller = zmq.Poller()
        poller.register(self.state_sub_socket, zmq.POLLIN)
        socks = dict(poller.poll(timeout))
        if self.state_sub_socket in socks:                      #获取接收到的信息
            return self.state_sub_socket.recv_json()
        return None

    def send_control_command(self, cmd):
        if isinstance(cmd, np.ndarray):
            cmd = cmd.tolist()
        self.cmd_pub_socket.send_json({'control_cmd': cmd})

    def _get_status(self):
        mode_names = {0: "Passive", 1: "Stand", 2: "Move", 3: "Idle"}
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        with self.mode_lock:
            return f"{timestamp} | Mode: {mode_names.get(self.mode, 'Unknown')} | Forward Torque: {self.user_ctrl_sign:.2f} Nm"

    def _clear_console(self):
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')

    def __del__(self):
        try:
            self.stop()
            self.state_sub_socket.close()
            self.cmd_pub_socket.close()
            self.context.term()
        except:
            pass

if __name__ == "__main__":
    try:
        controller = MujocoController()
        controller.start()

        while controller.running:
            time.sleep(0.001)


    except KeyboardInterrupt:
        print("Keyboard Interrupt. Shutting down...")
    finally:
        controller.stop()
