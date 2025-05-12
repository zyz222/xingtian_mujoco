import time
import queue
import threading
import numpy as np
from xingtian_controller import MujocoController
from INTERFACE.keyboard import KeyboardController
from INTERFACE.trajectory_body_generator import TrajectoryGenerator
from FSM.FSM import StateMachine

class MainController:
    def __init__(self):
        # 线程间通信队列
        self.command_queue = queue.Queue()
        self.status_queue = queue.Queue(maxsize=1)  # 只保留最新状态
        
        # 控制器实例
        self.ctrl = MujocoController(self.status_queue)
        self.keyboard_ctrl = KeyboardController(self.command_queue)
        self.traj_gen = TrajectoryGenerator()

        self.state_machine = StateMachine(self.ctrl.robot)     #传入动力学实例
        
        # 线程控制
        self.stop_event = threading.Event()
        self.last_fsm_time = time.time()
        self.fsm_torque_command = np.zeros(12)
        self.state_cmd = "passive"
        self.desired_velocity = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    def start(self):
        print("[INFO] Starting MainController...")
        try:
            # 启动子模块
            if not self.ctrl.start():
                raise RuntimeError("Failed to start MujocoController")
            self.keyboard_ctrl.start(self.stop_event)
            
            # 启动控制线程
            self.control_thread = threading.Thread(
                target=self._control_loop,
                daemon=True
            )
            self.control_thread.start()
            
            print("[INFO] Controller started successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Startup failed: {e}")
            self.stop()
            return False



    def stop(self):
        
        self.stop_event.set()
        self.ctrl.stop()
        self.keyboard_ctrl.stop()
        
        if self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)
        print("[INFO] Stopping MainController...")
    def _control_loop(self):
        """主控制循环"""
        try:
            while not self.stop_event.is_set():
                # 获取键盘指令（非阻塞）
                if not self.command_queue.empty():
                    self.state_cmd, self.desired_velocity = self.command_queue.get()
                    
                    if self.state_cmd == "quit":
                        break
                    # 更新运动模式
                    if self.state_cmd != self.state_machine.current_state.__class__.__name__.lower():
                        print(f"[INFO] Changing state to: {self.state_cmd}")
                        self.state_machine.change_state(self.state_cmd, self.ctrl.q, self.ctrl.v,self.ctrl.a, self.ctrl.joint_torque, self.ctrl.wheel_force)
                    else:
                        print(f"[INFO] State is already {self.state_cmd}, no change required")

                print(f"[INFO] Received command:, {self.state_cmd} \n")
                # 获取当前状态并更新轨迹
                q = self.ctrl.q
                v = self.ctrl.v
                a = self.ctrl.a
                tau = self.ctrl.joint_torque
                wheel_force = self.ctrl.wheel_force
                trajectory = self.traj_gen.generate(
                    q[:3], 
                    self.desired_velocity
                )
                # 获取步态相关参数以及足端状态
                # GAIT


                
                # self.state_machine.change_state(self.state_cmd,q,v,a,tau,wheel_force)
                
     

                # 更新状态指令
                torque = self.state_machine.update(
                    q, v, a,tau,
                    {"trajectory": trajectory}
                )
                # 发送控制指令
                if torque is not None:
                    self.fsm_torque_command = torque
                    self.ctrl.send_control_command(self.fsm_torque_command)
                
                time.sleep(0.001)
                
        except Exception as e:
            print(f"[ERROR] Control loop crashed: {e}")
        finally:
            self.stop()

if __name__ == "__main__":
    controller = MainController()
    try:
        if controller.start():
            while not controller.stop_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down by user request...")
    except Exception as e:
        print(f"\n[CRITICAL] Main loop error: {str(e)}")
    finally:
        controller.stop()