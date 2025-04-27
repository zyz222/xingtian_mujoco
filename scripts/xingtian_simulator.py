import mujoco
from mujoco import viewer
import numpy as np
import time
import zmq
import json
import threading
from collections import deque

class MujocoSimulator:
    def __init__(self, model_path, state_pub_port=5556, cmd_sub_port=5555):
        # 初始化MuJoCo模型和数据
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        initial_joint_angles = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0]
        for i in range(8):
            self.mj_data.qpos[7 + i] = initial_joint_angles[i]

        # 注意后面轮子的初始角一般设0
        for i in range(8, 12):
            self.mj_data.qpos[7 + i] = 0.0
        # 控制参数
        self.control_cmd = np.zeros(self.mj_model.nu)  # 控制指令
        self.sim_running = True  # 仿真运行标志
        self.sim_paused = False  # 仿真暂停标志
        
        # 通信设置
        self.state_pub_port = state_pub_port
        self.cmd_sub_port = cmd_sub_port
        self.cmd_queue = deque(maxlen=12)  # 控制指令队列
        
        # 初始化传感器索引
        # self._init_sensor_indices()
        
        # 初始化通信线程
        self._init_communication()
        self.last_flip_time = time.time()
        self.sign = 1
    def _init_sensor_indices(self):
        """初始化传感器名称到索引的映射"""
        # print("Available sensors:")
        # print(self.mj_data.sensordata)
        # for i in range(self.mj_model.nsensor):
        #     name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        #     print(f"{i}: {name} (value={self.mj_data.sensordata[i]})")
        self.sensor_indices = {}
        for i in range(self.mj_model.nsensor):
            sensor_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name:
                self.sensor_indices[sensor_name] = i

    def _init_communication(self):
        """初始化ZeroMQ通信"""
        context = zmq.Context()
        
        # 状态发布者
        self.state_pub_socket = context.socket(zmq.PUB)
        self.state_pub_socket.bind(f"tcp://*:{self.state_pub_port}")
        
        # 控制指令订阅者
        self.cmd_sub_socket = context.socket(zmq.SUB)
        self.cmd_sub_socket.bind(f"tcp://*:{self.cmd_sub_port}")
        self.cmd_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # 启动通信线程
        self.comm_thread = threading.Thread(target=self._communication_loop)
        self.comm_thread.daemon = True
        self.comm_thread.start()

    def _communication_loop(self):
        """通信线程主循环"""
        poller = zmq.Poller()
        poller.register(self.cmd_sub_socket, zmq.POLLIN)
        
        while self.sim_running:
            # 接收控制指令
            socks = dict(poller.poll(1))  # 1ms超时
            if self.cmd_sub_socket in socks:
                try:
                    msg = self.cmd_sub_socket.recv_json()
                    if 'control_cmd' in msg:
                        self.cmd_queue.append(np.array(msg['control_cmd']))
                except Exception as e:
                    print(f"Command receive error: {e}")
            
            # 短暂休眠避免占用太多CPU
            time.sleep(0.0001)

    def _get_sensor_data(self, sensor_name):
        """
        直接返回传感器的完整数据数组
        :param sensor_name: 传感器名称
        :return: numpy数组 (自动包含所有维度数据)
        """
        try:
            return self.mj_data.sensor(sensor_name).data.copy()
        except Exception as e:
            print(f"传感器 {sensor_name} 读取错误: {str(e)}")
            # 根据传感器类型返回适当维度的零数组
            if 'quat' in sensor_name:
                return np.zeros(4)
            elif any(x in sensor_name for x in ['gyro', 'acc', 'pos', 'vel', 'force']):
                return np.zeros(3)
            else:
                return np.zeros(1)
    def _update_robot_state(self):
        """更新机器人状态并发布"""
        # 打印所有可用传感器用于调试
        # if not hasattr(self, '_sensors_printed'):
        #     print("\n=== 可用传感器列表 ===")
        #     for i in range(self.mj_model.nsensor):
        #         name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        #         print(f"{i}: {name} = {self.mj_data.sensordata[i]}")
        #     self._sensors_printed = True
        # 关节位置传感器
        joint_pos = {
            'LF_hip': self._get_sensor_data('LF_hip_pos'),
            'LF_knee': self._get_sensor_data('LF_knee_pos'),
            'LR_hip': self._get_sensor_data('LR_hip_pos'),
            'LR_knee': self._get_sensor_data('LR_knee_pos'),
            'RR_hip': self._get_sensor_data('RR_hip_pos'),
            'RR_knee': self._get_sensor_data('RR_knee_pos'),
            'RF_hip': self._get_sensor_data('RF_hip_pos'),
            'RF_knee': self._get_sensor_data('RF_knee_pos'),
            'LF_wheel': self._get_sensor_data('LF_wheel_pos'),
            'LR_wheel': self._get_sensor_data('LR_wheel_pos'),
            'RR_wheel': self._get_sensor_data('RR_wheel_pos'),
            'RF_wheel': self._get_sensor_data('RF_wheel_pos')
        }

        # 关节速度传感器
        joint_vel = {
            'LF_hip': self._get_sensor_data('LF_hip_vel'),
            'LF_knee': self._get_sensor_data('LF_knee_vel'),
            'LR_hip': self._get_sensor_data('LR_hip_vel'),
            'LR_knee': self._get_sensor_data('LR_knee_vel'),
            'RR_hip': self._get_sensor_data('RR_hip_vel'),
            'RR_knee': self._get_sensor_data('RR_knee_vel'),
            'RF_hip': self._get_sensor_data('RF_hip_vel'),
            'RF_knee': self._get_sensor_data('RF_knee_vel'),
            'LF_wheel': self._get_sensor_data('LF_wheel_vel'),
            'LR_wheel': self._get_sensor_data('LR_wheel_vel'),
            'RR_wheel': self._get_sensor_data('RR_wheel_vel'),
            'RF_wheel': self._get_sensor_data('RF_wheel_vel')
        }

        # 关节扭矩传感器
        joint_torque = {
            'LF_hip': self._get_sensor_data('LF_hip_torque'),
            'LF_knee': self._get_sensor_data('LF_knee_torque'),
            'LR_hip': self._get_sensor_data('LR_hip_torque'),
            'LR_knee': self._get_sensor_data('LR_knee_torque'),
            'RR_hip': self._get_sensor_data('RR_hip_torque'),
            'RR_knee': self._get_sensor_data('RR_knee_torque'),
            'RF_hip': self._get_sensor_data('RF_hip_torque'),
            'RF_knee': self._get_sensor_data('RF_knee_torque'),
            'LF_wheel': self._get_sensor_data('LF_wheel_torque'),
            'LR_wheel': self._get_sensor_data('LR_wheel_torque'),
            'RR_wheel': self._get_sensor_data('RR_wheel_torque'),
            'RF_wheel': self._get_sensor_data('RF_wheel_torque')
        }

# IMU数据
        imu_data = {
            'quat': self._get_sensor_data('imu_quat'),  # 四元数(w,x,y,z)
            'gyro': self._get_sensor_data('imu_gyro'),  # 角速度(x,y,z)
            'acc': self._get_sensor_data('imu_acc'),    # 加速度(x,y,z)
            'pos': self._get_sensor_data('frame_pos'),  # 位置(x,y,z)
            'linvel': self._get_sensor_data('frame_vel') # 线速度(x,y,z)
        }
        
        # 轮子接触力 (3D向量)
        wheel_force = {
            'LF': self._get_sensor_data('LF_wheel_force'),
            'LR': self._get_sensor_data('LR_wheel_force'),
            'RR': self._get_sensor_data('RR_wheel_force'),
            'RF': self._get_sensor_data('RF_wheel_force')
        }
        # 构建状态字典
        state = {
            'joint_pos': [joint_pos[j].tolist() for j in [
                'LF_hip', 'LF_knee', 'LR_hip', 'LR_knee',
                'RR_hip', 'RR_knee', 'RF_hip', 'RF_knee'
            ]],
            'joint_vel': [joint_vel[j].tolist() for j in [
                'LF_hip', 'LF_knee', 'LR_hip', 'LR_knee',
                'RR_hip', 'RR_knee', 'RF_hip', 'RF_knee'
            ]],
            'joint_torque': [joint_torque[j].tolist() for j in [
                'LF_hip', 'LF_knee', 'LR_hip', 'LR_knee',
                'RR_hip', 'RR_knee', 'RF_hip', 'RF_knee'
            ]],
            'wheel_pos': [joint_pos[j].tolist() for j in [
                'LF_wheel', 'LR_wheel', 'RR_wheel', 'RF_wheel'
            ]],
            'wheel_vel': [joint_vel[j].tolist() for j in [
                'LF_wheel', 'LR_wheel', 'RR_wheel', 'RF_wheel'
            ]],
            'wheel_torque': [joint_torque[j].tolist() for j in [
                'LF_wheel', 'LR_wheel', 'RR_wheel', 'RF_wheel'
            ]],
            'wheel_force': [wf.tolist() for wf in [
                wheel_force['LF'],
                wheel_force['LR'],
                wheel_force['RR'],
                wheel_force['RF']
            ]],
            'imu': {
                'quat': imu_data['quat'].tolist(),
                'gyro': imu_data['gyro'].tolist(),
                'acc': imu_data['acc'].tolist(),
                'pos': imu_data['pos'].tolist(),
                'linvel': imu_data['linvel'].tolist()
            },
            'body_pos': self.mj_data.body("base_link").xpos.copy().tolist(),
            'body_quat': self.mj_data.body("base_link").xquat.copy().tolist(),
            'body_linvel': self.mj_data.body("base_link").cvel[:3].copy().tolist(),
            'body_angvel': self.mj_data.body("base_link").cvel[3:6].copy().tolist(),
            'timestamp': time.time()
        }

        # 发布状态
        try:
            self.state_pub_socket.send_json(state)
        except Exception as e:
            print(f"State publish error: {e}")

    def _process_control_commands(self):
        """处理接收到的控制指令"""
        # now = time.time()
        # if now - self.last_flip_time > 0.002:
        #     self.sign *= -1
        #     self.control_cmd = np.ones(12) * 0.5 * self.sign
        #     self.last_flip_time = now
            # now_time = time.time()
        while self.cmd_queue:
            cmd = self.cmd_queue.popleft()
            if len(cmd) == self.mj_model.nu:
                self.control_cmd = cmd
            else:
                print(f"Invalid command length: expected {self.mj_model.nu}, got {len(cmd)}")

    def run(self):
        """运行仿真主循环"""
        try:
            viewer_handle = viewer.launch_passive(self.mj_model, self.mj_data)
            last_step = time.time()
            
            while self.sim_running:
                if self.sim_paused:
                    time.sleep(0.01)
                    continue
                
                viewer_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True    # 显示约束
                viewer_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True        # 显示约束
                # 控制仿真频率 (500Hz)
                now = time.time()
                if now - last_step < 0.002:
                    continue
                # print(self.mj_data.sensordata)
                # 处理控制指令
                self._process_control_commands()
                
                # 应用控制指令
                self.mj_data.ctrl[:] = self.control_cmd
                
                # 执行仿真步
                mujoco.mj_step(self.mj_model, self.mj_data)
                
                # 更新并发布机器人状态
                self._update_robot_state()
                
                # 同步查看器
                viewer_handle.sync()
                last_step = now
                
        except Exception as e:
            print(f"Simulator error: {e}")
        finally:
            if 'viewer_handle' in locals():
                viewer_handle.close()
            self.sim_running = False
            self.comm_thread.join()

if __name__ == "__main__":
    try:
        # 创建仿真器实例
        sim = MujocoSimulator("/home/zyz/workspace/xingtian_mujoco/xingtian_sym_model/urdf/scene_terrain.xml")
        
        # 启动仿真
        sim.run()
        
    except Exception as e:
        print(f"Initialization error: {e}")