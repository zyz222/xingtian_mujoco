# import mujoco
# from mujoco import viewer
# import numpy as np


# model = mujoco.MjModel.from_xml_path("/home/zyz/workspace/xingtian_mujoco/xingtian_sym_model/urdf/scene_terrain.xml")  # 使用自带的humanoid模型
# # model = mujoco.MjModel.from_xml_path("/home/zyz/workspace/xingtian_mujoco/xingtian_sym_model/b2w/scene.xml")
# data = mujoco.MjData(model)
# # 启动可视化

# viewer_handle = viewer.launch(model, data)

# print("Available sensors:", 
#       [model.sensor(i).name for i in range(model.nsensor)])


# while viewer_handle.is_running():

#     # 仿真步进
#     mujoco.mj_step(model, data)
    
#     # 更新 viewer
#     viewer_handle.sync()

# import mujoco
# from mujoco import viewer
# import numpy as np
# import time
# import signal
# import atexit
# import glfw

# class SensorViewer:
#     def __init__(self, model_path):
#         # 初始化MuJoCo
#         self.model = mujoco.MjModel.from_xml_path(model_path)
#         self.data = mujoco.MjData(self.model)
#         self._running = True
        
#         # 设置安全退出
#         signal.signal(signal.SIGINT, self._handle_signal)
#         atexit.register(self.cleanup)
        
#         # 打印传感器信息
#         self._print_sensor_info()

#     def _print_sensor_info(self):
#         """打印所有传感器信息"""
#         print("\n=== 可用传感器列表 ===")
#         print(f"传感器总数: {self.model.nsensor}")
        
#         for i in range(self.model.nsensor):
#             sensor = self.model.sensor(i)
#             print(f"{i}. {sensor.name} (类型: {self._get_sensor_type(sensor.type)})")
        
#         print("====================\n")

#     def _get_sensor_type(self, type_id):
#         """获取传感器类型名称"""
#         types = {
#             0: "无效传感器",
#             1: "触觉",
#             2: "加速度计",
#             3: "速度计",
#             4: "陀螺仪",
#             5: "力扭矩",
#             6: "力矩",
#             7: "位置",
#             8: "速度",
#             9: "加速度",
#             10: "执行器力",
#             11: "执行器扭矩",
#             # ...其他类型参考MuJoCo文档
#         }
#         return types.get(int(type_id), f"未知类型({type_id})")

#     def _handle_signal(self, signum, frame):
#         """处理中断信号"""
#         self._running = False

#     def cleanup(self):
#         """安全清理资源"""
#         try:
#             if glfw.init._initialized:
#                 glfw.terminate()
#         except:
#             pass
#         print("\n资源已清理")

#     def run(self):
#         """运行查看器"""
#         try:
#             with viewer.launch_passive(self.model, self.data) as v:
#                 print("仿真运行中... (按Ctrl+C退出)")
#                 while self._running:
#                     # 执行仿真步
#                     mujoco.mj_step(self.model, self.data)
                    
#                     # 打印传感器数据示例（前5个传感器）
#                     if self.model.nsensor > 0:
#                         sensor_data = self.data.sensordata[:min(5, self.model.nsensor)]
#                         print(f"\r当前传感器数据: {np.array2string(sensor_data, precision=3)}", end="")
                    
#                     # 同步查看器
#                     v.sync()
#                     time.sleep(0.005)
                    
#         except Exception as e:
#             print(f"\n错误发生: {e}")
#         finally:
#             self.cleanup()

# if __name__ == "__main__":
#     # 替换为您的模型路径
#     model_path = "/home/zyz/workspace/xingtian_mujoco/xingtian_sym_model/urdf/scene_terrain.xml"
    
#     # 创建并运行查看器
#     viewer_app = SensorViewer(model_path)
#     viewer_app.run()

import mujoco
import numpy as np

def calc_joint_zero_angle(model, data, joint_name):
    # 找到joint所属body
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    body_id = model.joint_bodyid[joint_id]
    
    # 子body的初始朝向（xmat是3x3旋转矩阵）
    body_xmat = data.xmat[body_id].reshape(3,3)
    
    # 取body的局部X轴（第一列）
    body_x_axis_world = body_xmat[:, 0]  # X方向在世界坐标系下的方向

    # 世界X轴
    world_x_axis = np.array([1, 0, 0])

    # 计算与世界水平X方向的夹角
    cos_theta = np.clip(np.dot(body_x_axis_world, world_x_axis), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# 使用示例
model = mujoco.MjModel.from_xml_path('/home/zyz/workspace/xingtian_mujoco/xingtian_sym_model/urdf/scene_terrain.xml')
data = mujoco.MjData(model)

# 执行一次仿真，让xmat填充
mujoco.mj_step(model, data)

joint_name = 'LF_hip_joint'
angle_deg = calc_joint_zero_angle(model, data, joint_name)

print(f"关节 {joint_name} 零位置与水平X方向夹角: {angle_deg:.2f}°")

