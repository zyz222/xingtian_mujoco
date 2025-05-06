


from pathlib import Path
import numpy as np, pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

# 1. 载入模型
urdf   = Path("xingtian_sym_model/urdf/xingtian_fixed.urdf").resolve()
mesh_dirs = [str(urdf.parent.parent / "meshes")]

model, cmodel, vmodel = pin.buildModelFromUrdf(
    str(urdf), pin.JointModelFreeFlyer()  
)

# 2. 启动 Meshcat
viz = MeshcatVisualizer(model, cmodel, vmodel)
viz.initViewer(open=True)
viz.loadViewerModel()

# 3. ☆ 打开视觉几何体
viz.displayVisuals(True)          # ← 没这行就只看见坐标轴

# 4. 刷新位姿
q0 = pin.neutral(model)
viz.display(q0)
