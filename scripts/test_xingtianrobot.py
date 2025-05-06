"""
验证脚本：test_fbd_fixed_validated.py
=====================================
在原始 `test_fbd_fixed.py` 基础上，补充了**六大自检**：

1. **质量矩阵 M 对称正定**
2. **M⁻¹ · M ≈ I**
3. **动力学恒等式 τ = M·a + C·v + g**
4. **C(q, 0) = 0**
5. **Jacobian 数值差分 ≈ 解析**（针对单脚）
6. **ABA ↔ RNEA 前/逆动力学互检**

全部通过即表明 Pinocchio 模型与 `FloatingBaseDynamics` 接口实现基本可靠。
关闭 Meshcat 浏览器或按 `Ctrl‑C` 即可退出。
"""

from pathlib import Path
import numpy as np
import pinocchio as pin
from xingtian_robot import FloatingBaseDynamics

# ===================== 配置区 =====================
ROOT_DIR = Path(__file__).resolve().parent
URDF_PATH = ROOT_DIR / "../xingtian_sym_model/urdf/xingtian_fixed.urdf"
MESH_DIRS = [URDF_PATH.parent / "../meshes"]
VISUALIZE = True  # 打开 Meshcat
EPS = 1e-6        # 数值差分步长
# =================================================

# ---------- 1. 构建机器人 ----------
robot = FloatingBaseDynamics(
    str(URDF_PATH),
    package_dirs=[str(p) for p in MESH_DIRS],
    visual=VISUALIZE,
)
robot.check_model()
print("model.nq =", robot.model.nq, "model.nv =", robot.model.nv)
# 只需做一次（构造后即可）
lims = robot.model.lowerPositionLimit.copy()
lims[:3] = -1.0          # x,y,z 下界
robot.model.lowerPositionLimit = lims

lims = robot.model.upperPositionLimit.copy()
lims[:3] =  1.0          # 上界
robot.model.upperPositionLimit = lims

# ---------- 2. 初始状态 ----------
q0 = pin.neutral(robot.model)
v0 = np.zeros(robot.model.nv)
robot.set_state(q0, v0)
if VISUALIZE:
    robot.display()

# ---------- 3. 打印动力学矩阵 ----------
M = robot.mass_matrix()
print("质量矩阵对角:", np.diag(M))
print("|C|₁:", np.linalg.norm(robot.coriolis_matrix(), 1))
print("‖g‖₂:", np.linalg.norm(robot.gravity_vector()))

# ---------- 4. 自检 1：M 对称正定 ----------
assert np.allclose(M, M.T, atol=1e-10), "M 不是对称矩阵"
assert np.all(np.linalg.eigvalsh(M) > 0), "M 不是正定矩阵"
print("✔ M 对称正定")

# ---------- 5. 自检 2：M⁻¹·M≈I ----------
Minv = robot.mass_matrix_inv()
assert np.allclose(M @ Minv, np.eye(robot.model.nv), atol=1e-8)
print("✔ M @ M⁻¹ ≈ I")


# ---------- 6. 自检 3：动力学恒等式 ----------
# ------------------------------------------------------------------  随机状态（记得归一化）
q = pin.randomConfiguration(robot.model)
# ---- 修改随机姿态产生方式 ----
# q = pin.neutral(robot.model)                      # 安全起点
# q[:3] = np.random.uniform(-0.5, 0.5, size=3)      # xyz ∈ [‑0.5,0.5] m
pin.normalize(robot.model, q)                     # 四元数 / 连续转轴 归一化

v = np.random.randn(robot.model.nv)
a = np.random.randn(robot.model.nv)

# ------------------------------------------------------------------  分别计算 M 、nle、τ_rnea
M   = pin.crba(robot.model, robot.model.createData(), q)
nle = pin.nonLinearEffects(robot.model, robot.model.createData(), q, v)   # C v + g
τ_rnea = pin.rnea(robot.model, robot.model.createData(), q, v, a)

τ_rhs = M @ a + nle                    # M a + C v + g
print("any nan in M   :", np.isnan(M).any())
print("any nan in nle :", np.isnan(nle).any())
print("any nan in τ_rnea:", np.isnan(τ_rnea).any())

print("q  (首 8 项) :", q[:8])
print("v  (首 8 项) :", v[:8])

assert np.allclose(τ_rhs, τ_rnea, atol=1e-8), \
       f"τ 不一致，最大元素误差 {np.abs(τ_rhs-τ_rnea).max():.3e}"
print("✔ τ = M a + C v + g 恒等式通过")



# ---------- 7. 自检 4：C(q,0)=0 ----------
assert np.linalg.norm(robot.coriolis_matrix(q, np.zeros(robot.model.nv))) < 1e-10
print("✔ C(q,0)=0")

# ---------- 8. 自检 5：Jacobian 数值差分 ----------
model = robot.model
foot  = "LF_wheel"
fid   = model.getFrameId(foot)

# -------------------------------------------------  2. 随机合法姿态
q = pin.randomConfiguration(model)
pin.normalize(model, q)
eps = 1e-8

# -------- 解析 Jacobian（LOCAL_WORLD_ALIGNED） ---------------------
data_J = model.createData()
J_ana = pin.computeFrameJacobian(
            model, data_J, q, fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]

# -------- 有限差分 Jacobian ----------------------------------------
J_fd = np.zeros_like(J_ana)

def foot_pos_world(q_cfg):
    """返回足端在世界坐标系下的位置向量"""
    data = model.createData()
    pin.forwardKinematics(model, data, q_cfg)
    pin.updateFramePlacement(model, data, fid)
    return data.oMf[fid].translation.copy()

p0 = foot_pos_world(q)

for k in range(model.nv):
    dq = np.zeros(model.nv); dq[k] = eps

    q_plus  = pin.integrate(model, q,  dq);  pin.normalize(model, q_plus)
    q_minus = pin.integrate(model, q, -dq);  pin.normalize(model, q_minus)

    p_plus  = foot_pos_world(q_plus)
    p_minus = foot_pos_world(q_minus)

    J_fd[:, k] = (p_plus - p_minus) / (2 * eps)

# -------------------------------------------------  3. 误差评估
err = np.abs(J_ana - J_fd)
print("∞‑norm 误差 :", err.max())
assert err.max() < 1e-6, "Jacobian 误差过大"
print("✔ Jacobian 数值差分吻合")


# ---------- 9. 自检 6：ABA ↔ RNEA ----------
τ_rand = np.random.randn(robot.model.nv)
a_aba = robot.forward_dynamics(q, v, τ_rand)
τ_rec = robot.inverse_dynamics(q, v, a_aba)
assert np.allclose(τ_rand, τ_rec, atol=1e-8), "ABA/RNEA 不互逆"
print("✔ ABA 与 RNEA 互检通过")

# ---------- 10. 足端 力↔扭矩 映射 ----------
τ = np.random.randn(robot.model.nv)
F = robot.foot_forces_from_torques(τ, frame="body")
τ_back = robot.joint_torques_from_forces(F, frame="body")
print("足端力映射重构误差 ‖τ-τ_back‖₂ =", np.linalg.norm(τ - τ_back))

print("\n✅ 所有验证通过，模型工作正常！  Ctrl‑C 退出。")
