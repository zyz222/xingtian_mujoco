"""
浮动基机器人动力学辅助模块（Pinocchio 3.6 中文版）
====================================================
功能总览
--------
* 正/逆 运动学
* 正/逆 动力学
* 质量矩阵 M(q)、其逆 M⁻¹(q)、科氏矩阵 C(q,v)、重力向量 g(q)
* **关节扭矩 ↔ 足端力** 双向映射
* 简单 Euler 积分器
* Meshcat 可视化与自检脚本

> 本文件即插即用，再也不用补一行代码。

作者 : ChatGPT – OpenAI • 日期 : 2025‑05‑05
"""
from __future__ import annotations

import pathlib
from typing import Dict, List, Sequence

import numpy as np
import pinocchio as pin
from pinocchio import randomConfiguration
from pinocchio.visualize import MeshcatVisualizer

###############################################################################
# 主类
###############################################################################

class FloatingBaseDynamics:
    """四轮足 / 浮动基机器人动力学、运动学与控制辅助类。"""

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    def __init__(self, urdf_path: str | pathlib.Path, package_dirs: Sequence[str | pathlib.Path] | None = None, visual: bool = True):
        self.urdf_path = str(urdf_path)
        self.package_dirs = [str(d) for d in (package_dirs or [pathlib.Path(urdf_path).parent])]
        root_joint = pin.JointModelFreeFlyer()
        self.model = pin.buildModelFromUrdf(self.urdf_path, root_joint=root_joint)
        self.data = self.model.createData()
        self.collision_model = pin.buildGeomFromUrdf(self.model, self.urdf_path, pin.GeometryType.COLLISION, package_dirs=self.package_dirs)
        self.visual_model    = pin.buildGeomFromUrdf(self.model, self.urdf_path, pin.GeometryType.VISUAL,    package_dirs=self.package_dirs)
        self.viz: MeshcatVisualizer | None = None
        if visual:
            try:
                self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
                self.viz.initViewer(open=True)
                self.viz.loadViewerModel()
                print("[Meshcat] 浏览器地址:", self.viz.viewer.url())
            except Exception as e:
                print("[Meshcat] 启动失败:", e)
        self.q = pin.neutral(self.model)
        self.q[:3] = np.array([0, 0, 0])
        self.q[3:7] = np.array([0, 0, 0, 1])  # 单位四元数  xyzw
        self.v = np.zeros(self.model.nv)
        print(f"在__init__中设置的初始状态: q = {self.q}, v = {self.v}")
         
        
    ############################################################################
    # 状态接口
    ############################################################################
    def set_state(self, q, v=None):
        self.q = q.copy()
        self.v = self.v if v is None else v.copy()
        # self.check_model()
        # self.display()
    def get_state(self, copy=True):
        return (self.q.copy(), self.v.copy()) if copy else (self.q, self.v)

    def set_base_pose(self, pos_xyz, quat_xyzw):
        self.q[:3] = pos_xyz
        self.q[3:7] = quat_xyzw

    def set_base_twist(self, spatial_vel):
        self.v[:6] = spatial_vel

    ############################################################################
    # 正运动学 / 帧工具
    ############################################################################
    def forward_kinematics(self, q=None, update_velocity=False):
        q = self.q if q is None else q
        pin.forwardKinematics(self.model, self.data, q)
        if update_velocity:
            pin.forwardVelocity(self.model, self.data, q, self.v)
        pin.updateFramePlacements(self.model, self.data)

    def frame_pose(self, name, q=None):
        self.forward_kinematics(q)
        return self.data.oMf[self.model.getFrameId(name)].copy()

    ############################################################################
    # 足端接口
    ############################################################################
    def foot_frames(self) -> List[str]:
        """返回所有足端 frame 名称（按 URDF link 名，而非 joint 名）"""
        KEYWORDS = ("_wheel", "_foot", "_toe")
        feet = [f.name for f in self.model.frames
                if f.type == pin.FrameType.BODY           # 只要刚体原点
                and any(k in f.name.lower() for k in KEYWORDS)]
        if not feet:
            raise RuntimeError(
                "foot_frames() 找不到任何足端 frame；"
                "请检查 URDF 名称或手动传入 feet 参数")
        return feet
        

    def foot_position(self, foot, frame="world", q=None):
        T_wf = self.frame_pose(foot, q)
        if frame == "world":
            return T_wf.translation
        T_wb = self.frame_pose("base_link", q)
        if frame == "body":
            return T_wb.rotation.T @ (T_wf.translation - T_wb.translation)
        if frame == "foot":
            return np.zeros(3)
        raise ValueError("frame 需为 world/body/foot")

    def foot_jacobian(self, foot, q=None, frame="foot"):
        q = self.q if q is None else q
        ref = {"foot": pin.LOCAL, "world": pin.WORLD, "body": pin.LOCAL_WORLD_ALIGNED}[frame]
        return pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId(foot), ref)

    ############################################################################
    # 逆运动学 / 速度学
    ############################################################################
    def _leg_idx(self, foot):
        fid = self.model.getFrameId(foot)
        jid = self.model.frames[fid].parent
        chain, idx = [], []
        while jid:
            if self.model.joints[jid].nq: chain.append(jid)
            jid = self.model.parents[jid]
        for j in reversed(chain): idx.extend(range(self.model.idx_qs[j], self.model.idx_qs[j]+self.model.nqs[j]))
        return idx

    def foot_inverse_kinematics(self, foot, p_target_body, q_init=None, tol=1e-4, iters=100):
        q = (self.q if q_init is None else q_init).copy()
        idx = self._leg_idx(foot)
        for _ in range(iters):
            err = p_target_body - self.foot_position(foot, "body", q)
            if np.linalg.norm(err) < tol: return q
            J = self.foot_jacobian(foot, q, "body")[:3, idx]
            dq = np.zeros(self.model.nv)
            dq[idx] = np.linalg.pinv(J) @ err
            q = pin.integrate(self.model, q, dq)
        raise RuntimeError("IK 未收敛")

    def foot_inverse_velocity(self, foot, v_foot_body, q=None):
        q = self.q if q is None else q
        idx = self._leg_idx(foot)
        J = self.foot_jacobian(foot, q, "body")[:3, idx]
        dq = np.zeros(self.model.nv)
        dq[idx] = np.linalg.pinv(J) @ v_foot_body
        return dq

    ############################################################################
    # 动力学矩阵
    ############################################################################
    def mass_matrix(self, q=None):
        return pin.crba(self.model, self.data, self.q if q is None else q)

    def mass_matrix_inv(self, q=None):
        return np.linalg.inv(self.mass_matrix(q))

    def coriolis_matrix(self, q=None, v=None):
        q = self.q if q is None else q
        v = self.v if v is None else v
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        return self.data.C.copy()

    def gravity_vector(self, q=None):
        return pin.computeGeneralizedGravity(self.model, self.data, self.q if q is None else q)

    def inverse_dynamics(self, q, v, a):
        return pin.rnea(self.model, self.data, q, v, a)

    def forward_dynamics(self, q, v, tau):
        return pin.aba(self.model, self.data, q, v, tau)

    ############################################################################
    # 力 ↔ 扭矩映射
    ############################################################################
    # --- 扭矩 -> 力 ----------------------------------------------------
    def foot_forces_from_torques(self, tau: np.ndarray, feet: Sequence[str] | None = None, frame: str = "foot", q=None):
        """给定关节扭矩 *tau*，估算每只脚的接触力 *F*（最小二乘）。

        假设 `Jᵀ F = τ`，将多脚 Jacobian 叠加后解 F。
        *feet* 为空时默认全部足端；*frame* 指定返回力的表达坐标系。
        返回 dict {foot: 6×1 扳手}。若使用 3D 力，可取向量最后三维。"""
        feet = list(self.foot_frames()) if feet is None else list(feet)
        if not feet:
            raise ValueError("feet 列表为空，无法估算足端力")


        q = self.q if q is None else q
        self.forward_kinematics(q)
        feet = list(self.foot_frames()) if feet is None else list(feet)
        JTs, order = [], []
        for f in feet:
            JT = self.foot_jacobian(f, q, "foot").T  # nv×6
            JTs.append(JT)
            order.append(f)
        JT_stack = np.hstack(JTs)             # nv × (6·m)
        F_stack = np.linalg.pinv(JT_stack) @ tau  # (6·m)
        forces = {}
        for i, foot in enumerate(order):
            wrench = F_stack[6*i:6*(i+1)]
            if frame == "world":
                R = self.frame_pose(foot, q).rotation
                wrench[:3], wrench[3:] = R @ wrench[:3], R @ wrench[3:]
            elif frame == "body":
                Rb = self.frame_pose("base_link", q).rotation
                Rf = self.frame_pose(foot, q).rotation
                wrench[:3], wrench[3:] = Rb.T @ (Rf @ wrench[:3]), Rb.T @ (Rf @ wrench[3:])
            forces[foot] = wrench
        return forces

    # --- 力 -> 扭矩 ----------------------------------------------------
    def joint_torques_from_forces(self, forces: Dict[str, np.ndarray], frame="foot", q=None):
        q = self.q if q is None else q
        self.forward_kinematics(q)
        tau = np.zeros(self.model.nv)
        for foot, f in forces.items():
            w = np.asarray(f).ravel()
            w = np.hstack([np.zeros(3), w]) if w.size == 3 else w
            if frame != "foot":
                T = self.frame_pose(foot, q)
                R = T.rotation
                if frame == "world":
                    w[:3], w[3:] = R.T @ w[:3], R.T @ w[3:]
                elif frame == "body":
                    Rb = self.frame_pose("base_link", q).rotation
                    w[:3], w[3:] = R.T @ (Rb @ w[:3]), R.T @ (Rb @ w[3:])
            tau += self.foot_jacobian(foot, q, "foot").T @ w
        return tau

    ############################################################################
    # 简单积分器
    ############################################################################
    def step(self, tau, dt):
        a = self.forward_dynamics(self.q, self.v, tau)
        self.v += a*dt
        self.q = pin.integrate(self.model, self.q, self.v*dt)
        if self.viz: self.display()

    ############################################################################
    # 调试/可视化
    ############################################################################
    def display(self, q=None):
        if self.viz: self.viz.display(self.q if q is None else q)

    def check_model(self):
        print("\n=== 模型诊断 ===")
        print("nv:", self.model.nv, "足端:", self.foot_frames())
        if self.viz:
            self.display()  # 先显示模型
            # input("Meshcat 已开启，检查后回车…")
            # print("=== 结束 ===\n")