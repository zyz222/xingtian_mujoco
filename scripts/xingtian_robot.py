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
from typing import Dict, List, Sequence,Optional

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
    def __init__(self, mjcf_path: str | pathlib.Path, package_dirs: Sequence[str | pathlib.Path] | None = None, visual: bool = True):
        if not pathlib.Path(mjcf_path).exists():
            raise FileNotFoundError(f"URDF路径 '{mjcf_path}' 不存在或无法访问。")
        
        self.mjcf_path = str(mjcf_path)
        self.package_dirs = [str(d) for d in (package_dirs or [pathlib.Path(mjcf_path).parent])]
        root_joint = pin.JointModelFreeFlyer()
        self.model = pin.buildModelFromUrdf(self.mjcf_path, root_joint=root_joint)
        print("关节顺序表 (含浮动基座):")
        for i in range(1, len(self.model.names)):  # 跳过world关节
            print(f"Index {i-1}: {self.model.names[i]}")

        # for f in self.model.frames:
        #     if f.name == "LF_wheel":
        #         print(f.name, f.type)
        self.data = self.model.createData()
        # mesh_loader = None
        # self.collision_model = pin.buildGeomFromMJCF(self.model, self.mjcf_path, pin.GeometryType.COLLISION, package_dirs)
        # self.visual_model = pin.buildGeomFromMJCF(self.model, self.mjcf_path, pin.GeometryType.VISUAL, package_dirs)
        # self.viz: MeshcatVisualizer | None = None
        # if visual:
        #     try:
        #         self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        #         self.viz.initViewer(open=True)
        #         self.viz.loadViewerModel()
        #         print("[Meshcat] 浏览器地址:", self.viz.viewer.url())
        #     except Exception as e:
        #         print("[Meshcat] 启动失败:", e)
        self.q = pin.neutral(self.model)
        self.q[:3] = np.array([0, 0, 0])
        self.q[3:7] = np.array([0, 0, 0, 1])  # 单位四元数  xyzw
        self.v = np.zeros(self.model.nv)
        # print(f"在__init__中设置的初始状态: q = {self.q}, v = {self.v}")
         
        
    ############################################################################
    # 状态接口
    ############################################################################
    def set_robot_state(self, q, v=None):
        self.q = q.copy()
        self.v = self.v if v is None else v.copy()
        # self.check_model()
        # self.display()
    def get_robot_state(self, copy=True):
        return (self.q.copy(), self.v.copy()) if copy else (self.q, self.v)


    ############################################################################
    # 正运动学 / 帧工具   
    ############################################################################
    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> None:
        """缓存正运动学结果。"""
        q = self.q if q is None else q
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
            
    # 返回坐标系的位置
    def frame_pose(self, name: str, q: Optional[np.ndarray] = None) -> pin.SE3:
        self.forward_kinematics(q)
        return self.data.oMf[self.model.getFrameId(name)].copy()      
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
        

    

    def foot_jacobian(self, foot, q=None, frame="foot"):
        q = self.q if q is None else q
        ref = {"foot": pin.LOCAL, "world": pin.WORLD, "body": pin.LOCAL_WORLD_ALIGNED}[frame]
        return pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId(foot), ref)
    
    def foot_jacobian_dot_v(self, foot: str, q: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None, frame: str = "foot") -> np.ndarray:
        """Compute Jdot * v for the specified foot frame.

        Steps:
        1. Compute joint Jacobians and their time variation.
        2. Retrieve the frame Jacobian time variation via `getFrameJacobianTimeVariation`.
        3. Multiply by joint velocity vector v to obtain the 6‑D spatial acceleration term.
        """
        q = self.q if q is None else q
        v = self.v if v is None else v
        ref = {"foot": pin.LOCAL, "world": pin.WORLD, "body": pin.LOCAL_WORLD_ALIGNED}[frame]

        # 1) compute J(q) and Jdot(q,v)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)

        # 2) get frame-level Jdot
        fid = self.model.getFrameId(foot)
        Jdot = pin.getFrameJacobianTimeVariation(self.model, self.data, fid, ref)

        # 3) Jdot * v
        return Jdot @ v
    ############################################################################
    # 逆运动学 / 速度学
    ############################################################################
    def _leg_idx(self, foot: str) -> List[int]:
        fid = self.model.getFrameId(foot)
        jid = self.model.frames[fid].parentJoint
        chain: List[int] = []
        while jid:
            joint = self.model.joints[jid]
            if joint.shortname() == "JointModelFreeFlyer":
                break
            if joint.nq:
                chain.append(jid)
            jid = self.model.parents[jid]
        idx: List[int] = []
        for j in reversed(chain):
            idx.extend(range(self.model.idx_qs[j], self.model.idx_qs[j] + self.model.nqs[j]))
        return idx
    def is_target_reachable(self, foot: str, p_target_body: np.ndarray, margin: float = 0.01) -> bool:
        """
        判断目标足端位置是否在工作空间内。
        :param foot: 足端 frame 名称
        :param p_target_body: 目标足端位置 (机体坐标系下)
        :param margin: 安全裕度，默认 1cm
        :return: True 可达，False 不可达
        """
        idx = self._leg_idx(foot)  
        link_lengths = []
        
        # 计算腿部各连杆长度
        for j in idx:
            joint_placement = self.model.jointPlacements[j]
            link_length = np.linalg.norm(joint_placement.translation)
            link_lengths.append(link_length)

   

        total_length = sum(link_lengths) + margin  # 加上安全裕度
        target_dist = np.linalg.norm(p_target_body)

        print(f"[Reachability Check] Foot: {foot}, Target Distance: {target_dist:.3f}, Max Reach: {total_length:.3f}")

        return target_dist <= total_length
    # *代表是分割符，代表是关键字参数
    def foot_inverse_kinematics(self, foot: str, p_target_body: np.ndarray, *, q_init: Optional[np.ndarray] = None, tol: float = 1e-3, max_iters: int = 40, regularization: float = 1e-4) -> np.ndarray:
        #初始化关节角度
        q = (self.q if q_init is None else q_init).copy()
        idx = self._leg_idx(foot)
        if self.is_target_reachable(foot, p_target_body):
            try:
                for i in range(max_iters):
                    err = p_target_body - self.foot_position(foot,"body", q)
                    # print("Current foot position:", self.foot_position(foot,"body", q))
                    # print("Target foot position:", p_target_body)
                    # print("Error norm:", np.linalg.norm(err))
                    # 如果误差小于容忍度，则结束
                    if np.linalg.norm(err) < tol:
                        print("IK Converged!")
                        return q
                    
                    J = self.foot_jacobian(foot, q=q, frame="body")[:3, idx]
                    
                    # 伪逆计算，加入正则化项
                    J_pseudo_inv = J.T @ np.linalg.inv(J @ J.T + regularization * np.eye(3))
                    # J_pseudo_inv = np.linalg.pinv(J + regularization * np.eye(J.shape[1]))  # Tikhonov Regularization
                    dq_leg = J_pseudo_inv @ err
                    
                    dq = np.zeros(self.model.nv)
                    dq[idx] = dq_leg
                    
                    q = pin.integrate(self.model, q, dq)          # 将关节速度增量进行积分到关节位置上
                    
                raise RuntimeError("IK did not converge in the given iterations.")
            except RuntimeError as e:
                print("IK Failed:", e)
        else:
            print("⚠️ 目标位置不可达，跳过逆解！")
        

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
    def mass_matrix(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """计算质量矩阵 M(q)。"""
        return pin.crba(self.model, self.data, self.q if q is None else q)

    def mass_matrix_inv(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """计算质量矩阵的逆 M⁻¹(q)。"""
        M = self.mass_matrix(q)
        return np.linalg.inv(M)

    def coriolis_matrix(self, q: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None) -> np.ndarray:
        """计算科氏矩阵 C(q,v)。"""
        q = self.q if q is None else q
        v = self.v if v is None else v
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        return self.data.C.copy()

    def gravity_vector(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """计算重力向量 g(q)。"""
        return pin.computeGeneralizedGravity(self.model, self.data, self.q if q is None else q)

    def nonlinear_effects(self, q: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None) -> np.ndarray:
        """计算非线性项 b(q,v) = C(q,v)·v + g(q)。"""
        q = self.q if q is None else q
        v = self.v if v is None else v
        return self.coriolis_matrix(q, v) @ v + self.gravity_vector(q)
    
    # ------------------------------------------------------------------
    # 正 / 逆动力学
    # ------------------------------------------------------------------
    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """计算逆动力学 (关节扭矩)。"""
        return pin.rnea(self.model, self.data, q, v, a)

    def forward_dynamics(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """计算正动力学 (关节加速度)。"""
        return pin.aba(self.model, self.data, q, v, tau)

    ############################################################################
    # 力 ↔ 扭矩映射
    ############################################################################
    # --- 扭矩 -> 力 ----------------------------------------------------
    def foot_forces_from_torques(self, tau: np.ndarray, feet: Sequence[str] | None = None, frame: str = "foot", q=None):
        """给定关节扭矩 *tau*，估算每只脚的接触力 *F*（最小二乘）。"""
        feet = list(self.foot_frames()) if feet is None else list(feet)
        if not feet:
            raise ValueError("feet 列表为空，无法估算足端力")

        q = self.q if q is None else q                                  # 获取当前关节角度
        self.forward_kinematics(q)                                       # 计算当前角度下的所有链路位姿。
        JTs, order = [], []
        for f in feet:      # 遍历每个足端
            JT = self.foot_jacobian(f, q, "foot").T  # nv × 6        6*18  雅可比矩阵     18*6
            JTs.append(JT)
            order.append(f)

        JT_stack = np.hstack(JTs)  # nv × (6 * m)      横向拼接   18*24的雅可比矩阵
        
        # 🚨 关键修改：去掉基座自由度，只保留关节自由度
        JT_stack_reduced = JT_stack[6:, :]  # 只保留关节相关的 Jacobian   去掉前6行
        tau_reduced = tau  # 只保留关节相关的扭矩

        # 最小二乘求解足端力
        F_stack = np.linalg.pinv(JT_stack_reduced) @ tau_reduced  # (6 * m, )

        # 解析各足端力
        forces = {}
        for i, foot in enumerate(order):
            wrench = F_stack[6 * i:6 * (i + 1)]     #按每六个元素划分。
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