import torch
import numpy as np
import pybullet as p
import articulate as art
import xml.etree.ElementTree as ET
from typing import Optional
from articulate.utils.bullet import *
from articulate.utils.rbdl import *
from utils import *
from qpsolvers import solve_qp
from config import paths


class PhysicsOptimizer:
    """
    物理优化器（对应 PIP: Yi et al., CVPR 2022, Sec. 3.2.3 Motion Tracking Optimizer）

    论文中的QP（Eq. 7）核心形式：
      变量：x = [q̈, λ, τ]
        - q̈ ∈ R^N：广义加速度（N = DoF = 3 + 3J，在本实现中用 self.model.qdot_size 表示）
        - λ ∈ R^(3*n_c)：接触点的地面反力/GRF（每个接触点3维力，n_c为接触点数）
        - τ ∈ R^N：广义力，其中 τ[:6] 为 root residual force（论文允许小的残余来补偿模型/真实差异）

      目标：min  E_PD(q̈) + E_reg(λ, τ)
        - E_PD = kθ * Eθ + kr * Er
          * Eθ = || q̈[3:] - θ̈_des ||^2          （Dual PD 的角度控制项，见论文 Sec.3.2.2）
          * Er = || J q̈ + Jdot qdot - r̈_des ||^2 （Dual PD 的位置/全局姿态控制项）
        - E_reg = kλ Eλ + k_res E_res + kτ Eτ      （Eq. 9）
          * Eλ  = Σ_c d_c ||λ_c||^2               （Signorini 相关的正则，d_c 为接触点高度）
          * E_res = ||τ[:6]||^2                   （root residual force 正则）
          * Eτ    = ||τ[6:]||^2                   （关节力矩正则）

      约束（Eq. 7）：
        - 动力学方程（equation of motion）:  M(q) q̈ + h(q,qdot) = J_c^T λ + τ
        - 摩擦锥（friction cone）:            λ ∈ F（实现中用线性化金字塔近似保持QP）
        - 无滑动/防穿透（no sliding）:        ṙ_j(q̈) ∈ C（实现中用离散时间的速度不等式近似）

    注意：本仓库在论文基础上还加入了若干“工程化”扩展（见代码中 [Repo扩展/非论文] 标注），
    例如接触力跨帧平滑约束、Unity侧显式脚底接触点输入等，用于提升实时稳定性/可控性。
    """
    test_contact_joints = ['LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2',
                           'SPINE3', 'LSHOULDER', 'RSHOULDER', 'HEAD',
                           'LELBOW', 'RELBOW', 'LHAND', 'RHAND', 'LFOOT', 'RFOOT'
                           ]  # 'LANKLE', 'RANKLE', 'NECK', 'LWRIST', 'RWRIST', 'LCLAVICLE', 'RCLAVICLE'

    def __init__(self, debug=False):
        mu = 0.6
        supp_poly_size = 0.2
        self.debug = debug
        self.model = RBDLModel(paths.physics_model_file, update_kinematics_by_hand=True)
        self.params = read_debug_param_values_from_json(paths.physics_parameter_file)

        # 角色质量/重力（用于把“几倍重力/体重”转换成牛顿）
        # - 优先使用 physics_parameters.json 里的 body_mass_kg（便于你手动校准单位/比例）
        # - 否则从 URDF 的 inertial.mass 汇总得到总质量
        self.gravity = float(self.params.get('gravity', 9.81))
        self.body_mass_kg = self._get_body_mass_kg(paths.physics_model_file)

        self.friction_constraint_matrix = np.array([[np.sqrt(2), -mu, 0],
                                                    [-np.sqrt(2), -mu, 0],
                                                    [0, -mu, np.sqrt(2)],
                                                    [0, -mu, -np.sqrt(2)]])
        self.support_polygon = np.array([[-supp_poly_size / 2,  0,  -supp_poly_size / 2],
                                         [ supp_poly_size / 2,  0,  -supp_poly_size / 2],
                                         [-supp_poly_size / 2,  0,   supp_poly_size / 2],
                                         [ supp_poly_size / 2,  0,   supp_poly_size / 2]])

        if debug:
            p.connect(p.GUI)
            p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
            self.id_robot = p.loadURDF(paths.physics_model_file, [0, 0, 0], useFixedBase=False, flags=p.URDF_MERGE_FIXED_LINKS)
            change_color(self.id_robot, [198 / 255, 238 / 255, 0, 1.0])
            p.loadURDF(paths.plane_file, [0, -0.881, 0.0], [-0.7071068, 0, 0, 0.7071068])
            load_debug_params_into_bullet_from_json(paths.physics_parameter_file)

        # states
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        # 上一帧接触点的力（用于平滑约束），key = (joint_name, point_label)，value = np.ndarray shape(3,)
        self.prev_contact_forces = {}
        # 上一帧左右脚各自的脚底合力（用于门控）；shape(3,)
        self.prev_left_foot_grf = None
        self.prev_right_foot_grf = None
        self.reset_states()

    @staticmethod
    def _calc_total_mass_from_urdf(urdf_file_path: str) -> float:
        """
        计算URDF文件中所有 link/inertial/mass 的总质量（kg）。
        """
        tree = ET.parse(urdf_file_path)
        root = tree.getroot()
        total_mass = 0.0
        for link in root.findall('link'):
            inertial = link.find('inertial')
            if inertial is None:
                continue
            mass_element = inertial.find('mass')
            if mass_element is None:
                continue
            mass_value = mass_element.get('value')
            if mass_value is None:
                continue
            try:
                total_mass += float(str(mass_value).strip().strip('"').strip("'"))
            except ValueError:
                continue
        return float(total_mass)

    def _get_body_mass_kg(self, urdf_file_path: str) -> float:
        m = float(self.params.get('body_mass_kg', 0.0))
        if m > 0.0:
            return m
        try:
            return self._calc_total_mass_from_urdf(urdf_file_path)
        except Exception:
            return 0.0

    def reset_states(self):
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.prev_contact_forces = {}
        self.prev_left_foot_grf = None
        self.prev_right_foot_grf = None

    def optimize_frame(self, pose, jvel, contact, acc):
        # === 论文符号对照 ===
        # q_ref: 论文中的“参考运动” q̂ / φ（网络预测的姿态），这里转换到RBDL表示的广义坐标 q_ref
        # v_ref: 论文中的关节速度 v（网络预测），用于构造 r_ref / r̈_des（Eq.(4)(5)）
        # contact: 论文中的 foot-ground contact probabilities c（2维：左右脚）；本仓库支持更丰富结构，见下
        q_ref = smpl_to_rbdl(pose, torch.zeros(3))[0]  # 神经网络预测的姿态（转换到RBDL广义坐标）
        v_ref = jvel if isinstance(jvel, np.ndarray) else jvel.numpy()  # 神经网络预测的关节速度
        # contact 既可能来自网络输出（torch.Tensor），也可能来自Unity传入（list/np.ndarray）
        if isinstance(contact, torch.Tensor):
            c_ref = contact.sigmoid().detach().cpu().numpy().astype(np.float32).reshape(-1)
        elif contact is None:
            c_ref = np.zeros((0,), dtype=np.float32)
        else:
            c_ref = np.asarray(contact, dtype=np.float32).reshape(-1)

        # [Repo扩展/非论文] contact 新结构（10 floats）：
        # [0]=左脚接触程度, [1]=右脚接触程度
        # [2:6]=左脚4点(0/1), [6:10]=右脚4点(0/1)
        # 兼容旧结构（2 floats）：仅左右脚接触程度
        foot_stable = np.zeros(2, dtype=np.float32)
        if c_ref.size >= 1:
            foot_stable[0] = float(c_ref[0])
        if c_ref.size >= 2:
            foot_stable[1] = float(c_ref[1])
        foot_point_flags = c_ref[2:10].copy() if c_ref.size >= 10 else None

        a_ref = acc.numpy() if isinstance(acc, torch.Tensor) else np.array(acc)   # IMU传感器测量的加速度
        q = self.q
        qdot = self.qdot

        if q is None:
            self.q = q_ref
            # 保持返回结构稳定：(pose_opt, tran_opt, labeled_grf, tau)
            # - pose_opt: 与输入 pose 类型一致（torch/numpy）
            # - tran_opt: torch.Tensor shape(3,)
            # - labeled_grf: dict，首帧无接触信息时为空
            # - tau: 广义力 τ，shape(self.model.qdot_size,)；首帧返回全0
            return pose, torch.zeros(3), {}, np.zeros(self.model.qdot_size, dtype=np.float64)
        print('optimize frame')
        # === Contact Point Determination（论文 Sec.3.2.3）===
        # 论文做法：先判定哪些关节与地面接触，再在每个接触关节处画 L×L 方形取4个顶点作为接触点（facet contact更稳定）。
        # 本实现的“support_polygon + pos”就是这一4点近似（L≈20cm）。
        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))
        Js = [np.empty((0, self.model.qdot_size))]
        collision_points, collision_joints = [], []

        foot_point_labels = ['front-left', 'front-right', 'back-left', 'back-right']  # 对应 self.support_polygon 的4个点

        # 记录每个脚底点（0..3）对应的 collision_points 索引（用于回传GRF时固定4点顺序）
        left_point_to_collision_idx = {}   # key: 0..3 -> collision_idx
        right_point_to_collision_idx = {}  # key: 0..3 -> collision_idx

        # 记录每个 collision point 的元信息，便于debug输出与回传结构化GRF
        collision_point_meta = []  # list[tuple[str joint_name, str point_label]]

        def _add_contact_point(joint_name: str, joint_id: int, pos: np.ndarray, point_i: int, point_map: Optional[dict]):
            """添加一个接触点到QP（collision_points + Jacobian），并可选记录脚底点映射。"""
            ps = self.support_polygon[point_i] + pos
            collision_points.append(ps)
            collision_point_meta.append((joint_name, foot_point_labels[point_i] if point_i < 4 else f'point-{point_i}'))
            pb = self.model.calc_base_to_body_coordinates(q, joint_id, ps)
            Js.append(self.model.calc_point_Jacobian(q, joint_id, pb))
            if point_map is not None and point_i < 4:
                point_map[point_i] = len(collision_points) - 1

        for joint_name in self.test_contact_joints:
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)

            # [Repo扩展/非论文] 新结构：脚底接触点由 foot_point_flags(0/1) 显式给定（每只脚4点）
            # - 论文原始做法：一旦判定“该脚关节接触”，就取该关节处 L×L 方形的 4 个顶点作为接触点（facet contact）。
            # - 本扩展做法：允许上层（Unity/其他模块）直接指定 4 点里哪些点参与接触，用于更精细/可控的足底接触建模。
            if joint_id in (Body.LFOOT, Body.RFOOT) and foot_point_flags is not None:
                is_left = (joint_id == Body.LFOOT)
                flags = foot_point_flags[:4] if is_left else foot_point_flags[4:]
                point_map = left_point_to_collision_idx if is_left else right_point_to_collision_idx
                selected = [i for i, f in enumerate(flags) if float(f) > 0.5]

                # 若显式点接触为空，但模型穿透地面，则仍然加约束防止穿模（保留原意）
                if selected or pos[1] <= self.params['floor_y']:
                    collision_joints.append(joint_name)
                    use_indices = selected if selected else [0, 1, 2, 3]
                    for point_i in use_indices:
                        _add_contact_point(joint_name, joint_id, pos, point_i, point_map)
                continue

            # 旧结构/其它关节：维持论文的“接触判定 + 固定4点facet contact”
            # 论文 Contact Determination（Page 5）要点：
            # - 脚：df < 0.5cm 或 (df < 3cm 且 接触概率 cf > 0.5)
            # - 非脚：dn < 0.5cm
            # 这里把“距离”用 pos[1] 与 floor_y 的差近似，3cm 对应 floor_y + 0.03，0.5cm 近似为 pos[1] <= floor_y。
            if (joint_id == Body.LFOOT and foot_stable[0] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03) or \
               (joint_id == Body.RFOOT and foot_stable[1] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03) or \
               (pos[1] <= self.params['floor_y']):
                collision_joints.append(joint_name)
                point_map = None
                if joint_id == Body.LFOOT:
                    point_map = left_point_to_collision_idx
                elif joint_id == Body.RFOOT:
                    point_map = right_point_to_collision_idx

                for point_i in range(4):
                    _add_contact_point(joint_name, joint_id, pos, point_i, point_map)
                    
        Js = np.vstack(Js)
        nc = len(collision_points)

        # === QP组装方式（对应论文 Eq.7 的 E_PD + E_reg）===
        # 这里把目标拆成三块最小二乘：
        #   - 对 q̈：  Σ ||A1 q̈ - b1||^2   -> E_PD
        #   - 对 λ：  Σ ||A2 λ - b2||^2    -> E_λ（Eq.9）
        #   - 对 τ：  Σ ||A3 τ - b3||^2    -> E_res, E_τ（Eq.9）
        #
        # 最终通过 P=AᵀA, q= -Aᵀb 转成标准QP：min 1/2 xᵀ P x + qᵀ x
        #
        # minimize   ||A1 * qddot - b1||^2     for A1, b1 in zip(As1, bs1)
        #            + ||A2 * lambda - b2||^2  for A2, b2 in zip(As2, bs2)
        #            + ||A3 * tau - b3||^2     for A3, b3 in zip(As3, bs3)
        # s.t.       G1 * qddot <= h1          for G1, h1 in zip(Gs1, hs1)
        #            G2 * lambda <= h2         for G2, h2 in zip(Gs2, hs2)
        #            G3 * tau <= h3            for G3, h3 in zip(Gs3, hs3)
        #            A_ * x = b_
        As1, bs1, As2, bs2, As3, bs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], \
                                       [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], \
                                       [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        A_, b_ = None, None

        # === Dual PD Controller Term E_PD ===
        # 1) Joint Rotation Controller（论文 Sec.3.2.2，公式给出 θ̈_des = kpθ( E(φ) - θ ) - kdθ θ̇）
        #    对应论文中 Eθ = || q̈[3:] - θ̈_des ||^2 （Sec.3.2.3, Page 6）
        #    这里用 angle_difference(...) 实现 (E(φ) - θ) 的角度差，并通过 A 选出 q̈ 的角速度部分。
        if True:
            A = np.hstack((np.zeros((self.model.qdot_size - 3, 3)), np.eye((self.model.qdot_size - 3))))
            b = self.params['kp_angular'] * art.math.angle_difference(q_ref[3:], q[3:]) - self.params['kd_angular'] * qdot[3:]
            As1.append(A)  # 72 * 75
            bs1.append(b)  # 72

        # joint position PD controller (using root velocity + ref pose to determine target joint position)
        if False:
            for joint_name in ['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                               'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                               'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHAND', 'RHAND']:
                joint_id = vars(Body)[joint_name]
                cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
                cur_pos = self.model.calc_body_position(q, joint_id)
                tar_pos = self.model.calc_body_position(q_ref, joint_id) - q_ref[:3] + q[:3] + v_ref[0] * self.params['delta_t']
                a_des = 3600 * (tar_pos - cur_pos) - 60 * cur_vel
                A = self.model.calc_point_Jacobian(q, joint_id)
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des
                As1.append(A * 2)
                bs1.append(b * 2)

        # 2) Joint Position Controller（论文 Sec.3.2.2, Eq.(4)(5)）
        #    论文：r_ref = r + T(v) Δt (Eq.4),  r̈_des = kp_r(r_ref - r) - kd_r ṙ (Eq.5)
        #    并在优化器里用 Er = || J q̈ + Jdot qdot - r̈_des ||^2 （Sec.3.2.3, Page 6）
        #
        #    实现细节：
        #    - A 取点Jacobian J
        #    - self.model.calc_point_acceleration(q, qdot, 0, ...) 在 q̈=0 时主要贡献 Jdot*qdot
        #      因此 b = -(Jdot*qdot) + r̈_des，使得 (J q̈ + Jdot qdot) ≈ r̈_des
        if True:
            for joint_name, v in zip(['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                                      'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                                      'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST'], v_ref[:22]):
                joint_id = vars(Body)[joint_name]
                if joint_id == Body.LFOOT or joint_id == Body.RFOOT: continue
                cur_vel = self.model.calc_point_velocity(q, qdot, joint_id) # ?
                a_des = self.params['kp_linear'] * v * self.params['delta_t'] - self.params['kd_linear'] * cur_vel #对应论文的
                A = self.model.calc_point_Jacobian(q, joint_id)
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des
                As1.append(A * self.params['coeff_jvel'])
                bs1.append(b * self.params['coeff_jvel'])

        # joint velocity (without Jdot * qdot term)
        if False:
            for joint_name, v in zip(
                    ['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                     'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                     'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHAND', 'RHAND'], v_ref):
                joint_id = vars(Body)[joint_name]
                A = self.model.calc_point_Jacobian(q, joint_id)
                b = (-self.model.calc_point_velocity(q, qdot, joint_id) + v) / self.params['delta_t']
                As1.append(A * 2)
                bs1.append(b * 2)

        # IMU acceleration
        if False:
            for joint_name, a in zip(['LWRIST', 'RWRIST', 'LKNEE', 'RKNEE', 'HEAD', 'ROOT'], a_ref):
                joint_id = vars(Body)[joint_name]
                offset = np.zeros(3)
                A = self.model.calc_point_Jacobian(q, joint_id, offset)
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(self.model.qdot_size), joint_id, offset) + a
                bs1.append(b * self.params['coeff_acc'])
                As1.append(A * self.params['coeff_acc'])

        # lambda size
        if False:
            As2.append(np.eye(nc * 3) * self.params['coeff_lambda_old'])
            bs2.append(np.zeros(nc * 3))

        # === Regularization Term E_reg（论文 Eq.(9)）===
        # Eλ：惩罚违反 Signorini 接触条件的力（论文写作：Eλ = Σ_c d_c ||λ_c||^2）
        # 直觉：接触点离地越高（d_c 越大），越不该产生接触力；接近地面/轻微穿透时允许力支撑。
        # 这里用 A = diag(d_c I_3) 实现：||A λ||^2 = Σ_c d_c^2 ||λ_c||^2（与论文形式等价到权重尺度上）
        if True:
            if nc != 0:
                A = [np.eye(3) * max(cp[1] - self.params['floor_y'], 0.005) for cp in collision_points]
                A = art.math.block_diagonal_matrix_np(A)
                As2.append(A * self.params['coeff_lambda'])
                bs2.append(np.zeros(nc * 3))

        # [Repo扩展/非论文] 左/右脚“合力跨帧平滑”（soft penalty，而非 hard 约束）
        # 目标项：E_F = ||F - F_prev||^2，其中 F = Σ_{i in foot} f_i （对左右脚分别计算）
        # 实现为最小二乘：|| (c * R) * λ - (c * F_prev) ||^2
        # - R 是 3 x (3*nc) 的线性求和矩阵，每行对应 x/y/z 分量的求和
        # - c 是可调系数（越大越“粘”上一帧，越平滑但可能影响快速换步）
        # if True:
        #     if nc > 0:
        #         # 统一门控逻辑：
        #         # - prev_F 太小：认为上一帧无有效接触，不加“跨帧合力”项/约束，避免抬脚时被硬拉
        #         # - stable 低于阈值：认为该脚本帧不稳定/不接触，不加项（阈值为0时等价于不启用此门控）
        #         eps = float(self.params.get('contact_total_force_gate_eps', 1e-3))
        #         def _build_total_force_sum_matrix(foot_indices):
        #             """
        #             构造 F = R * λ 的线性求和矩阵 R（3 x (3*nc)）。
        #             其中 foot_indices 是 collision_points 中属于某只脚的点索引集合。
        #             """
        #             if foot_indices is None:
        #                 return None
        #             idxs = sorted({int(i) for i in foot_indices if 0 <= int(i) < nc})
        #             if len(idxs) == 0:
        #                 return None
        #             R = np.zeros((3, nc * 3), dtype=np.float64)
        #             for i in idxs:
        #                 R[0, i * 3 + 0] = 1.0
        #                 R[1, i * 3 + 1] = 1.0
        #                 R[2, i * 3 + 2] = 1.0
        #             return R

        #         def _should_apply_total_force_term(prev_F, foot_indices, stable: float) -> bool:
        #             if prev_F is None:
        #                 return False
        #             if foot_indices is None or len(foot_indices) == 0:
        #                 return False
        #             if np.linalg.norm(np.asarray(prev_F, dtype=np.float64).reshape(3)) <= eps:
        #                 return False
        #             return True

        #         # soft penalty（最小二乘项）
        #         total_force_coeff = float(self.params.get('contact_total_force_smooth_coeff', 0.0))
        #         if total_force_coeff > 0.0:
        #             def _add_foot_total_force_smooth_term(foot_indices, prev_F, stable: float):
        #                 if not _should_apply_total_force_term(prev_F, foot_indices, stable=stable):
        #                     return
        #                 R = _build_total_force_sum_matrix(foot_indices)
        #                 if R is None:
        #                     return
        #                 prev_F = np.asarray(prev_F, dtype=np.float64).reshape(3)
        #                 As2.append(R * total_force_coeff)
        #                 bs2.append(prev_F * total_force_coeff)

        #             left_indices = sorted(left_point_to_collision_idx.values())
        #             right_indices = sorted(right_point_to_collision_idx.values())
        #             _add_foot_total_force_smooth_term(left_indices, self.prev_left_foot_grf, stable=float(foot_stable[0]))
        #             _add_foot_total_force_smooth_term(right_indices, self.prev_right_foot_grf, stable=float(foot_stable[1]))
        # [Repo扩展/非论文] 基于“重心投影”的左右脚法向力(Fy)分配先验（soft penalty）
        # 你提到的诉求：双脚8点都触地时，希望左右脚承重大小能随重心偏移而变化。
        # 现有实现只靠动力学平衡 + 正则，双支撑时 λ 分配往往冗余，容易趋向均分/数值偏置。
        #
        # 这里采用一个保持QP线性的“比例先验”：
        #   设 Fy_L = Σ_{i∈L} Fy_i, Fy_R = Σ_{i∈R} Fy_i
        #   用 CoM 在左右脚中心连线 (xz 平面) 上的投影得到 α∈[0,1]（α=0靠左脚，α=1靠右脚）
        #   期望比例 Fy_R / (Fy_L + Fy_R) ≈ α
        #   等价线性形式： (1-α) Fy_R - α Fy_L ≈ 0
        if True:
            if nc > 0:
                coeff = float(self.params.get('com_load_balance_coeff', 0.0))
                if coeff > 0.0:
                    stable_min = float(self.params.get('com_load_balance_stable_min', 0.5))
                    # 需要左右脚都在接触集合里
                    left_indices = sorted(left_point_to_collision_idx.values())
                    right_indices = sorted(right_point_to_collision_idx.values())
                    if len(left_indices) > 0 and len(right_indices) > 0 and float(foot_stable[0]) >= stable_min and float(foot_stable[1]) >= stable_min:
                        # 估计左右脚中心（用当前参与QP的接触点平均；8点全接触时就是各4点均值）
                        pL = np.mean(np.asarray([collision_points[i] for i in left_indices], dtype=np.float64), axis=0)
                        pR = np.mean(np.asarray([collision_points[i] for i in right_indices], dtype=np.float64), axis=0)

                        # 估计CoM：直接使用 RBDLModel 的 CalcCenterOfMass 封装
                        # 注意：不同 pyrbdl 版本可能没有该接口/签名不同，如遇异常则退化为 root 位置。
                        try:
                            _, com = self.model.calc_center_of_mass_position(q, qdot)
                        except Exception:
                            com = np.asarray(q[:3], dtype=np.float64).reshape(3)

                        # 在xz平面做投影比例
                        d = (pR[[0, 2]] - pL[[0, 2]]).astype(np.float64)
                        denom = float(np.dot(d, d))
                        if denom > 1e-10:
                            t = float(np.dot((com[[0, 2]] - pL[[0, 2]]).astype(np.float64), d) / denom)
                            alpha = float(np.clip(t, 0.0, 1.0))

                            row = np.zeros((nc * 3,), dtype=np.float64)
                            # 取各点的 Fy 分量：lambda 排列为 [Fx0,Fy0,Fz0,...]，Fy 在 1::3
                            for i in right_indices:
                                row[i * 3 + 1] += (1.0 - alpha)
                            for i in left_indices:
                                row[i * 3 + 1] += (-alpha)

                            As2.append(row.reshape(1, -1) * coeff)
                            bs2.append(np.zeros((1,), dtype=np.float64))

        # E_res 与 E_τ：对 τ 的L2正则（论文 Eq.(9)：Eres = ||τ[:6]||^2, Eτ = ||τ[6:]||^2）
        # 说明：论文权重默认 k_res=0.1, k_τ=0.01；这里对应 physics_parameters.json 中 coeff_virtual/coeff_tau。
        # [Repo扩展/非论文] 这里对 residual force 额外乘了 3.16（经验尺度因子，用于数值/单位匹配与稳定性调参）。
        if True:
            As3.append(art.math.block_diagonal_matrix_np([
                np.eye(6) * self.params['coeff_virtual']* 3.16, # 对应论文中的残余力约束
                np.eye(self.model.qdot_size - 6) * self.params['coeff_tau']#关节扭矩约束
            ]))
            bs3.append(np.zeros(self.model.qdot_size))

        # [Repo扩展/非论文] 约束虚拟力（tau[:3]）的“模长”上限（只处理前三维）
        # 说明：
        # - 真实的 L2 范数约束 ||tau[:3]||_2 <= r 属于二阶锥约束（SOCP），普通QP求解器不支持。
        # - 这里用保守的线性近似：对每个分量施加盒约束 |tau_i| <= r/sqrt(3)，从而保证 ||tau[:3]||_2 <= r。
        # - 若你希望更“贴近球形”的近似，需要引入更多方向的线性面或切换SOCP求解器。
        if True:
            vf_norm_max = float(self.params.get('virtual_force_norm_max', 0.0))
            if vf_norm_max > 0.0:
                bound = vf_norm_max / np.sqrt(3.0)
                rows = []
                rhs = []
                for i in range(3):
                    row = np.zeros(self.model.qdot_size, dtype=np.float64)
                    row[i] = 1.0
                    rows.append(row)
                    rhs.append(bound)
                    rows.append(-row)
                    rhs.append(bound)
                Gs3.append(np.vstack(rows))
                hs3.append(np.asarray(rhs, dtype=np.float64))

        # === Sliding / Anti-penetration Constraints（论文 Eq.7: ṙ_j(q̈) ∈ C, Page 6）===
        # 论文：对“接触关节”的速度施加边界，水平滑动速度 < σ，同时防止向下穿透。
        # 实现：使用离散时间近似 v_{t+1} ≈ v_t + Δt * J * q̈，将速度范围改写成 q̈ 的线性不等式。
        # 这里先对非脚关节（test_contact_joints[:-2]）在低于地面时施加较宽松的盒约束。
        if True:
            for joint_name in self.test_contact_joints[:-2]:
                joint_id = vars(Body)[joint_name]
                pos = self.model.calc_body_position(q, joint_id)
                if pos[1] <= self.params['floor_y']:
                    J = self.model.calc_point_Jacobian(q, joint_id)
                    v = self.model.calc_point_velocity(q, qdot, joint_id)
                    Gs1.append(-self.params['delta_t'] * J)
                    hs1.append(v - [-1e-1, 0, -1e-1])
                    Gs1.append(self.params['delta_t'] * J)
                    hs1.append(-v + [1e-1, 1e2, 1e-1])

        # 对脚关节的无滑动约束：用 stable（论文中的 contact probability c，经 sigmoid 后 ∈ (0,1)）自适应阈值。
        # stable 越大 -> 越“确信在接触” -> th 越小（更严格的无滑动）；stable 越小 -> th 越大（更宽松）。
        # [Repo扩展/非论文] safe_stable 的裁剪用于避免 log(0)/数值爆炸（工程保护，不影响论文主思想）。
        if True:
            for joint_name, stable in zip(['LFOOT', 'RFOOT'], foot_stable):
                joint_id = vars(Body)[joint_name]
                pos = self.model.calc_body_position(q, joint_id)


                J = self.model.calc_point_Jacobian(q, joint_id)
                v = self.model.calc_point_velocity(q, qdot, joint_id)

                # 添加对stable值的保护，防止出现无穷大
                safe_stable = max(min(stable, 0.84999), 1e-6)  # 限制范围在[1e-6, 0.84999]

                th_raw = -np.log(safe_stable / 0.85)
                th = max(th_raw, 0.01)
                th_y = (self.params['floor_y'] - pos[1]) / self.params['delta_t']
                Gs1.append(-self.params['delta_t'] * J)
                hs1.append(v - [-th, th_y, -th])
                Gs1.append(self.params['delta_t'] * J)
                hs1.append(-v + [th, max(th, th_y) + 1e-6, th])

        # === Friction Cone Constraint（论文 Eq.7: λ ∈ F, Page 6）===
        # 论文指出摩擦锥可线性化以保持QP；这里用4个半空间拼成“金字塔”近似库仑摩擦锥（μ=0.6）。
        if True:
            if nc > 0:
                Gs2.append(art.math.block_diagonal_matrix_np([self.friction_constraint_matrix] * nc))
                hs2.append(np.zeros(nc * 4))

        # === [Repo扩展] Total GRF Y Upper Bound（硬约束）===
        # 约束“所有接触点地面反力”的法向（Y轴）合力上限：
        #   Σ_i Fy_i <= k * m * g
        # - k 通过 physics_parameters.json 的 max_total_grf_y_multiple 配置（默认建议 3.0）
        # - m 为 self.body_mass_kg（优先读取 params['body_mass_kg']，否则从URDF惯性质量汇总）
        # - 这是线性不等式，可直接加到 λ 的 Gx<=h 里（G2 * lambda <= h2）
        if True:
            if nc > 0:
                k = float(self.params.get('max_total_grf_y_multiple', 0.0))
                if k > 0.0 and float(self.body_mass_kg) > 0.0 and float(self.gravity) > 0.0:
                    fy_max = k * float(self.body_mass_kg) * float(self.gravity)
                    # λ 排列为 [Fx0,Fy0,Fz0, Fx1,Fy1,Fz1, ...]，因此取每个点的 Fy 分量求和
                    row = np.zeros((nc * 3,), dtype=np.float64)
                    row[1::3] = 1.0
                    Gs2.append(row.reshape(1, -1))
                    hs2.append(np.asarray([fy_max], dtype=np.float64))
        # === Equation of Motion（论文 Eq.7: M q̈ + h = Jcᵀ λ + τ）===
        # 将动力学方程写成线性等式约束 A_ x = b_：
        #   [-M,  Jᵀ,  I] [q̈, λ, τ]ᵀ = h
        if True:
            M = self.model.calc_M(q)
            h = self.model.calc_h(q, qdot)
            A_ = np.hstack((-M, Js.T, np.eye(self.model.qdot_size)))
            b_ = h

        As1, bs1, As2, bs2, As3, bs3 = np.vstack(As1), np.concatenate(bs1), np.vstack(As2), np.concatenate(bs2), np.vstack(As3), np.concatenate(bs3)
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = np.vstack(Gs1), np.concatenate(hs1), np.vstack(Gs2), np.concatenate(hs2), np.vstack(Gs3), np.concatenate(hs3)
        G_ = art.math.block_diagonal_matrix_np([Gs1, Gs2, Gs3])
        h_ = np.concatenate((hs1, hs2, hs3))
        P_ = art.math.block_diagonal_matrix_np([np.dot(As1.T, As1), np.dot(As2.T, As2), np.dot(As3.T, As3)])
        q_ = np.concatenate((-np.dot(As1.T, bs1), -np.dot(As2.T, bs2), -np.dot(As3.T, bs3)))

        # fast solvers are less accurate/robust, and may fail
        init = self.last_x if len(self.last_x) == len(q_) else None
        
        # # 添加调试输出
        # if self.debug:
        #     print("Debugging QP matrices:")
        #     print(f"P_ shape: {P_.shape}, P_ dtype: {P_.dtype}")
        #     print(f"q_ shape: {q_.shape}, q_ dtype: {q_.dtype}")
        #     print(f"G_ shape: {G_.shape}, G_ dtype: {G_.dtype}")
        #     print(f"h_ shape: {h_.shape}, h_ dtype: {h_.dtype}")
        #     print(f"A_ shape: {A_.shape}, A_ dtype: {A_.dtype}")
        #     print(f"b_ shape: {b_.shape}, b_ dtype: {b_.dtype}")
        #
        #     # 检查是否有 NaN 或无穷大值
        #     print(f"P_ has NaN: {np.isnan(P_).any()}, P_ has Inf: {np.isinf(P_).any()}")
        #     print(f"q_ has NaN: {np.isnan(q_).any()}, q_ has Inf: {np.isinf(q_).any()}")
        #     print(f"G_ has NaN: {np.isnan(G_).any()}, G_ has Inf: {np.isinf(G_).any()}")
        #     print(f"h_ has NaN: {np.isnan(h_).any()}, h_ has Inf: {np.isinf(h_).any()}")
        #     print(f"A_ has NaN: {np.isnan(A_).any()}, A_ has Inf: {np.isinf(A_).any()}")
        #     print(f"b_ has NaN: {np.isnan(b_).any()}, b_ has Inf: {np.isinf(b_).any()}")
        #
        #     # 检查矩阵是否为负或零
        #     if P_.shape[0] > 0 and P_.shape[1] > 0:
        #         eigenvals = np.linalg.eigvals(P_)
        #         print(f"P_ min eigenvalue: {np.min(eigenvals)}, max eigenvalue: {np.max(eigenvals)}")
        #
        #     if init is not None:
        #         print(f"init shape: {init.shape}, init has NaN: {np.isnan(init).any()}, init has Inf: {np.isinf(init).any()}")
        
        x = solve_qp(P_, q_, G_, h_, A_, b_, solver='quadprog', initvals=init)

        if x is None or np.linalg.norm(x) > 10000:
            if self.debug:
                print("Using cvxopt solver as fallback")
            x = solve_qp(P_, q_, G_, h_, A_, b_, solver='cvxopt', initvals=init)

        qddot = x[:self.model.qdot_size]
        GRF = x[self.model.qdot_size:-self.model.qdot_size]
        tau = x[-self.model.qdot_size:]

        # 更新“上一帧接触力”状态：只保留当前帧仍在接触集合中的点
        # 如果某点不再接触（本帧未进入 collision_points），它将自动从字典中消失，从而不再受平滑约束
        if nc > 0:
            grf_components_for_state = GRF.reshape(-1, 3)
            new_prev = {}
            for i in range(min(nc, len(collision_point_meta))):
                key = collision_point_meta[i]
                new_prev[key] = grf_components_for_state[i].copy()
            self.prev_contact_forces = new_prev
            # 只统计脚底点（左右脚各4点）各自合力，忽略其它接触点
            left_indices = sorted(left_point_to_collision_idx.values())
            right_indices = sorted(right_point_to_collision_idx.values())
            self.prev_left_foot_grf = np.sum(grf_components_for_state[left_indices], axis=0).copy() if len(left_indices) > 0 else None
            self.prev_right_foot_grf = np.sum(grf_components_for_state[right_indices], axis=0).copy() if len(right_indices) > 0 else None
        else:
            self.prev_contact_forces = {}
            self.prev_left_foot_grf = None
            self.prev_right_foot_grf = None

        # 添加调试信息，打印 tau 和 GRF 的形状
        #                                                                                                            if self.debug:
            # 计算并输出GRF在所有轴上的合力
            # if nc > 0:
            #     grf_components = GRF.reshape(-1, 3)
            #     grf_total = np.sum(grf_components, axis=0)
            #     grf_magnitude = np.linalg.norm(grf_total)
            #     print(f"GRF total force - X: {grf_total[0]:.6f}, Y: {grf_total[1]:.6f}, Z: {grf_total[2]:.6f}")
            #     print(f"GRF total magnitude: {grf_magnitude:.6f}")

            #     # 输出左右脚各自的GRF
            #     left_foot_indices = [left_point_to_collision_idx[i] for i in range(4) if i in left_point_to_collision_idx]
            #     right_foot_indices = [right_point_to_collision_idx[i] for i in range(4) if i in right_point_to_collision_idx]

            #     if left_foot_indices:
            #         left_grf = grf_components[left_foot_indices]
            #         left_total = np.sum(left_grf, axis=0)
            #         print(f"LFOOT GRF - X: {left_total[0]:.6f}, Y: {left_total[1]:.6f}, Z: {left_total[2]:.6f}")
                
            #     if right_foot_indices:
            #         right_grf = grf_components[right_foot_indices]
            #         right_total = np.sum(right_grf, axis=0)
            #         print(f"RFOOT GRF - X: {right_total[0]:.6f}, Y: {right_total[1]:.6f}, Z: {right_total[2]:.6f}")

            #     # 打印每个接触点的力分量，并标注是哪个关节以及具体的点
            #     for i, point_force in enumerate(grf_components):
            #         if i < len(collision_point_meta):
            #             joint_name, point_label = collision_point_meta[i]
            #             print(f"GRF component [{joint_name} - {point_label}]: "
            #                   f"X: {point_force[0]:.6f}, Y: {point_force[1]:.6f}, Z: {point_force[2]:.6f}")
            #         else:
            #             print(f"GRF component [Unknown - point {i}]: "
            #                   f"X: {point_force[0]:.6f}, Y: {point_force[1]:.6f}, Z: {point_force[2]:.6f}")
        # === Dynamic State Updates（论文 Sec.3.2.4）===
        # 论文用有限差分更新：qdot_{t+1} = qdot_t + q̈ Δt,  q_{t+1} = q_t + qdot_{t+1} Δt
        qdot = qdot + qddot * self.params['delta_t']
        q = q + qdot * self.params['delta_t']
        # [Repo实现差异] 这里最终把 self.q 直接回贴到 q_ref（而非积分得到的 q），属于工程折中：
        # - 好处：姿态更贴近网络输出，减少积分漂移；
        # - 代价：动力学积分状态与输出姿态不完全一致（但 qdot/λ/τ 仍来自QP）。
        self.q = q
        self.qdot = qdot
        self.last_x = x

        if self.debug:
            # self.clock.tick(60)   # please install pygame
            set_pose(self.id_robot, q)
            #print("当前帧的全局姿态", q)
            self.params = read_debug_param_values_from_bullet()

            #if False:   # visualize GRF (no smoothing)
            # p.removeAllUserDebugItems()
            # for point, force in zip(collision_points, GRF.reshape(-1, 3)):
            #     p.addUserDebugLine(point, point + force * 1e-2, [1, 0, 0])

        pose_opt, tran_opt = rbdl_to_smpl(q)
        pose_opt = torch.from_numpy(pose_opt).float()[0]
        tran_opt = torch.from_numpy(tran_opt).float()[0]
        
        # Create labeled GRF data
        labeled_grf = {}
        grf_components = GRF.reshape(-1, 3) if nc > 0 else np.zeros((0, 3), dtype=np.float32)

        # 始终输出左右脚各4点（固定顺序），未参与QP的点补0，保证Unity侧解包长度稳定（8点*3=24 floats）
        left_grf_data = []
        for i, point_label in enumerate(foot_point_labels):
            idx = left_point_to_collision_idx.get(i, None)
            force = grf_components[idx].tolist() if idx is not None and idx < len(grf_components) else [0.0, 0.0, 0.0]
            left_grf_data.append({'point': point_label, 'force': force})
        labeled_grf['left_foot'] = left_grf_data

        right_grf_data = []
        for i, point_label in enumerate(foot_point_labels):
            idx = right_point_to_collision_idx.get(i, None)
            force = grf_components[idx].tolist() if idx is not None and idx < len(grf_components) else [0.0, 0.0, 0.0]
            right_grf_data.append({'point': point_label, 'force': force})
        labeled_grf['right_foot'] = right_grf_data
            
        # Add other contact points if any
        all_foot_indices = set(list(left_point_to_collision_idx.values()) + list(right_point_to_collision_idx.values()))
        other_indices = [i for i in range(len(grf_components)) if i not in all_foot_indices]
        if other_indices:
            other_grf_data = []
            for i, idx in enumerate(other_indices):
                if idx < len(collision_point_meta):
                    joint_name, point_label = collision_point_meta[idx]
                    other_grf_data.append({
                        'joint': joint_name,
                        'point': point_label,
                        'force': grf_components[idx].tolist()
                    })
            labeled_grf['other_contacts'] = other_grf_data

        # 返回广义力 τ（长度 = self.model.qdot_size = 75）
        # - τ[:6]：root residual force/torque（常被称为“虚拟力/残余力”）
        # - τ[6:]：各关节力矩（23个关节 * 3维），按 RBDL 顺序排列（每段3维对应该关节的XYZ分量）：
        #   τ[6:9]    = LHIP
        #   τ[9:12]   = LKNEE
        #   τ[12:15]  = LANKLE
        #   τ[15:18]  = LFOOT
        #   τ[18:21]  = RHIP
        #   τ[21:24]  = RKNEE
        #   τ[24:27]  = RANKLE
        #   τ[27:30]  = RFOOT
        #   τ[30:33]  = SPINE1
        #   τ[33:36]  = SPINE2
        #   τ[36:39]  = SPINE3
        #   τ[39:42]  = LCLAVICLE
        #   τ[42:45]  = LSHOULDER
        #   τ[45:48]  = LELBOW
        #   τ[48:51]  = LWRIST
        #   τ[51:54]  = LHAND
        #   τ[54:57]  = RCLAVICLE
        #   τ[57:60]  = RSHOULDER
        #   τ[60:63]  = RELBOW
        #   τ[63:66]  = RWRIST
        #   τ[66:69]  = RHAND
        #   τ[69:72]  = NECK
        #   τ[72:75]  = HEAD
        generalized_tau = tau

        return pose_opt, tran_opt, labeled_grf, generalized_tau
