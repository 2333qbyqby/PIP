import torch
import numpy as np
import pybullet as p
import articulate as art
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Tuple, List
from articulate.utils.bullet import *
from articulate.utils.rbdl import *
from utils import *
from qpsolvers import solve_qp
from config import paths
from q_recorder import record_q_frame


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
    例如 Unity 侧显式接触点输入、结构化GRF回传等，用于提升实时稳定性/可控性。
    """
    test_contact_joints = ['LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2',
                           'SPINE3', 'LSHOULDER', 'RSHOULDER', 'HEAD',
                           'LELBOW', 'RELBOW', 'LHAND', 'RHAND', 'LFOOT', 'RFOOT'
                           ]  # 'LANKLE', 'RANKLE', 'NECK', 'LWRIST', 'RWRIST', 'LCLAVICLE', 'RCLAVICLE'

    @staticmethod
    def _safe_unit_vector(v: Any, default: np.ndarray) -> np.ndarray:
        """
        尝试把输入 v 转成单位向量；失败则返回 default（也会被归一化）。
        """
        d = np.asarray(default, dtype=np.float64).reshape(3)
        dn = float(np.linalg.norm(d))
        if dn > 1e-12:
            d = d / dn
        try:
            x = np.asarray(v, dtype=np.float64).reshape(3)
            n = float(np.linalg.norm(x))
            if n > 1e-12:
                return x / n
        except Exception:
            pass
        return d

    @staticmethod
    def _make_tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        给定单位法向 n，构造一组正交切向基 (t1, t2)。
        """
        n = np.asarray(n, dtype=np.float64).reshape(3)
        # 选择一个不与 n 共线的参考向量
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(n, up))) > 0.9:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t1 = np.cross(up, n)
        t1n = float(np.linalg.norm(t1))
        if t1n < 1e-12:
            # 极端退化情况再换一次参考向量
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            t1 = np.cross(up, n)
            t1n = float(np.linalg.norm(t1))
        if t1n > 1e-12:
            t1 = t1 / t1n
        else:
            # 最后兜底：随便给一个
            t1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t2 = np.cross(n, t1)
        t2n = float(np.linalg.norm(t2))
        if t2n > 1e-12:
            t2 = t2 / t2n
        return t1, t2

    @staticmethod
    def _parse_contact_dict(contact: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        解析 contact dict：全关节 (c + p4) + 可选 surface plane (n + p0)。

        支持两类结构：
        - 方案A（推荐）：{ "order": [...], "c":[...], "p":[[...],[...],...], "n":[[...],...], "p0":[[...],...] }
        - 方案B：{ "joints": { "LHIP": {"c":0.2,"p":[0,0,0,0],"n":[0,1,0],"p0":[0,-0.87,0]}, ... } }

        返回：joint_name -> dict:
          - c: float (接触程度，用于无滑动阈值)
          - p: np.ndarray shape(4,) or None (四角点mask，决定哪些点加入QP)
          - n: np.ndarray shape(3,) or None (接触面法向，世界坐标；若不提供则走默认地面)
          - p0: np.ndarray shape(3,) or None (接触面上一点，世界坐标；若不提供则走默认地面)
        """
        contact_by_joint: Dict[str, Dict[str, Any]] = {}

        # 方案A：order + 数组
        if isinstance(contact.get("order"), list) and isinstance(contact.get("c"), list):
            order = contact.get("order") or []
            c_list = contact.get("c") or []
            p_list = contact.get("p", None)
            n_list = contact.get("n", None)
            p0_list = contact.get("p0", None)
            n = min(len(order), len(c_list))
            for i in range(n):
                jn = str(order[i])
                try:
                    c_val = float(c_list[i])
                except Exception:
                    c_val = 0.0

                p4 = None
                if isinstance(p_list, list) and i < len(p_list):
                    try:
                        p4 = np.asarray(p_list[i], dtype=np.float32).reshape(4)
                    except Exception:
                        p4 = None

                n_vec = None
                if isinstance(n_list, list) and i < len(n_list):
                    try:
                        n_vec = np.asarray(n_list[i], dtype=np.float64).reshape(3)
                    except Exception:
                        n_vec = None
                p0_vec = None
                if isinstance(p0_list, list) and i < len(p0_list):
                    try:
                        p0_vec = np.asarray(p0_list[i], dtype=np.float64).reshape(3)
                    except Exception:
                        p0_vec = None

                contact_by_joint[jn] = {"c": c_val, "p": p4, "n": n_vec, "p0": p0_vec}
            return contact_by_joint

        # 方案B：joints dict
        joints = contact.get("joints")
        if isinstance(joints, dict):
            for jn, payload in joints.items():
                c_val = 0.0
                p4 = None
                n_vec = None
                p0_vec = None
                if isinstance(payload, dict):
                    try:
                        c_val = float(payload.get("c", 0.0))
                    except Exception:
                        c_val = 0.0
                    if "p" in payload:
                        try:
                            p4 = np.asarray(payload.get("p"), dtype=np.float32).reshape(4)
                        except Exception:
                            p4 = None
                    if "n" in payload:
                        try:
                            n_vec = np.asarray(payload.get("n"), dtype=np.float64).reshape(3)
                        except Exception:
                            n_vec = None
                    if "p0" in payload:
                        try:
                            p0_vec = np.asarray(payload.get("p0"), dtype=np.float64).reshape(3)
                        except Exception:
                            p0_vec = None
                contact_by_joint[str(jn)] = {"c": c_val, "p": p4, "n": n_vec, "p0": p0_vec}
            return contact_by_joint

        # 兼容：若用户直接传 { "LHIP": {"c":..,"p":[..]} , ... }
        for k, v in contact.items():
            if k in ("v", "order", "c", "p", "joints"):
                continue
            if isinstance(v, dict) and ("c" in v or "p" in v):
                try:
                    c_val = float(v.get("c", 0.0))
                except Exception:
                    c_val = 0.0
                p4 = None
                if "p" in v:
                    try:
                        p4 = np.asarray(v.get("p"), dtype=np.float32).reshape(4)
                    except Exception:
                        p4 = None
                n_vec = None
                p0_vec = None
                if "n" in v:
                    try:
                        n_vec = np.asarray(v.get("n"), dtype=np.float64).reshape(3)
                    except Exception:
                        n_vec = None
                if "p0" in v:
                    try:
                        p0_vec = np.asarray(v.get("p0"), dtype=np.float64).reshape(3)
                    except Exception:
                        p0_vec = None
                contact_by_joint[str(k)] = {"c": c_val, "p": p4, "n": n_vec, "p0": p0_vec}

        return contact_by_joint

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
        self.reset_states()
        self.frame_idx = 0

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
        self.frame_idx = 0

    def optimize_frame(self, pose, jvel, contact, acc):
        frame_idx = int(self.frame_idx)
        self.frame_idx += 1
        # === 论文符号对照 ===
        # q_ref: 论文中的“参考运动” q̂ / φ（网络预测的姿态），这里转换到RBDL表示的广义坐标 q_ref
        # v_ref: 论文中的关节速度 v（网络预测），用于构造 r_ref / r̈_des（Eq.(4)(5)）
        # contact: 论文中的 foot-ground contact probabilities c（2维：左右脚）；本仓库支持更丰富结构，见下
        q_ref = smpl_to_rbdl(pose, torch.zeros(3))[0]  # 神经网络预测的姿态（转换到RBDL广义坐标）
        # 记录每一帧“Unity/上层送来的参考姿态”（这里对应 q_ref）
        try:
            record_q_frame(q_ref=q_ref, frame_idx=frame_idx)
        except Exception:
            # 记录失败不应影响主流程
            pass
        v_ref = jvel if isinstance(jvel, np.ndarray) else jvel.numpy()  # 神经网络预测的关节速度
        # contact 既可能来自网络输出（torch.Tensor），也可能来自Unity传入（list/np.ndarray），或新结构 dict/json
        contact_by_joint: Optional[Dict[str, Dict[str, Any]]] = None
        if isinstance(contact, torch.Tensor):
            c_ref = contact.sigmoid().detach().cpu().numpy().astype(np.float32).reshape(-1)
        elif contact is None:
            c_ref = np.zeros((0,), dtype=np.float32)
        elif isinstance(contact, dict):
            c_ref = np.zeros((0,), dtype=np.float32)
            contact_by_joint = self._parse_contact_dict(contact)
        else:
            c_ref = np.asarray(contact, dtype=np.float32).reshape(-1)

        # 已移除旧格式 contact=10个float（2+8：每脚4点mask）。
        # 兼容极简结构（2 floats）：仅左右脚接触程度（与论文一致，也用于网络输出）。
        if (not isinstance(contact, torch.Tensor)) and (not isinstance(contact, dict)) and (c_ref.size == 10):
            raise ValueError("已移除旧格式 contact=10个float（2+8）。请改用 dict/json 新结构。")

        foot_stable = np.zeros(2, dtype=np.float32)
        if c_ref.size >= 1:
            foot_stable[0] = float(c_ref[0])
        if c_ref.size >= 2:
            foot_stable[1] = float(c_ref[1])

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

        # 记录每个 collision point 的元信息，便于构造“无滑动/防穿透约束”与回传结构化GRF。
        # 每个元素是一个 dict，尽量保持字段清晰，避免隐式约定：
        # - joint_name: str
        # - joint_id: Body enum
        # - point_label: str（脚底4点标签或 point-i）
        # - pb: np.ndarray shape(3,)（接触点在 body 坐标系下的坐标，用于 calc_point_Jacobian/velocity）
        collision_point_meta = []

        # 对“无滑动约束”使用的接触程度（0~1）：joint_name -> c
        # - dict contact：使用每个关节自己的 c
        # - 非 dict contact：目前只支持左右脚（2 floats），因此这里只会包含 LFOOT/RFOOT
        contact_degree_by_joint = {}
        if contact_by_joint is not None:
            for jn in contact_by_joint:
                try:
                    contact_degree_by_joint[str(jn)] = float(contact_by_joint[jn].get("c", 0.0))
                except Exception:
                    contact_degree_by_joint[str(jn)] = 0.0
        else:
            # 仅 2 floats：左右脚接触程度
            contact_degree_by_joint["LFOOT"] = float(foot_stable[0])
            contact_degree_by_joint["RFOOT"] = float(foot_stable[1])

        def _add_contact_point(
            joint_name: str,
            joint_id: int,
            pos: np.ndarray,
            point_i: int,
            point_map: Optional[dict],
            surface_n: np.ndarray,
            surface_p0: np.ndarray,
        ):
            """
            添加一个接触点到QP（collision_points + Jacobian），并可选记录脚底点映射。

            Route-B: 每个接触点带接触面 (n, p0)；并保存局部基 R_T（把世界向量转到 [t1,n,t2] 坐标）。
            """
            ps = self.support_polygon[point_i] + pos
            collision_points.append(ps)
            pb = self.model.calc_base_to_body_coordinates(q, joint_id, ps)
            point_label = foot_point_labels[point_i] if point_i < 4 else f"point-{point_i}"

            n_unit = self._safe_unit_vector(surface_n, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            t1, t2 = self._make_tangent_basis(n_unit)
            R_T = np.vstack([t1.reshape(1, 3), n_unit.reshape(1, 3), t2.reshape(1, 3)]).astype(np.float64)

            collision_point_meta.append(
                {
                    "joint_name": str(joint_name),
                    "joint_id": joint_id,
                    "point_label": str(point_label),
                    "pb": pb,
                    "surface_n": n_unit,
                    "surface_p0": np.asarray(surface_p0, dtype=np.float64).reshape(3),
                    "R_T": R_T,
                }
            )
            Js.append(self.model.calc_point_Jacobian(q, joint_id, pb))
            if point_map is not None and point_i < 4:
                point_map[point_i] = len(collision_points) - 1

        for joint_name in self.test_contact_joints:
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)

            # 新结构：全关节 contact (c + p4) 优先（替代高度阈值接触判定；保留防穿模兜底）
            if contact_by_joint is not None:
                payload = contact_by_joint.get(joint_name, None)
                c_val = 0.0
                p4 = None
                n_vec = None
                p0_vec = None
                if isinstance(payload, dict):
                    try:
                        c_val = float(payload.get("c", 0.0))
                    except Exception:
                        c_val = 0.0
                    p4 = payload.get("p", None)
                    n_vec = payload.get("n", None)
                    p0_vec = payload.get("p0", None)

                # 接触点判定：严格以 p4（显式指定每个点是否接触）为准。
                # - 只要 p4 中有任意点被激活，就认为该关节接触；只加入被激活的点。
                # - c_val（接触程度）不参与“是否接触/有哪些接触点”的判断，只用于后面的“无滑动约束阈值”。
                use_indices = []
                if p4 is not None:
                    p4_arr = None
                    try:
                        p4_arr = np.asarray(p4, dtype=np.float32).reshape(-1)
                    except Exception:
                        p4_arr = None

                    if p4_arr is not None and p4_arr.size >= 4:
                        i = 0
                        while i < 4:
                            try:
                                if float(p4_arr[i]) > 0.5:
                                    use_indices.append(int(i))
                            except Exception:
                                pass
                            i += 1

                if len(use_indices) > 0:
                    collision_joints.append(joint_name)
                    point_map = None
                    if joint_id == Body.LFOOT:
                        point_map = left_point_to_collision_idx
                    elif joint_id == Body.RFOOT:
                        point_map = right_point_to_collision_idx

                    # 默认接触面：地面（y=floor_y）。Route-B 下若用户提供 n/p0 则覆盖默认。
                    floor_y = float(self.params.get("floor_y", 0.0))
                    default_n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    default_p0 = np.array([0.0, floor_y, 0.0], dtype=np.float64)
                    surface_n = default_n if n_vec is None else np.asarray(n_vec, dtype=np.float64).reshape(3)
                    surface_p0 = default_p0 if p0_vec is None else np.asarray(p0_vec, dtype=np.float64).reshape(3)

                    for point_i in use_indices:
                        _add_contact_point(joint_name, joint_id, pos, int(point_i), point_map, surface_n, surface_p0)
                continue

            # 非 dict 输入时：仅保留“左右脚接触判定 + 防穿模兜底”。
            # - Unity dict 模式下，接触完全由上层决定，不走这里。
            # - 网络输出/极简输入（2 floats）时，只提供左右脚概率，因此这里也只对脚生效。
            floor_y = float(self.params["floor_y"])
            if joint_id in (Body.LFOOT, Body.RFOOT):
                is_left = (joint_id == Body.LFOOT)
                prob = float(foot_stable[0] if is_left else foot_stable[1])
                # 3cm 门控（论文脚部判定的一部分）+ 防穿模兜底（脚低于地面时强制接触）
                should_contact = (prob > 0.5 and pos[1] <= floor_y + 0.03) or (pos[1] <= floor_y)
                if should_contact:
                    collision_joints.append(joint_name)
                    point_map = left_point_to_collision_idx if is_left else right_point_to_collision_idx
                    surface_n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    surface_p0 = np.array([0.0, float(floor_y), 0.0], dtype=np.float64)
                    for point_i in range(4):
                        _add_contact_point(joint_name, joint_id, pos, point_i, point_map, surface_n, surface_p0)
                    
        Js = np.vstack(Js)
        nc = len(collision_points)

        # === Debug: print contact joints & contact points each frame ===
        # 开关：physics_parameters.json -> debug_print_contacts = 1
        if float(self.params.get("debug_print_contacts", 0.0)) > 0.5:
            try:
                print(f"[contacts] frame={frame_idx} nc={nc} joints={collision_joints}")
                for i in range(min(nc, len(collision_point_meta))):
                    meta = collision_point_meta[i]
                    if isinstance(meta, dict):
                        jn = str(meta.get("joint_name", ""))
                        pl = str(meta.get("point_label", ""))
                    elif isinstance(meta, (list, tuple)) and len(meta) >= 2:
                        jn = str(meta[0])
                        pl = str(meta[1])
                    else:
                        jn = ""
                        pl = str(meta)
                    pt = np.asarray(collision_points[i], dtype=np.float64).reshape(3)
                    print(f"  - {i}: {jn}/{pl} pos=({pt[0]:.4f},{pt[1]:.4f},{pt[2]:.4f})")
            except Exception as e:
                print(f"[contacts] debug_print_contacts failed: {e}")

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
                # Route-B: 用“点到接触平面的符号距离”替代 (y-floor_y)
                #   s = n · (p - p0)  (s<0 表示穿透到平面内侧)
                #   d_c = max(s, eps) -> 离表面越远越压制接触力；接近/穿透时允许支撑力。
                eps = 0.005
                A = []
                for i, cp in enumerate(collision_points):
                    meta = collision_point_meta[i] if i < len(collision_point_meta) else {}
                    n = np.asarray(meta.get("surface_n", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
                    p0 = np.asarray(meta.get("surface_p0", [0.0, float(self.params.get("floor_y", 0.0)), 0.0]), dtype=np.float64).reshape(3)
                    s = float(np.dot(n, (np.asarray(cp, dtype=np.float64).reshape(3) - p0)))
                    d_c = max(s, eps)
                    A.append(np.eye(3) * d_c)
                A = art.math.block_diagonal_matrix_np(A)
                As2.append(A * self.params['coeff_lambda'])
                bs2.append(np.zeros(nc * 3))

        # [Repo扩展/非论文] 基于“重心投影”的左右脚法向力(Fy)分配先验（soft penalty）
        # 你提到的诉求：双脚8点都触地时，希望左右脚承重大小能随重心偏移而变化。
        # 现有实现只靠动力学平衡 + 正则，双支撑时 λ 分配往往冗余，容易趋向均分/数值偏置。
        
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
                            # Route-B: “法向力”不再固定为 Fy，而是 fn = n^T f。
                            for i in right_indices:
                                meta_i = collision_point_meta[i] if i < len(collision_point_meta) else {}
                                n_i = np.asarray(meta_i.get("surface_n", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
                                row[i * 3: i * 3 + 3] += (1.0 - alpha) * n_i
                            for i in left_indices:
                                meta_i = collision_point_meta[i] if i < len(collision_point_meta) else {}
                                n_i = np.asarray(meta_i.get("surface_n", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
                                row[i * 3: i * 3 + 3] += (-alpha) * n_i

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
        # === Sliding / Anti-penetration Constraints（论文 Eq.7: ṙ_j(q̈) ∈ C, Page 6）===
        # 论文：对“接触关节/接触点”的速度施加边界，水平滑动速度 < σ，同时防止向下穿透。
        # 实现：逐接触点构造约束（更符合“接触点由上层输入决定”的设定）。
        # 使用离散时间近似：v_{t+1} ≈ v_t + Δt * J * q̈
        #
        # 关键点：每个接触点所属关节的接触程度 c(0~1) 都会影响无滑动阈值：
        # - c 越大 -> 阈值 th 越小 -> 无滑动越严格
        # - c 越小 -> 阈值 th 越大 -> 无滑动更宽松
        #
        # safe_c 的裁剪用于避免 log(0)/数值爆炸（工程保护）。
        if True:
            dt = float(self.params["delta_t"])
            c_clip_min = 1e-6
            c_clip_max = 0.84999
            c_ref_max = 0.85
            th_min = 0.01

            i = 0
            while i < len(collision_point_meta):
                meta = collision_point_meta[i]
                joint_name = str(meta.get("joint_name", ""))
                joint_id = meta.get("joint_id", None)
                pb = meta.get("pb", None)

                if joint_id is None or pb is None:
                    i += 1
                    continue

                # 取该点所属关节的接触程度（默认0）
                c_val = 0.0
                if joint_name in contact_degree_by_joint:
                    try:
                        c_val = float(contact_degree_by_joint[joint_name])
                    except Exception:
                        c_val = 0.0

                # 计算该接触点的 J 与 v（点速度），并转换到接触面局部坐标系 [t1, n, t2]
                J_world = self.model.calc_point_Jacobian(q, joint_id, pb)
                v_world = self.model.calc_point_velocity(q, qdot, joint_id, pb)
                R_T = np.asarray(meta.get("R_T", np.eye(3)), dtype=np.float64).reshape(3, 3)
                J = R_T @ J_world
                v = R_T @ np.asarray(v_world, dtype=np.float64).reshape(3)

                # 由 c_val 生成水平无滑动阈值 th
                safe_c = c_val
                if safe_c < c_clip_min:
                    safe_c = c_clip_min
                if safe_c > c_clip_max:
                    safe_c = c_clip_max

                th_raw = -np.log(safe_c / c_ref_max)
                th = th_raw
                if th < th_min:
                    th = th_min

                # 防穿透（Route-B）：约束下一帧法向速度，使得 s_{t+1} >= 0
                # s = n · (p - p0)，其中 n/p0 为接触面（世界坐标）
                cp = np.asarray(collision_points[i], dtype=np.float64).reshape(3) if i < len(collision_points) else np.zeros(3, dtype=np.float64)
                n = np.asarray(meta.get("surface_n", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
                p0 = np.asarray(meta.get("surface_p0", [0.0, float(self.params.get("floor_y", 0.0)), 0.0]), dtype=np.float64).reshape(3)
                s = float(np.dot(n, (cp - p0)))
                # v_next_n >= -s/dt
                th_n = (-s) / dt

                Gs1.append(-dt * J)
                hs1.append(v - [-th, th_n, -th])
                Gs1.append(dt * J)
                hs1.append(-v + [th, max(th, th_n) + 1e-6, th])

                i += 1

        # === Friction Cone Constraint（论文 Eq.7: λ ∈ F, Page 6）===
        # 论文指出摩擦锥可线性化以保持QP；这里用4个半空间拼成“金字塔”近似库仑摩擦锥（μ=0.6）。
        # Route-B: 不再假设“Y轴是法向”，而是对每个接触点用局部基 [t1,n,t2] 施加摩擦锥：
        #   f_local = R_T f_world,  A_local f_local <= 0  => (A_local R_T) f_world <= 0
        if True:
            if nc > 0:
                blocks: List[np.ndarray] = []
                for i in range(nc):
                    meta = collision_point_meta[i] if i < len(collision_point_meta) else {}
                    R_T = np.asarray(meta.get("R_T", np.eye(3)), dtype=np.float64).reshape(3, 3)
                    blocks.append((self.friction_constraint_matrix @ R_T).astype(np.float64))
                Gs2.append(art.math.block_diagonal_matrix_np(blocks))
                hs2.append(np.zeros(nc * 4, dtype=np.float64))

                # 可选：法向力非负（防止“吸地/拉物体”）: fn = n^T f >= 0  => (-n^T) f <= 0
                if float(self.params.get("enforce_contact_fn_nonnegative", 1.0)) > 0.5:
                    nn_blocks: List[np.ndarray] = []
                    for i in range(nc):
                        meta = collision_point_meta[i] if i < len(collision_point_meta) else {}
                        n = np.asarray(meta.get("surface_n", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
                        nn_blocks.append((-n.reshape(1, 3)).astype(np.float64))
                    Gs2.append(art.math.block_diagonal_matrix_np(nn_blocks))
                    hs2.append(np.zeros(nc, dtype=np.float64))

        # === [Repo扩展] Total GRF Normal Upper Bound（硬约束）===
        # Route-B: 不再固定用 Fy，而是约束“所有接触点的法向力分量”合力上限：
        #   Σ_i (n_i^T f_i) <= k * m * g
        # - k 通过 physics_parameters.json 的 max_total_grf_y_multiple 配置（默认建议 3.0）
        # - m 为 (人体质量 + 外载荷质量)：
        #   * 人体质量 self.body_mass_kg（优先读取 params['body_mass_kg']，否则从URDF惯性质量汇总）
        #   * 外载荷质量：dumbbell_mass_left_kg / dumbbell_mass_right_kg（点质量模型）
        # - 这是线性不等式，可直接加到 λ 的 Gx<=h 里（G2 * lambda <= h2）
        if True:
            if nc > 0:
                k = float(self.params.get('max_total_grf_y_multiple', 0.0))
                if k > 0.0 and float(self.body_mass_kg) > 0.0 and float(self.gravity) > 0.0:
                    mL = float(self.params.get('dumbbell_mass_left_kg', 0.0))
                    mR = float(self.params.get('dumbbell_mass_right_kg', 0.0))
                    total_mass = float(self.body_mass_kg) + max(mL, 0.0) + max(mR, 0.0)
                    fy_max = k * total_mass * float(self.gravity)
                    row = np.zeros((nc * 3,), dtype=np.float64)
                    for i in range(nc):
                        meta = collision_point_meta[i] if i < len(collision_point_meta) else {}
                        n_i = np.asarray(meta.get("surface_n", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
                        row[i * 3: i * 3 + 3] += n_i
                    Gs2.append(row.reshape(1, -1))
                    hs2.append(np.asarray([fy_max], dtype=np.float64))
        # === Equation of Motion（论文 Eq.7: M q̈ + h = Jcᵀ λ + τ）===
        # 将动力学方程写成线性等式约束 A_ x = b_：
        #   [-M,  Jᵀ,  I] [q̈, λ, τ]ᵀ = h
        if True:
            M = self.model.calc_M(q)
            h = self.model.calc_h(q, qdot)

            # === [Repo扩展/哑铃外载荷] 动态点质量外力并入动力学（仍保持线性QP）===
            # 设哑铃为“绑在手上的点质量” m，手点世界加速度：
            #   a_hand = J_hand(q) q̈ + Jdot_hand(q,qdot) qdot
            # 哑铃对手的反作用力（世界坐标，y-up）：
            #   f = m (g_vec - a_hand),  其中 g_vec = [0, -g, 0]^T
            # 外力对应的广义力：
            #   Q = J_hand^T f = J^T m g_vec - m J^T J q̈ - m J^T (Jdot qdot)
            # 代入 M q̈ + h = Jc^T λ + τ + Q 得：
            #   (M + m J^T J) q̈ + h + m J^T (Jdot qdot) - J^T m g_vec = Jc^T λ + τ
            # 写成当前实现的线性等式形式：
            #   [-(M + m J^T J), Jc^T, I] [q̈, λ, τ]^T = h + m J^T (Jdot qdot) - J^T m g_vec
            M_eff = M.astype(np.float64, copy=True)
            b_eff = h.astype(np.float64, copy=True)

            g_vec = np.array([0.0, -float(self.gravity), 0.0], dtype=np.float64)
            
            mL = float(self.params.get('dumbbell_mass_left_kg', 0.0))
            mR = float(self.params.get('dumbbell_mass_right_kg', 0.0))

            def _add_hand_point_mass(body_id, m_kg: float):
                nonlocal M_eff, b_eff
                if m_kg <= 0.0:
                    return
                
                # 硬编码偏移：点在“手link局部坐标系”里，先用 10cm 量级试
                if body_id == Body.LHAND:
                    offset_b = np.array([0.0, 0.0, 0.00], dtype=np.float64)
                elif body_id == Body.RHAND:
                    offset_b = np.array([0.0, 0.0, 0.00], dtype=np.float64)
                else:
                    offset_b = np.zeros(3, dtype=np.float64)
                J = self.model.calc_point_Jacobian(q, body_id, offset_b).astype(np.float64)
                # 令 q̈=0 时的点加速度，主要是 Jdot*qdot（+ 其它科氏项；RBDL会给出一致的表达）
                a0 = self.model.calc_point_acceleration(
                    q, qdot, np.zeros(self.model.qdot_size, dtype=np.float64), body_id, offset_b
                ).astype(np.float64)
                M_eff = M_eff + float(m_kg) * (J.T @ J)
                b_eff = b_eff + float(m_kg) * (J.T @ a0) - (J.T @ (float(m_kg) * g_vec))

            _add_hand_point_mass(Body.LHAND, mL)
            _add_hand_point_mass(Body.RHAND, mR)

            A_ = np.hstack((-M_eff, Js.T, np.eye(self.model.qdot_size)))
            b_ = b_eff

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
            #             meta = collision_point_meta[i]
            #             joint_name = str(meta.get("joint_name", ""))
            #             point_label = str(meta.get("point_label", ""))
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

        # [Repo扩展] 返回“全接触点列表（固定全集）”：
        # - 以 test_contact_joints 为全集，对每个关节固定输出 4 个点（support_polygon 的四角）
        # - 若该点本帧进入 QP 接触集合，则填入对应 GRF；否则 force 填 [0,0,0]
        #
        # 注意：
        # - 即便上层显式只选择了部分点参与接触，这里仍会输出 4 点，其余点为 0，便于 Unity 侧做固定结构解析。
        contact_force_by_meta = {}
        if nc > 0 and len(collision_point_meta) > 0:
            n_out = min(nc, len(collision_point_meta), len(grf_components))
            for idx in range(n_out):
                meta = collision_point_meta[idx]
                # 用可哈希的稳定键： (joint_name, point_label)
                joint_name = str(meta.get("joint_name", "")) if isinstance(meta, dict) else ""
                point_label = str(meta.get("point_label", "")) if isinstance(meta, dict) else ""
                key = (joint_name, point_label)
                contact_force_by_meta[key] = grf_components[idx].tolist()

        contacts = []
        for joint_name in self.test_contact_joints:
            for point_label in foot_point_labels:  # 固定 4 点
                force = contact_force_by_meta.get((joint_name, point_label), [0.0, 0.0, 0.0])
                contacts.append({
                    'joint': str(joint_name),
                    'point': str(point_label),
                    'force': force,
                })
        labeled_grf['contacts'] = contacts

        # # 始终输出左右脚各4点（固定顺序），未参与QP的点补0，保证Unity侧解包长度稳定（8点*3=24 floats）
        # left_grf_data = []
        # for i, point_label in enumerate(foot_point_labels):
        #     idx = left_point_to_collision_idx.get(i, None)
        #     force = grf_components[idx].tolist() if idx is not None and idx < len(grf_components) else [0.0, 0.0, 0.0]
        #     left_grf_data.append({'point': point_label, 'force': force})
        # labeled_grf['left_foot'] = left_grf_data

        # right_grf_data = []
        # for i, point_label in enumerate(foot_point_labels):
        #     idx = right_point_to_collision_idx.get(i, None)
        #     force = grf_components[idx].tolist() if idx is not None and idx < len(grf_components) else [0.0, 0.0, 0.0]
        #     right_grf_data.append({'point': point_label, 'force': force})
        # labeled_grf['right_foot'] = right_grf_data
            
        # # Add other contact points if any
        # all_foot_indices = set(list(left_point_to_collision_idx.values()) + list(right_point_to_collision_idx.values()))
        # other_indices = [i for i in range(len(grf_components)) if i not in all_foot_indices]
        # if other_indices:
        #     other_grf_data = []
        #     for i, idx in enumerate(other_indices):
        #         if idx < len(collision_point_meta):
        #             joint_name, point_label = collision_point_meta[idx]
        #             other_grf_data.append({
        #                 'joint': joint_name,
        #                 'point': point_label,
        #                 'force': grf_components[idx].tolist()
        #             })
        #     labeled_grf['other_contacts'] = other_grf_data

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
