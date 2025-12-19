import torch
import numpy as np
import pybullet as p
import articulate as art
from articulate.utils.bullet import *
from articulate.utils.rbdl import *
from utils import *
from qpsolvers import solve_qp
from config import paths


class PhysicsOptimizer:
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

    def reset_states(self):
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)

    def optimize_frame(self, pose, jvel, contact, acc):
        q_ref = smpl_to_rbdl(pose, torch.zeros(3))[0]  # 神经网络预测的姿态
        v_ref = jvel if isinstance(jvel, np.ndarray) else jvel.numpy()  # 神经网络预测的关节速度
        c_ref = np.array(c_ref, dtype=np.float32).reshape(-1)

        # contact 新结构（10 floats）：
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
            return pose, torch.zeros(3)
        print('optimize frame')
        # determine the contact joints and points
        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))
        Js = [np.empty((0, self.model.qdot_size))]
        collision_points, collision_joints = [], []

        foot_point_labels = ['front-left', 'front-right', 'back-left', 'back-right']
        # 记录每个脚底点（0..3）对应的 collision_points 索引（只记录参与QP的点）
        left_point_to_collision_idx = {}
        right_point_to_collision_idx = {}
        # 记录每个 collision point 的元信息，便于debug输出与回传结构化GRF
        collision_point_meta = []  # list[tuple[str joint_name, str point_label]]
        
        for joint_name in self.test_contact_joints:
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)

            # 新结构：脚底接触点由foot_point_flags(0/1)显式给定，用来替代旧的“固定4点+高度阈值”的取法
            if joint_id in (Body.LFOOT, Body.RFOOT) and foot_point_flags is not None:
                if joint_id == Body.LFOOT:
                    flags = foot_point_flags[:4]
                    point_map = left_point_to_collision_idx
                else:
                    flags = foot_point_flags[4:]
                    point_map = right_point_to_collision_idx

                selected = [i for i, f in enumerate(flags) if float(f) > 0.5]

                # 若显式点接触为空，但模型穿透地面，则仍然加约束防止穿模（旧逻辑保留）
                if selected or pos[1] <= self.params['floor_y']:
                    collision_joints.append(joint_name)
                    use_indices = selected if selected else list(range(4))
                    for i in use_indices:
                        ps = self.support_polygon[i] + pos
                        collision_points.append(ps)
                        collision_point_meta.append((joint_name, foot_point_labels[i]))
                        pb = self.model.calc_base_to_body_coordinates(q, joint_id, ps)
                        Js.append(self.model.calc_point_Jacobian(q, joint_id, pb))
                        point_map[i] = len(collision_points) - 1
                continue

            # 旧结构/其它关节：维持原来的接触判定与固定4点
            if (joint_id == Body.LFOOT and foot_stable[0] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03) or \
               (joint_id == Body.RFOOT and foot_stable[1] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03) or \
               (pos[1] <= self.params['floor_y']):
                collision_joints.append(joint_name)
                for i, ps in enumerate(self.support_polygon + pos):
                    collision_points.append(ps)
                    # 非脚部关节也给一个稳定的point label，方便debug；脚部在旧结构下也沿用这套label
                    collision_point_meta.append((joint_name, foot_point_labels[i] if i < len(foot_point_labels) else f'point-{i}'))
                    pb = self.model.calc_base_to_body_coordinates(q, joint_id, ps)
                    Js.append(self.model.calc_point_Jacobian(q, joint_id, pb))
                    # 旧结构下脚底也是固定4点：这里也要记录索引，保证回传GRF时左右脚4点不丢失
                    if joint_id == Body.LFOOT and i < 4:
                        left_point_to_collision_idx[i] = len(collision_points) - 1
                    elif joint_id == Body.RFOOT and i < 4:
                        right_point_to_collision_idx[i] = len(collision_points) - 1
                    
        Js = np.vstack(Js)
        nc = len(collision_points)

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

        # joint angle PD controller（对应论文公式3）
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

        # joint position PD controller (using joint velocity to determine target joint position)（论文公式5）
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

        # Signorini’s conditions of lambda对应论文公式9中的E_lambda
        if True:
            if nc != 0:
                A = [np.eye(3) * max(cp[1] - self.params['floor_y'], 0.005) for cp in collision_points]
                A = art.math.block_diagonal_matrix_np(A)
                As2.append(A * self.params['coeff_lambda'])
                bs2.append(np.zeros(nc * 3))

        # tau size(包含残余力约束)
        if True:
            As3.append(art.math.block_diagonal_matrix_np([
                np.eye(6) * self.params['coeff_virtual'] * 3.16, # 对应论文中的残余力约束
                np.eye(self.model.qdot_size - 6) * self.params['coeff_tau']#关节扭矩约束
            ]))
            bs3.append(np.zeros(self.model.qdot_size))

        # contacting body joint velocity
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

        # contacting foot velocity
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

        # GRF friction cone constraint
        if True:
            if nc > 0:
                Gs2.append(art.math.block_diagonal_matrix_np([self.friction_constraint_matrix] * nc))
                hs2.append(np.zeros(nc * 4))

        # equation of motion (equality constraint)
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

        # 添加调试信息，打印 tau 和 GRF 的形状
        if self.debug:
            # 计算并输出GRF在所有轴上的合力
            if nc > 0:
                grf_components = GRF.reshape(-1, 3)
                grf_total = np.sum(grf_components, axis=0)
                grf_magnitude = np.linalg.norm(grf_total)
                print(f"GRF total force - X: {grf_total[0]:.6f}, Y: {grf_total[1]:.6f}, Z: {grf_total[2]:.6f}")
                print(f"GRF total magnitude: {grf_magnitude:.6f}")

                # 输出左右脚各自的GRF
                left_foot_indices = list(left_point_to_collision_idx.values())
                right_foot_indices = list(right_point_to_collision_idx.values())

                if left_foot_indices:
                    left_grf = grf_components[left_foot_indices]
                    left_total = np.sum(left_grf, axis=0)
                    print(f"LFOOT GRF - X: {left_total[0]:.6f}, Y: {left_total[1]:.6f}, Z: {left_total[2]:.6f}")
                
                if right_foot_indices:
                    right_grf = grf_components[right_foot_indices]
                    right_total = np.sum(right_grf, axis=0)
                    print(f"RFOOT GRF - X: {right_total[0]:.6f}, Y: {right_total[1]:.6f}, Z: {right_total[2]:.6f}")

                # 打印每个接触点的力分量，并标注是哪个关节以及具体的点
                for i, point_force in enumerate(grf_components):
                    if i < len(collision_point_meta):
                        joint_name, point_label = collision_point_meta[i]
                        print(f"GRF component [{joint_name} - {point_label}]: "
                              f"X: {point_force[0]:.6f}, Y: {point_force[1]:.6f}, Z: {point_force[2]:.6f}")
                    else:
                        print(f"GRF component [Unknown - point {i}]: "
                              f"X: {point_force[0]:.6f}, Y: {point_force[1]:.6f}, Z: {point_force[2]:.6f}")
        qdot = qdot + qddot * self.params['delta_t']
        q = q + qdot * self.params['delta_t']
        #self.q = q
        self.q = q_ref.copy()
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

        # 添加虚拟力的六项到返回值
        virtual_force = tau[:6]  # 提取虚拟力的六项

        return pose_opt, tran_opt, labeled_grf, virtual_force
