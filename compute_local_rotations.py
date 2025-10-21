import numpy as np
import torch
from articulate.model import ParametricModel
from articulate.math import forward_kinematics_R, inverse_kinematics_R, svd_rotate
from config import paths
import pickle


def compute_local_rotations_from_global_positions(global_positions, root_rotation=None, smpl_model_path=None):
    """
    根据SMPL各个关节的全局位置计算每个关节的局部旋转矩阵
    
    Args:
        global_positions: 全局关节位置，形状为 [num_joints, 3] 或 [batch_size, num_joints, 3]
        root_rotation: 根骨骼的全局旋转矩阵，形状为 [3, 3] 或 [batch_size, 3, 3]
        smpl_model_path: SMPL模型文件路径，默认使用配置中的路径
    
    Returns:
        local_rotations: 局部旋转矩阵，形状为 [batch_size, num_joints, 3, 3] 或 [num_joints, 3, 3]
    """
    # 确保输入是numpy数组
    global_positions = np.array(global_positions)
    
    # 处理输入维度
    if global_positions.ndim == 2:
        # 单帧情况: [num_joints, 3]
        global_positions = np.expand_dims(global_positions, axis=0)
        single_frame = True
    else:
        # 多帧情况: [batch_size, num_joints, 3]
        single_frame = False
    
    batch_size, num_joints, _ = global_positions.shape
    
    # 加载SMPL模型
    if smpl_model_path is None:
        smpl_model_path = paths.smpl_file
    
    # 使用ParametricModel加载模型获取父关节索引
    model = ParametricModel(smpl_model_path)
    parent_indices = model.parent
    
    # 转换为torch tensor进行计算
    global_positions_torch = torch.from_numpy(global_positions).float()
    
    # 计算骨骼向量 (从父关节到子关节的向量)
    bone_vectors = torch.zeros_like(global_positions_torch)
    bone_vectors[:, 0, :] = global_positions_torch[:, 0, :]  # 根关节位置就是骨骼向量
    
    for i in range(1, num_joints):
        parent_idx = parent_indices[i]
        if parent_idx is not None:  # 确保父关节存在
            bone_vectors[:, i, :] = global_positions_torch[:, i, :] - global_positions_torch[:, parent_idx, :]
    
    # 获取零姿态下的关节位置
    zero_pose_joints, _ = model.get_zero_pose_joint_and_vertex()
    zero_pose_joints = zero_pose_joints.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_joints, 3]
    
    # 计算骨骼向量 (从零姿态下的父关节到子关节的向量)
    zero_bone_vectors = torch.zeros_like(zero_pose_joints)
    zero_bone_vectors[:, 0, :] = zero_pose_joints[:, 0, :]
    
    for i in range(1, num_joints):
        parent_idx = parent_indices[i]
        if parent_idx is not None:
            zero_bone_vectors[:, i, :] = zero_pose_joints[:, i, :] - zero_pose_joints[:, parent_idx, :]
    
    # 估计全局旋转矩阵
    global_rotations = torch.zeros((batch_size, num_joints, 3, 3))
    
    # 对于根关节，使用提供的根骨骼旋转或默认单位矩阵
    if root_rotation is not None:
        root_rotation = torch.from_numpy(np.array(root_rotation)).float()
        if root_rotation.ndim == 2:
            root_rotation = root_rotation.unsqueeze(0).repeat(batch_size, 1, 1)
        global_rotations[:, 0, :, :] = root_rotation
    else:
        # 默认单位矩阵
        global_rotations[:, 0, :, :] = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 对于其他关节，我们需要根据骨骼向量来估计旋转
    for i in range(1, num_joints):
        # 使用SVD方法计算从零姿态骨骼向量到目标骨骼向量的旋转
        source_points = zero_bone_vectors[:, i, :].unsqueeze(1)  # [batch_size, 1, 3]
        target_points = bone_vectors[:, i, :].unsqueeze(1)       # [batch_size, 1, 3]
        
        # 需要确保向量非零才能计算旋转
        source_norm = torch.norm(source_points, dim=2, keepdim=True)
        target_norm = torch.norm(target_points, dim=2, keepdim=True)
        
        # 只有当两个向量都不为零时才计算旋转
        valid = (source_norm.squeeze(-1) > 1e-6) & (target_norm.squeeze(-1) > 1e-6)
        
        if valid.any():
            # 对于有效的样本，计算旋转
            # 修正维度问题：扩展向量以满足svd_rotate函数的需求
            valid_count = source_points[valid].shape[0]
            if valid_count > 0:
                # 创建3个点的集合，第一个点是原始向量，其他两个是正交向量
                source_batch = torch.zeros(valid_count, 3, 3, device=source_points.device)
                target_batch = torch.zeros(valid_count, 3, 3, device=target_points.device)
                
                src_vecs = source_points[valid].squeeze(1)  # [valid_count, 3]
                tgt_vecs = target_points[valid].squeeze(1)  # [valid_count, 3]
                
                for j in range(valid_count):
                    # 获取单个向量
                    src_vec = src_vecs[j, :]  # [3]
                    tgt_vec = tgt_vecs[j, :]  # [3]
                    
                    # 标准化向量
                    src_vec_norm = src_vec / (torch.norm(src_vec) + 1e-8)
                    tgt_vec_norm = tgt_vec / (torch.norm(tgt_vec) + 1e-8)
                    
                    # 创建正交基
                    # 选择一个不与主向量平行的向量
                    if abs(src_vec_norm[0]) < 0.9:
                        aux_vec = torch.tensor([1.0, 0.0, 0.0], device=src_vec.device)
                    else:
                        aux_vec = torch.tensor([0.0, 1.0, 0.0], device=src_vec.device)
                    
                    # 构造正交基
                    src_orth1 = torch.cross(src_vec_norm, aux_vec)
                    src_orth1 = src_orth1 / (torch.norm(src_orth1) + 1e-8)
                    src_orth2 = torch.cross(src_vec_norm, src_orth1)
                    
                    tgt_orth1 = torch.cross(tgt_vec_norm, aux_vec)
                    tgt_orth1 = tgt_orth1 / (torch.norm(tgt_orth1) + 1e-8)
                    tgt_orth2 = torch.cross(tgt_vec_norm, tgt_orth1)
                    
                    # 构造3x3点集
                    source_batch[j, 0, :] = src_vec
                    source_batch[j, 1, :] = src_orth1
                    source_batch[j, 2, :] = src_orth2
                    
                    target_batch[j, 0, :] = tgt_vec
                    target_batch[j, 1, :] = tgt_orth1
                    target_batch[j, 2, :] = tgt_orth2
                
                # 计算旋转矩阵
                rotation_matrices = svd_rotate(source_batch, target_batch)
                
                # 如果源向量和目标向量长度不同，添加缩放
                scale = target_norm[valid] / source_norm[valid]
                # 正确应用缩放因子到旋转矩阵
                scaled_rotation = rotation_matrices * scale.unsqueeze(-1).unsqueeze(-1)
                # 修复索引错误，正确地将旋转矩阵分配给global_rotations
                for idx, batch_idx in enumerate(torch.where(valid)[0]):
                    global_rotations[batch_idx, i, :, :] = scaled_rotation[idx]
        else:
            # 如果向量为零或非常小，使用单位矩阵
            global_rotations[:, i, :, :] = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # 使用逆向运动学计算局部旋转矩阵
    local_rotations = inverse_kinematics_R(global_rotations, parent_indices)
    
    # 转换回numpy数组
    local_rotations = local_rotations.numpy()
    
    # 如果是单帧输入，去掉batch维度
    if single_frame:
        local_rotations = local_rotations[0]
    
    return local_rotations


def compute_local_rotations_from_relative_positions(relative_positions, root_rotation=None, smpl_model_path=None):
    """
    根据SMPL各个关节相对于父关节的位置计算每个关节的局部旋转矩阵
    
    Args:
        relative_positions: 相对于父关节的位置，形状为 [num_joints, 3] 或 [batch_size, num_joints, 3]
        root_rotation: 根骨骼的全局旋转矩阵，形状为 [3, 3] 或 [batch_size, 3, 3]
        smpl_model_path: SMPL模型文件路径，默认使用配置中的路径
    
    Returns:
        local_rotations: 局部旋转矩阵，形状为 [batch_size, num_joints, 3, 3] 或 [num_joints, 3, 3]
    """
    # 确保输入是numpy数组
    relative_positions = np.array(relative_positions)
    
    # 处理输入维度
    if relative_positions.ndim == 2:
        # 单帧情况: [num_joints, 3]
        relative_positions = np.expand_dims(relative_positions, axis=0)
        single_frame = True
    else:
        # 多帧情况: [batch_size, num_joints, 3]
        single_frame = False
    
    batch_size, num_joints, _ = relative_positions.shape
    
    # 加载SMPL模型
    if smpl_model_path is None:
        smpl_model_path = paths.smpl_file
    
    # 使用ParametricModel加载模型获取父关节索引
    model = ParametricModel(smpl_model_path)
    parent_indices = model.parent
    
    # 转换为torch tensor进行计算
    relative_positions_torch = torch.from_numpy(relative_positions).float()
    
    # 获取零姿态下的关节相对位置
    zero_pose_joints, _ = model.get_zero_pose_joint_and_vertex()
    
    # 计算零姿态下相对于父关节的位置
    zero_pose_relative = torch.zeros_like(zero_pose_joints)
    zero_pose_relative[0] = zero_pose_joints[0]  # 根关节位置
    
    for i in range(1, len(parent_indices)):
        parent_idx = parent_indices[i]
        if parent_idx is not None:
            zero_pose_relative[i] = zero_pose_joints[i] - zero_pose_joints[parent_idx]
    
    zero_pose_relative = zero_pose_relative.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_joints, 3]
    
    # 估计局部旋转矩阵
    local_rotations = torch.zeros((batch_size, num_joints, 3, 3))
    
    # 对于根关节，使用提供的根骨骼旋转或默认单位矩阵
    if root_rotation is not None:
        root_rotation = torch.from_numpy(np.array(root_rotation)).float()
        if root_rotation.ndim == 2:
            root_rotation = root_rotation.unsqueeze(0).repeat(batch_size, 1, 1)
        local_rotations[:, 0, :, :] = root_rotation
    else:
        # 默认单位矩阵
        local_rotations[:, 0, :, :] = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 对于其他关节，我们需要根据相对位置来估计旋转
    for i in range(1, num_joints):
        # 使用SVD方法计算从零姿态相对位置到目标相对位置的旋转
        source_points = zero_pose_relative[:, i, :].unsqueeze(1)  # [batch_size, 1, 3]
        target_points = relative_positions_torch[:, i, :].unsqueeze(1)       # [batch_size, 1, 3]
        
        # 需要确保向量非零才能计算旋转
        source_norm = torch.norm(source_points, dim=2, keepdim=True)
        target_norm = torch.norm(target_points, dim=2, keepdim=True)
        
        # 只有当两个向量都不为零时才计算旋转
        valid = (source_norm.squeeze(-1) > 1e-6) & (target_norm.squeeze(-1) > 1e-6)
        
        if valid.any():
            # 对于有效的样本，计算旋转
            # 修正维度问题：扩展向量以满足svd_rotate函数的需求
            valid_count = source_points[valid].shape[0]
            if valid_count > 0:
                # 创建3个点的集合，第一个点是原始向量，其他两个是正交向量
                source_batch = torch.zeros(valid_count, 3, 3, device=source_points.device)
                target_batch = torch.zeros(valid_count, 3, 3, device=target_points.device)
                
                src_vecs = source_points[valid].squeeze(1)  # [valid_count, 3]
                tgt_vecs = target_points[valid].squeeze(1)  # [valid_count, 3]
                
                for j in range(valid_count):
                    # 获取单个向量
                    src_vec = src_vecs[j, :]  # [3]
                    tgt_vec = tgt_vecs[j, :]  # [3]
                    
                    # 标准化向量
                    src_vec_norm = src_vec / (torch.norm(src_vec) + 1e-8)
                    tgt_vec_norm = tgt_vec / (torch.norm(tgt_vec) + 1e-8)
                    
                    # 创建正交基
                    # 选择一个不与主向量平行的向量
                    if abs(src_vec_norm[0]) < 0.9:
                        aux_vec = torch.tensor([1.0, 0.0, 0.0], device=src_vec.device)
                    else:
                        aux_vec = torch.tensor([0.0, 1.0, 0.0], device=src_vec.device)
                    
                    # 构造正交基
                    src_orth1 = torch.cross(src_vec_norm, aux_vec)
                    src_orth1 = src_orth1 / (torch.norm(src_orth1) + 1e-8)
                    src_orth2 = torch.cross(src_vec_norm, src_orth1)
                    
                    tgt_orth1 = torch.cross(tgt_vec_norm, aux_vec)
                    tgt_orth1 = tgt_orth1 / (torch.norm(tgt_orth1) + 1e-8)
                    tgt_orth2 = torch.cross(tgt_vec_norm, tgt_orth1)
                    
                    # 构造3x3点集
                    source_batch[j, 0, :] = src_vec
                    source_batch[j, 1, :] = src_orth1
                    source_batch[j, 2, :] = src_orth2
                    
                    target_batch[j, 0, :] = tgt_vec
                    target_batch[j, 1, :] = tgt_orth1
                    target_batch[j, 2, :] = tgt_orth2
                
                # 计算旋转矩阵
                rotation_matrices = svd_rotate(source_batch, target_batch)
                
                # 如果源向量和目标向量长度不同，添加缩放
                scale = target_norm[valid] / source_norm[valid]
                # 正确应用缩放因子到旋转矩阵
                scaled_rotation = rotation_matrices * scale.unsqueeze(-1).unsqueeze(-1)
                # 修复索引错误，正确地将旋转矩阵分配给local_rotations
                for idx, batch_idx in enumerate(torch.where(valid)[0]):
                    local_rotations[batch_idx, i, :, :] = scaled_rotation[idx]
        else:
            # 如果向量为零或非常小，使用单位矩阵
            local_rotations[:, i, :, :] = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # 转换回numpy数组
    local_rotations = local_rotations.numpy()
    
    # 如果是单帧输入，去掉batch维度
    if single_frame:
        local_rotations = local_rotations[0]
    
    return local_rotations


def compute_local_rotations_iterative(global_positions, smpl_model_path=None, max_iterations=100, tolerance=1e-5):
    """
    使用迭代方法根据SMPL各个关节的全局位置计算每个关节的局部旋转矩阵
    
    Args:
        global_positions: 全局关节位置，形状为 [num_joints, 3] 或 [batch_size, num_joints, 3]
        smpl_model_path: SMPL模型文件路径，默认使用配置中的路径
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
    
    Returns:
        local_rotations: 局部旋转矩阵，形状为 [batch_size, num_joints, 3, 3] 或 [num_joints, 3, 3]
    """
    # 确保输入是numpy数组
    global_positions = np.array(global_positions)
    
    # 处理输入维度
    if global_positions.ndim == 2:
        # 单帧情况: [num_joints, 3]
        global_positions = np.expand_dims(global_positions, axis=0)
        single_frame = True
    else:
        # 多帧情况: [batch_size, num_joints, 3]
        single_frame = False
    
    batch_size, num_joints, _ = global_positions.shape
    
    # 加载SMPL模型
    if smpl_model_path is None:
        smpl_model_path = paths.smpl_file
    
    # 使用ParametricModel加载模型获取父关节索引
    model = ParametricModel(smpl_model_path)
    parent_indices = model.parent
    
    # 获取零姿态下的关节位置
    zero_pose_joints, _ = model.get_zero_pose_joint_and_vertex()
    zero_pose_joints = zero_pose_joints.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_joints, 3]
    
    # 转换为torch tensor进行计算
    global_positions_torch = torch.from_numpy(global_positions).float()
    
    # 初始化局部旋转矩阵
    local_rotations = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_joints, 1, 1)  # [batch_size, num_joints, 3, 3]
    
    # 迭代优化局部旋转矩阵
    for iteration in range(max_iterations):
        # 前向运动学计算当前估计的全局关节位置
        _, current_global_positions = model.forward_kinematics(local_rotations, calc_mesh=False)
        
        # 计算误差
        error = torch.norm(current_global_positions - global_positions_torch)
        
        # 如果误差小于容差，则停止迭代
        if error < tolerance:
            print(f"收敛于第 {iteration+1} 次迭代，误差: {error}")
            break
        
        # 计算需要调整的关节位置差
        position_diff = global_positions_torch - current_global_positions
        
        # 简单的梯度更新方法
        learning_rate = 0.1
        # 对于每个关节（除了根关节），根据位置差调整旋转
        for i in range(1, num_joints):
            parent_idx = parent_indices[i]
            if parent_idx is not None:
                # 计算关节i相对于其父关节的位置差
                joint_diff = position_diff[:, i, :]
                parent_diff = position_diff[:, parent_idx, :] if parent_idx >= 0 else torch.zeros_like(joint_diff)
                relative_diff = joint_diff - parent_diff
                
                # 将相对差转换为旋转调整（简化处理）
                # 这里我们只是简单地将差值添加到当前旋转中
                for b in range(batch_size):
                    if torch.norm(relative_diff[b]) > 1e-6:
                        # 创建一个基于位置差的小旋转调整
                        axis = torch.cross(torch.rand(3), relative_diff[b])
                        axis = axis / (torch.norm(axis) + 1e-8)
                        angle = learning_rate * torch.norm(relative_diff[b])
                        delta_rot = axis_angle_to_rotation_matrix_single(axis * angle)
                        # 应用旋转调整
                        local_rotations[b, i] = torch.matmul(delta_rot, local_rotations[b, i])

    # 返回局部旋转矩阵
    local_rotations = local_rotations.numpy()
    
    # 如果是单帧输入，去掉batch维度
    if single_frame:
        local_rotations = local_rotations[0]
    
    return local_rotations


def axis_angle_to_rotation_matrix_single(a):
    """
    将轴角转换为旋转矩阵 (单个向量)
    
    :param a: 轴角向量 (3,)
    :return: 旋转矩阵 (3, 3)
    """
    axis = a / (torch.norm(a) + 1e-8)
    angle = torch.norm(a)
    
    if angle < 1e-6:
        return torch.eye(3)
    
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    
    # 创建叉积矩阵
    cross_matrix = torch.zeros(3, 3)
    cross_matrix[0, 1] = -axis[2]
    cross_matrix[0, 2] = axis[1]
    cross_matrix[1, 0] = axis[2]
    cross_matrix[1, 2] = -axis[0]
    cross_matrix[2, 0] = -axis[1]
    cross_matrix[2, 1] = axis[0]
    
    R = cos * torch.eye(3) + (1 - cos) * torch.ger(axis, axis) + sin * cross_matrix
    return R


def example_usage():
    """
    使用示例
    """
    # 示例：创建一些测试数据
    # 假设有24个SMPL关节
    num_joints = 24
    test_positions = np.random.rand(num_joints, 3)  # 随机生成关节位置
    
    # 计算局部旋转矩阵（不指定根骨骼旋转）
    local_rots = compute_local_rotations_from_global_positions(test_positions)
    
    print(f"输入形状: {test_positions.shape}")
    print(f"输出形状: {local_rots.shape}")
    print("局部旋转矩阵计算完成（默认根骨骼旋转）")
    
    # 计算局部旋转矩阵（指定根骨骼旋转）
    root_rotation = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])  # 单位矩阵作为根骨骼旋转
    local_rots_with_root = compute_local_rotations_from_global_positions(test_positions, root_rotation)
    
    print(f"指定根骨骼旋转后输出形状: {local_rots_with_root.shape}")
    print("局部旋转矩阵计算完成（指定根骨骼旋转）")
    
    # 测试相对位置输入的函数
    # 创建一些相对位置数据进行测试
    test_relative_positions = np.random.rand(num_joints, 3)  # 随机生成相对位置
    local_rots_from_relative = compute_local_rotations_from_relative_positions(test_relative_positions)
    
    print(f"相对位置输入形状: {test_relative_positions.shape}")
    print(f"相对位置输出形状: {local_rots_from_relative.shape}")
    print("局部旋转矩阵计算完成（相对位置输入）")
    
    # 测试迭代方法
    local_rots_iter = compute_local_rotations_iterative(test_positions)
    print(f"迭代方法输出形状: {local_rots_iter.shape}")
    print("迭代方法局部旋转矩阵计算完成")


if __name__ == "__main__":
    example_usage()