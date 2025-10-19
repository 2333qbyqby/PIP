import socket
import threading
import queue
import time
from typing import List, Tuple, Optional
import pybullet as p
import numpy as np
import sys
import pathlib
from articulate.utils.rbdl import *
from articulate.utils.bullet import *
from utils import *
# 添加项目根目录到Python路径
project_root = pathlib.Path(__file__).parent
sys.path.append(str(project_root))

# 导入配置和工具
from config import paths
from utils import set_pose, _rbdl_to_bullet
import articulate as art
import torch


class UnityConnector:
    """
    Unity连接器，用于与Unity进行双向通信
    """

    def __init__(self, host='127.0.0.1', port=8888):
        """
        初始化Unity连接器
        
        Args:
            host: 服务器地址
            port: 服务器端口
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connection = None
        
        # 队列用于存储待发送的数据和已接收的数据
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()
        
        # 线程控制
        self.send_thread = None
        self.receive_thread = None
        self.running = False
        
        # PyBullet相关属性
        self.physics_client = None
        self.robot_id = None
        self.skeleton_debug_items = []  # 用于存储骨架可视化调试项的ID
        
        # SMPL模型用于骨架可视化
        self.smpl_model = art.ParametricModel(paths.smpl_file)

    def connect_as_server(self) -> bool:
        """
        作为服务器连接到Unity客户端
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f'Unity通信服务器启动，监听 {self.host}:{self.port}')
            
            print("等待Unity客户端连接...")
            self.connection, addr = self.socket.accept()
            print(f"Unity客户端已连接: {addr}")
            
            self.running = True
            self._start_threads()
            return True
        except Exception as e:
            print(f"服务器连接失败: {e}")
            return False

    def connect_as_client(self) -> bool:
        """
        作为客户端连接到Unity服务器
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.host, self.port))
            print(f"已连接到Unity服务器 {self.host}:{self.port}")
            
            self.running = True
            self._start_threads()
            return True
        except Exception as e:
            print(f"客户端连接失败: {e}")
            return False

    def _start_threads(self):
        """
        启动发送和接收线程
        """
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        
        self.send_thread.start()
        self.receive_thread.start()

    def _send_loop(self):
        """
        发送数据的循环
        """
        while self.running:
            try:
                # 从队列中获取待发送的数据
                data = self.send_queue.get(timeout=1)
                if data is None:
                    continue
                    
                pose_data, tran_data, cj_data, grf_data = data
                success = self._send_motion_data(pose_data, tran_data, cj_data, grf_data)
                if not success:
                    print("发送数据失败")
                    self._handle_disconnect()
                    break
            except queue.Empty:
                continue
            except Exception as e:
                print(f"发送数据时出错: {e}")
                self._handle_disconnect()
                break

    def _receive_loop(self):
        """
        接收数据的循环
        """
        buffer = ""
        while self.running:
            try:
                # 接收数据
                chunk = self.connection.recv(4096).decode('utf-8')
                if not chunk:
                    print("连接已断开")
                    self._handle_disconnect()
                    break
                    
                buffer += chunk
                # 处理完整的数据包
                while '$' in buffer:
                    message, buffer = buffer.split('$', 1)
                    if message:
                        try:
                            parsed_data = self._parse_unity_data(message + '$')
                            self.receive_queue.put(parsed_data)
                        except ValueError as e:
                            print(f"数据解析错误: {e}")
            except Exception as e:
                print(f"接收数据时出错: {e}")
                self._handle_disconnect()
                break

    def _handle_disconnect(self):
        """
        处理连接断开事件
        """
        if self.running:
            print("检测到连接断开，正在关闭连接...")
            self.running = False
            self.close()

    def _send_motion_data(self, pose_data: List[float], tran_data: List[float], 
                         cj_data: Optional[List[int]] = None, 
                         grf_data: Optional[List[float]] = None) -> bool:
        """
        发送动作数据到Unity
        
        Args:
            pose_data: 姿态数据
            tran_data: 位移数据
            cj_data: 接触关节数据
            grf_data: 地面反作用力数据
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # 格式化数据
            pose_str = ','.join(['%g' % v for v in pose_data])
            tran_str = ','.join(['%g' % v for v in tran_data])
            cj_str = ','.join(['%d' % v for v in cj_data]) if cj_data else ''
            grf_str = ','.join(['%g' % v for v in grf_data]) if grf_data else ''
            
            # 构造消息字符串
            message = f"{pose_str}#{tran_str}#{cj_str}#{grf_str}$"
            
            self.connection.send(message.encode('utf8'))
            return True
        except Exception as e:
            print(f"发送数据失败: {e}")
            return False

    def _parse_unity_data(self, data: str) -> Tuple[List[float], List[float], List[int], List[float]]:
        """
        解析从Unity接收到的数据
        
        Args:
            data: 从Unity接收到的原始数据字符串
            
        Returns:
            tuple: (pose_data, tran_data, cj_data, grf_data) 解析后的数据元组
        """
        # 移除末尾的结束符
        data = data.rstrip('$')
        
        # 分割数据的各个部分
        parts = data.split('#')
        
        if len(parts) != 4:
            raise ValueError(f"数据格式错误：期望4个部分，实际收到{len(parts)}个部分")
        
        pose_str, tran_str, cj_str, grf_str = parts
        
        # 解析姿态数据 (pose_data)
        pose_data = [float(x) for x in pose_str.split(',')] if pose_str else []
        
        # 解析位移数据 (tran_data)
        tran_data = [float(x) for x in tran_str.split(',')] if tran_str else []
        
        # 解析接触关节数据 (cj_data)
        cj_data = [int(x) for x in cj_str.split(',')] if cj_str else []
        
        # 解析地面反作用力数据 (grf_data)
        grf_data = [float(x) for x in grf_str.split(',')] if grf_str else []
        
        return pose_data, tran_data, cj_data, grf_data

    def send_data(self, pose_data: List[float], tran_data: List[float], 
                  cj_data: Optional[List[int]] = None, 
                  grf_data: Optional[List[float]] = None):
        """
        将数据添加到发送队列
        
        Args:
            pose_data: 姿态数据
            tran_data: 位移数据
            cj_data: 接触关节数据
            grf_data: 地面反作用力数据
        """
        self.send_queue.put((pose_data, tran_data, cj_data, grf_data))

    def receive_data(self, timeout: Optional[float] = None) -> Optional[Tuple]:
        """
        从接收队列中获取数据
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            解析后的数据元组或None（如果超时）
        """
        try:
            return self.receive_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        """
        关闭连接
        """
        self.running = False
        
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            
        print("连接已关闭")

    def init_pybullet_visualization(self):
        """
        初始化PyBullet可视化环境
        """
        mu = 0.6
        supp_poly_size = 0.2
        self.model = RBDLModel(paths.physics_model_file, update_kinematics_by_hand=True)
        self.params = read_debug_param_values_from_json(paths.physics_parameter_file)
        self.friction_constraint_matrix = np.array([[np.sqrt(2), -mu, 0],
                                               [-np.sqrt(2), -mu, 0],
                                               [0, -mu, np.sqrt(2)],
                                               [0, -mu, -np.sqrt(2)]])
        self.support_polygon = np.array([[-supp_poly_size / 2, 0, -supp_poly_size / 2],
                                    [supp_poly_size / 2, 0, -supp_poly_size / 2],
                                    [-supp_poly_size / 2, 0, supp_poly_size / 2],
                                    [supp_poly_size / 2, 0, supp_poly_size / 2]])

        p.connect(p.GUI)
        p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)

        # 设置默认相机位置（可选）
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,  # 相机距离
            cameraYaw=45,  # 水平旋转角度
            cameraPitch=-30,  # 垂直倾斜角度
            cameraTargetPosition=[0, 0, 0]  # 相机焦点位置
        )
        # 禁用重力，仅用于可视化
        #p.setGravity(0, 0, 0)
        # 禁用实时模拟，防止模型因物理效应而移动
        #p.setRealTimeSimulation(0)
        self.id_robot = p.loadURDF(paths.physics_model_file, [0, 0, 0], useFixedBase=False,
                              flags=p.URDF_MERGE_FIXED_LINKS)
        change_color(self.id_robot, [198 / 255, 238 / 255, 0, 1.0])
        #p.loadURDF(paths.plane_file, [0, -0.881, 0.0], [-0.7071068, 0, 0, 0.7071068])
        p.loadURDF(paths.plane_file, [0, 0.881, 0.0], [0, 0, 0, 1])
        load_debug_params_into_bullet_from_json(paths.physics_parameter_file)

    def update_visualization(self, pose_data: List[float], tran_data: List[float], is_Global: bool = False,
                            cj_data: Optional[List[int]] = None,
                            grf_data: Optional[List[float]] = None):
        """
        更新可视化显示
        
        Args:
            pose_data: 从Unity接收的姿态数据(216个值，表示24个3x3旋转矩阵)
            tran_data: 位移数据
            cj_data: 接触关节数据
            grf_data: 地面反作用力数据
        """
        # 检查pose_data是否为216个值(24个关节的3x3矩阵)
        if len(pose_data) == 216:
            # 将216个值转换为24个3x3矩阵
            rotation_matrices = []
            for i in range(24):
                matrix = np.array(pose_data[i*9:(i+1)*9]).reshape(3, 3)
                rotation_matrices.append(matrix)
            
            # 第一个矩阵是全局根关节旋转，其余23个是相对父关节的旋转
            # 构造完整的SMPL格式姿态数据 [n, 24, 3, 3]
            poses = np.array(rotation_matrices).reshape(1, 24, 3, 3)
            
            # # 对左肩关节(索引16)的旋转矩阵取反
            # # 创建绕Y轴180度的旋转矩阵来实现取反效果
            # rotation_180_y = np.array([[-1, 0, 0],
            #                            [0, 1, 0],
            #                            [0, 0, -1]])
            # poses[0, 16] = np.dot(rotation_180_y, poses[0, 16])
            left_map = [1,0,2]
            right_map = [1,0,2]
            # 应用自定义坐标轴映射到肩膀关节
            poses[0] = self.apply_shoulder_axis_mapping(poses[0], left_map, right_map)
            
            # 打印肩关节及手臂关节的旋转矩阵
            #self.print_arm_rotation_matrices(poses[0])
            # 转换为RBDL格式的关节角度
            q = smpl_to_rbdl(poses, np.array(tran_data))[0]
            
            # 打印转换后的RBDL关节角度
            # self.print_rbdl_joint_angles(q)
            # 打印肩关节欧拉角
            self.print_shoulder_euler_angles(poses[0])
        else:
            # 兼容旧的72个值格式(轴角表示)
            q = np.concatenate([tran_data, pose_data])
        
        # 应用姿态到机器人
        set_pose(self.id_robot, q)

    def apply_shoulder_axis_mapping(self, poses, left_mapping=None, right_mapping=None):
        """
        应用自定义坐标轴映射到肩膀关节
        
        Args:
            poses: SMPL格式的姿态数据 [24, 3, 3]
            left_mapping: 左肩坐标轴映射，例如 [0, 1, 2] 表示保持原样，[1, 0, 2] 表示交换X和Y轴
                         也可以使用负数表示反转该轴，例如 [0, 1, -2] 表示反转Z轴
            right_mapping: 右肩坐标轴映射，规则同left_mapping
            
        Returns:
            修改后的姿态数据 [24, 3, 3]
        """
        import numpy as np
        
        # 复制输入数据以避免修改原始数据
        modified_poses = poses.copy()
        
        # 默认映射为保持原样
        if left_mapping is None:
            left_mapping = [0, 1, 2]
        if right_mapping is None:
            right_mapping = [0, 1, 2]
            
        # 处理左肩 (索引16)
        left_shoulder_matrix = modified_poses[16]
        new_left_matrix = np.zeros((3, 3))
        for i in range(3):
            axis_index = abs(left_mapping[i])
            if left_mapping[i] >= 0:
                new_left_matrix[:, i] = left_shoulder_matrix[:, axis_index]
            else:
                new_left_matrix[:, i] = -left_shoulder_matrix[:, axis_index]
        modified_poses[16] = new_left_matrix
        
        # 处理右肩 (索引17)
        right_shoulder_matrix = modified_poses[17]
        new_right_matrix = np.zeros((3, 3))
        for i in range(3):
            axis_index = abs(right_mapping[i])
            if right_mapping[i] >= 0:
                new_right_matrix[:, i] = right_shoulder_matrix[:, axis_index]
            else:
                new_right_matrix[:, i] = -right_shoulder_matrix[:, axis_index]
        modified_poses[17] = new_right_matrix
        
        return modified_poses

    def set_joint_angles_in_world_frame(self, joint_indices, angles):
        """
        在世界坐标系下直接设置关节角度
        
        Args:
            joint_indices: 关节索引列表
            angles: 对应的关节角度列表（弧度）
        """
        # 确保输入是列表形式
        if not isinstance(joint_indices, list):
            joint_indices = [joint_indices]
        if not isinstance(angles, list):
            angles = [angles]
            
        # 确保两个列表长度相同
        assert len(joint_indices) == len(angles), "关节索引列表和角度列表长度必须相同"
        
        # 设置每个关节的角度
        for joint_index, angle in zip(joint_indices, angles):
            # PyBullet中的关节索引需要考虑从1开始（因为0是base）
            # 同时需要考虑_rbdl_to_bullet的映射关系
            if joint_index < len(_rbdl_to_bullet):
                bullet_joint_index = _rbdl_to_bullet[joint_index]
                p.resetJointState(self.id_robot, bullet_joint_index, angle)
            else:
                print(f"警告: 关节索引 {joint_index} 超出范围")

    def get_joint_angles_in_world_frame(self, joint_indices):
        """
        获取世界坐标系下的关节角度
        
        Args:
            joint_indices: 关节索引列表
            
        Returns:
            对应的关节角度列表（弧度）
        """
        angles = []
        for joint_index in joint_indices:
            # PyBullet中的关节索引需要考虑从1开始（因为0是base）
            # 同时需要考虑_rbdl_to_bullet的映射关系
            if joint_index < len(_rbdl_to_bullet):
                bullet_joint_index = _rbdl_to_bullet[joint_index]
                joint_state = p.getJointState(self.id_robot, bullet_joint_index)
                angles.append(joint_state[0])  # joint_state[0]是关节位置
            else:
                print(f"警告: 关节索引 {joint_index} 超出范围")
                angles.append(None)
        return angles

    def print_arm_rotation_matrices(self, poses):
        """
        打印肩关节及以下手臂关节的旋转矩阵
        
        Args:
            poses: SMPL格式的姿态数据 [24, 3, 3]
        """
        # SMPL关节索引:
        # LCLAVICLE = 13, RCLAVICLE = 14
        # LSHOULDER = 16, RSHOULDER = 17
        # LELBOW = 18, RELBOW = 19
        # LWRIST = 20, RWRIST = 21
        # LHAND = 22, RHAND = 23
        
        joint_names = ['LCLAVICLE', 'RCLAVICLE', 'LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHAND', 'RHAND']
        joint_indices = [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
        
        print("=" * 50)
        print("Arm Joints Rotation Matrices:")
        print("=" * 50)
        
        for name, idx in zip(joint_names, joint_indices):
            print(f"{name} (Joint {idx}):")
            print(f"  {poses[idx, 0, :]}")
            print(f"  {poses[idx, 1, :]}")
            print(f"  {poses[idx, 2, :]}")
            print()
        
        print("=" * 50)
        print("=" * 50)

    def print_shoulder_euler_angles(self, poses):
        """
        打印左肩和右肩的欧拉角
        
        Args:
            poses: SMPL格式的姿态数据 [24, 3, 3]
        """
        # 根据SMPL关节定义:
        # LSHOULDER = 16
        # RSHOULDER = 17
        
        print("=" * 50)
        print("Shoulder Joint Rotation Matrices:")
        print("=" * 50)
        
        # 提取左肩旋转矩阵
        left_shoulder_rotation = poses[16]  # 3x3矩阵
        print(f"Left Shoulder (Index 16):")
        print(f"  {left_shoulder_rotation[0]}")
        print(f"  {left_shoulder_rotation[1]}")
        print(f"  {left_shoulder_rotation[2]}")
        
        # 提取右肩旋转矩阵
        right_shoulder_rotation = poses[17]  # 3x3矩阵
        print(f"Right Shoulder (Index 17):")
        print(f"  {right_shoulder_rotation[0]}")
        print(f"  {right_shoulder_rotation[1]}")
        print(f"  {right_shoulder_rotation[2]}")
        
        print("=" * 50)

    def add_shoulder_rotation(self, poses, left_rotation=None, right_rotation=None):
        """
        对肩膀关节添加额外的旋转
        
        Args:
            poses: SMPL格式的姿态数据 [24, 3, 3]
            left_rotation: 左肩额外旋转的欧拉角 (ZYX顺序) [3] 或旋转矩阵 [3, 3]
            right_rotation: 右肩额外旋转的欧拉角 (ZYX顺序) [3] 或旋转矩阵 [3, 3]
            
        Returns:
            修改后的姿态数据 [24, 3, 3]
        """
        import articulate as art
        import numpy as np
        
        # 复制输入数据以避免修改原始数据
        modified_poses = poses.copy()
        
        # 处理左肩旋转
        if left_rotation is not None:
            if isinstance(left_rotation, (list, tuple)):
                left_rotation = np.array(left_rotation)
                
            # 如果输入是欧拉角，则转换为旋转矩阵
            if left_rotation.shape == (3,):
                left_rotation_matrix = art.math.euler_angle_to_rotation_matrix_np(left_rotation, 'ZYX')
            elif left_rotation.shape == (3, 3):
                left_rotation_matrix = left_rotation
            else:
                raise ValueError("左肩旋转参数必须是形状为(3,)的欧拉角或形状为(3,3)的旋转矩阵")
            
            # 将额外旋转应用到左肩（矩阵相乘）
            modified_poses[16] = np.dot(left_rotation_matrix, modified_poses[16])
        
        # 处理右肩旋转
        if right_rotation is not None:
            if isinstance(right_rotation, (list, tuple)):
                right_rotation = np.array(right_rotation)
                
            # 如果输入是欧拉角，则转换为旋转矩阵
            if right_rotation.shape == (3,):
                right_rotation_matrix = art.math.euler_angle_to_rotation_matrix_np(right_rotation, 'ZYX')
            elif right_rotation.shape == (3, 3):
                right_rotation_matrix = right_rotation
            else:
                raise ValueError("右肩旋转参数必须是形状为(3,)的欧拉角或形状为(3,3)的旋转矩阵")
            
            # 将额外旋转应用到右肩（矩阵相乘）
            modified_poses[17] = np.dot(right_rotation_matrix, modified_poses[17])
        
        return modified_poses

# 使用示例
if __name__ == "__main__":
    # 创建连接器实例
    is_global = False
    connector = UnityConnector()
    connector.init_pybullet_visualization()
    # 检查命令行参数，如果提供了test参数则运行测试姿态
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("运行测试姿态...")
        connector.test_poses()
        connector.close()
    else:
        # 作为服务器启动（等待Unity连接）
        if connector.connect_as_server():
            try:
                frame_count = 0
                while connector.running:  # 修改为检查running状态
                    # 接收数据并进行可视化处理
                    received = connector.receive_data(timeout=0.01)  # 非阻塞检查
                    if received:
                        pose, tran, cj, grf = received
                        print(f"接收到数据: pose长度={len(pose)}, tran长度={len(tran)}")
                        # 调用PyBullet进行可视化
                        connector.update_visualization(pose, tran, is_global)
                    else:
                        # 如果没有接收到数据，使用测试数据进行可视化
                        # T-pose测试数据
                        test_pose = [0.0] * 72  # T-pose，所有关节角度为0
                        test_tran = [0.0, 0.0, 0.0]  # 根位置在原点
                        #connector.visualize_data(test_pose, test_tran)
                    
                    frame_count += 1
                    time.sleep(1/60)  # 60 FPS
                    
            except KeyboardInterrupt:
                print("\n停止通信")
            finally:
                connector.close()
        else:
            # 如果无法连接到Unity，运行测试姿态
            print("无法连接到Unity，运行测试姿态...")
            connector.test_poses()
            connector.close()