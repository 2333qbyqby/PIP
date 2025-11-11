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


class UnityConnector:
    """
    Unity连接器，用于与Unity进行双向通信
    """

    def __init__(self, host='127.0.0.1', port=8888, max_receive_queue_size=100):
        """
        初始化Unity连接器
        
        Args:
            host: 服务器地址
            port: 服务器端口
            max_receive_queue_size: 接收队列最大大小，防止数据积压
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connection = None
        
        # 队列用于存储待发送的数据和已接收的数据
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()
        self.max_receive_queue_size = max_receive_queue_size  # 最大队列大小
        self.receive_queue_dropped_count = 0  # 丢弃的数据包计数
        
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
        
        # Physics Optimizer用于物理优化
        self.physics_optimizer = None

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
                    
                pose_data, tran_data, grf_data, cj_data= data
                # 使用_pack_unity_data函数封装数据
                message = self._pack_unity_data(pose_data, tran_data,grf_data,cj_data)
                # 发送封装好的数据
                success = self._send_message(message)
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

    def _send_message(self, message: str) -> bool:
        """
        发送消息到Unity
        
        Args:
            message: 要发送的消息字符串
            
        Returns:
            bool: 发送是否成功
        """
        try:
            self.connection.send(message.encode('utf8'))
            return True
        except Exception as e:
            print(f"发送消息失败: {e}")
            return False

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
                            # 检查队列大小，防止积压过多数据
                            if self.receive_queue.qsize() >= self.max_receive_queue_size:
                                # 队列已满，丢弃最旧的数据
                                try:
                                    self.receive_queue.get_nowait()
                                    self.receive_queue_dropped_count += 1
                                    if self.receive_queue_dropped_count % 10 == 1:  # 每10次丢弃打印一次日志
                                        print(f"警告: 接收队列已满，已丢弃 {self.receive_queue_dropped_count} 个数据包")
                                except queue.Empty:
                                    pass
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

    def _pack_unity_data(self, pose_data: List[float], tran_data: List[float],
                         grf_data: Optional[List[float]] = None,
                         cj_data: Optional[List[int]] = None) -> str:
        """
        封装发送到Unity的数据
        
        Args:
            pose_data: 姿态数据
            tran_data: 位移数据
            cj_data: 接触关节数据
            grf_data: 地面反作用力数据
            
        Returns:
            str: 封装后的数据字符串
        """
        # 格式化数据，确保将numpy数组转换为Python标量
        pose_str = ','.join(['%g' % float(v) for v in pose_data])
        tran_str = ','.join(['%g' % float(v) for v in tran_data])
        grf_str = ','.join(['%g' % float(v) for v in grf_data]) if grf_data else ''
        print(grf_data)
        # 构造消息字符串
        message = f"{pose_str}#{tran_str}#{grf_str}$"
        return message

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
        
        # 解析接触数据 (cj_data)
        cj_data = [int(x) for x in cj_str.split(',')] if cj_str else []

        # 解析地面反作用力数据 (grf_data)
        velocity_data = [float(x) for x in grf_str.split(',')] if grf_str else []
        
        return pose_data, tran_data, cj_data, velocity_data

    def send_data(self, pose_data: List[float], tran_data: List[float],
                grf_data:List[float],
                cj_data: Optional[List[int]] = None):
        """
        将数据添加到发送队列
        
        Args:
            pose_data: 姿态数据
            tran_data: 位移数据
            grf_data: 地面反作用力数据
            cj_data: 接触关节数据
        """
        self.send_queue.put((pose_data, tran_data, grf_data, cj_data))

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
            cameraYaw=0,  # 水平旋转角度，0表示正后方
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

    def update_visualization(self, pose_data: List[float], tran_data: List[float],
                            cj_data: Optional[List[int]] = None,
                            grf_data: Optional[List[float]] = None):
        """
        更新可视化显示
        
        Args:
            pose_data: 从Unity接收的姿态数据(216个值，表示24个3x3旋转矩阵)或全局关节位置(72个值)
            tran_data: 位移数据 和 角色朝向
            cj_data: 接触关节数据
            grf_data: 地面反作用力数据
        """

        # 检查pose_data是否为216个值(24个关节的3x3矩阵)
        if len(pose_data) == 216:
            # 将216个值转换为24个3x3矩阵
            rotation_matrices = []
            for i in range(24):
                matrix = np.array(pose_data[i * 9:(i + 1) * 9]).reshape(3, 3)
                rotation_matrices.append(matrix)
            poses = np.array(rotation_matrices).reshape(1, 24, 3, 3)
            # 转换为RBDL格式的关节角度
            q = smpl_to_rbdl(poses, np.array(tran_data))[0]
        else:
            # 兼容旧的72个值格式(轴角表示)
            q = np.concatenate([tran_data, pose_data])

        # 应用姿态到机器人
        set_pose(self.id_robot, q)

    def get_receive_queue_info(self):
        """
        获取接收队列信息
        
        Returns:
            dict: 包含队列大小、最大容量和丢弃计数的信息
        """
        return {
            'queue_size': self.receive_queue.qsize(),
            'max_size': self.max_receive_queue_size,
            'dropped_count': self.receive_queue_dropped_count
        }

    def init_physics_optimizer(self, debug=False):
        """
        初始化物理优化器
        
        Args:
            debug: 是否启用调试模式
        """
        from dynamics import PhysicsOptimizer
        self.physics_optimizer = PhysicsOptimizer(debug=debug)

    def optimize_frame_with_physics(self, pose_data: List[float], jvel: List[float], contact: List[int], acc: List[float] = None):
        """
        使用物理优化器优化单帧数据
        
        Args:
            pose: 姿态数据
            jvel: 关节速度数据
            contact: 接触信息数据
            acc: 加速度数据
            
        Returns:
            tuple: (优化后的姿态, 优化后的位移)
        """
        poses = np.array([])
        velocitys = np.array([])
        contacts = np.array([0, 0])  # 默认值，表示没有接触
        # 检查pose_data是否为216个值(24个关节的3x3矩阵)
        if len(pose_data) == 216:
            # 将216个值转换为24个3x3矩阵
            rotation_matrices = []
            for i in range(24):
                matrix = np.array(pose_data[i * 9:(i + 1) * 9]).reshape(3, 3)
                rotation_matrices.append(matrix)
            poses = np.array(rotation_matrices).reshape(1, 24, 3, 3)
        if len(jvel) == 72:
            velocitys = np.array(jvel).reshape(24, 3)
        if len(contact) == 2 :
            contacts = np.array(contact).reshape(2)
        if self.physics_optimizer is None:
            raise RuntimeError("Physics optimizer not initialized. Call init_physics_optimizer() first.")
        
        # 添加调试信息
        # print("=== UnityConnector Debug Info ===")
        # print(f"poses shape: {poses.shape if hasattr(poses, 'shape') else 'N/A'}")
        # print(f"velocitys shape: {velocitys.shape if hasattr(velocitys, 'shape') else 'N/A'}")
        # print(f"contacts: {contacts}")
        return self.physics_optimizer.optimize_frame(poses, velocitys, contacts, acc)

# 使用示例
if __name__ == "__main__":
    # 创建连接器实例
    connector = UnityConnector()
    #connector.init_pybullet_visualization()
    connector.init_physics_optimizer(True)
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
                        pose, tran, cj, velocity = received
                        print(f"接收到数据: pose长度={len(pose)}, tran长度={len(tran)}")
                        # 调用PyBullet进行可视化
                        #connector.update_visualization(pose, tran, )
                        result = connector.optimize_frame_with_physics(pose, velocity, cj)
                        # 将result加入发送队列，并将结果发送给Unity
                        # 展开姿态数据为一维列表
                        pose_data = art.math.rotation_matrix_to_axis_angle(result[0]).flatten().tolist()
                        tran_data = result[1].tolist()
                        if(len(result) >2):
                            grf_data = result[2].flatten().tolist()
                        else:
                            grf_data = [0.0]*12
                        connector.send_data(pose_data, tran_data, grf_data)
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