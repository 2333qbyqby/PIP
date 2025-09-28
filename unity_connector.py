import socket
import threading
import queue
import time
from typing import List, Tuple, Optional


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


# 使用示例
if __name__ == "__main__":
    # 创建连接器实例
    connector = UnityConnector()
    
    # 作为服务器启动（等待Unity连接）
    if connector.connect_as_server():
        try:
            frame_count = 0
            while connector.running:  # 修改为检查running状态
                # 示例发送数据
                pose_data = [0.0] * 72  # SMPL pose参数 (24 joints * 3)
                tran_data = [0.0, 0.0, 0.0]  # 位移向量
                cj_data = [1, 2, 3]  # 接触关节索引
                grf_data = [0.0, 0.0, 0.0]  # 地面反作用力
                
                # 添加一些变化使动画可见
                pose_data[frame_count % 72] = frame_count * 0.01
                
                # 发送数据
                connector.send_data(pose_data, tran_data, cj_data, grf_data)
                
                # 检查是否有接收到的数据
                received = connector.receive_data(timeout=0.01)  # 非阻塞检查
                if received:
                    pose, tran, cj, grf = received
                    print(f"接收到数据: pose长度={len(pose)}, tran长度={len(tran)}")
                
                frame_count += 1
                time.sleep(1/60)  # 60 FPS
                
        except KeyboardInterrupt:
            print("\n停止通信")
        finally:
            connector.close()