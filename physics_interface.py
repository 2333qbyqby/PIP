import numpy as np
import torch
from dynamics import PhysicsOptimizer
from unity_connector import UnityConnector
import time


class PhysicsInterface:
    """
    物理优化接口，用于连接Unity和物理优化器
    """

    def __init__(self, host='127.0.0.1', port=8888, debug=False):
        """
        初始化物理优化接口
        
        Args:
            host: 服务器地址
            port: 服务器端口
            debug: 是否启用调试模式
        """
        self.connector = UnityConnector(host, port)
        self.physics_optimizer = PhysicsOptimizer(debug=debug)
        self.running = False

    def start_server(self):
        """
        启动服务器并等待Unity连接
        
        Returns:
            bool: 连接是否成功
        """
        return self.connector.connect_as_server()

    def process_data_loop(self):
        """
        处理数据的主循环
        """
        self.running = True
        print("开始处理数据...")
        
        try:
            while self.running and self.connector.running:
                # 从Unity接收数据
                received_data = self.connector.receive_data(timeout=0.1)
                
                if received_data:
                    # 解析接收到的数据
                    pose_data, tran_data, cj_data, grf_data = received_data
                    
                    # 将数据传递给物理优化器
                    pose_opt, tran_opt = self._process_physics_optimization(
                        pose_data, tran_data, cj_data, grf_data)
                    
                    # 将优化结果发送回Unity
                    self.connector.send_data(
                        pose_data=pose_opt.tolist(),
                        tran_data=tran_opt.tolist(),
                        cj_data=cj_data,  # 原样返回接触关节数据
                        grf_data=grf_data  # 原样返回地面反作用力数据
                    )
                    
                    print(f"处理了一帧数据: pose长度={len(pose_data)}, tran长度={len(tran_data)}")
                else:
                    # 如果没有接收到数据，短暂休眠以避免忙等待
                    time.sleep(0.001)
                    
        except KeyboardInterrupt:
            print("\n停止数据处理")
        except Exception as e:
            print(f"处理数据时出错: {e}")
        finally:
            self.stop()

    def _process_physics_optimization(self, pose_data, tran_data, cj_data, grf_data):
        """
        处理物理优化
        
        Args:
            pose_data: 姿态数据
            tran_data: 位移数据
            cj_data: 接触关节数据
            grf_data: 地面反作用力数据
            
        Returns:
            tuple: 优化后的姿态和位移数据
        """
        # 将输入数据转换为物理优化器所需的格式
        # pose_data: [72] -> [24, 3, 3]
        pose_tensor = torch.from_numpy(np.array(pose_data, dtype=np.float32)).view(24, 3, 3)
        
        # tran_data: [3] -> [3]
        tran_tensor = torch.from_numpy(np.array(tran_data, dtype=np.float32))
        
        # cj_data: [n] -> contact信息
        contact_tensor = torch.zeros(2)  # 默认值
        if cj_data:
            # 简化处理：假设cj_data中的关节索引对应左右脚
            if 10 in cj_data:  # LFOOT
                contact_tensor[0] = 1.0
            if 11 in cj_data:  # RFOOT
                contact_tensor[1] = 1.0
                
        # grf_data: [n*3] -> 加速度信息（简化处理）
        acc_tensor = torch.zeros(6, 3)  # 默认值
        if grf_data and len(grf_data) >= 3:
            # 将GRF数据作为加速度的近似值
            grf_array = np.array(grf_data)
            # 简化处理：只取前6个值作为加速度
            for i in range(min(6, len(grf_array)//3)):
                acc_tensor[i] = torch.from_numpy(grf_array[i*3:(i+1)*3])
        
        # 调用物理优化器
        pose_opt, tran_opt = self.physics_optimizer.optimize_frame(
            pose_tensor, 
            torch.zeros(24, 3),  # 简化处理：零速度
            contact_tensor, 
            acc_tensor
        )
        
        return pose_opt, tran_opt

    def stop(self):
        """
        停止处理并关闭连接
        """
        self.running = False
        self.connector.close()
        print("物理优化接口已关闭")


# 使用示例
if __name__ == "__main__":
    # 创建物理优化接口实例
    physics_interface = PhysicsInterface(debug=True)
    
    # 启动服务器并等待连接
    if physics_interface.start_server():
        try:
            # 开始处理数据
            physics_interface.process_data_loop()
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            physics_interface.stop()