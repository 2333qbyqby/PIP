import time
from typing import Any, List, Optional, Tuple

import pathlib
import sys
import argparse

# 添加项目根目录到Python路径（兼容直接运行本文件）
project_root = pathlib.Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from unity_connector_codec import UnityMessageCodec
from unity_connector_physics import PhysicsOptimizerAdapter
from unity_connector_transport import SocketTransport, ThreadedQueueBridge


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
        self.port = int(port)

        self.codec = UnityMessageCodec()
        self.transport = SocketTransport(self.host, self.port)

        # 网络收发桥（包含send/recv队列与线程）
        self._bridge = ThreadedQueueBridge(
            transport=self.transport,
            codec=self.codec,
            max_receive_queue_size=max_receive_queue_size,
            on_disconnect=self._handle_disconnect,
            enable_tau_debug_log=False,
        )

        # 兼容旧属性名（外部可能会直接访问）
        self.send_queue = self._bridge.send_queue
        self.receive_queue = self._bridge.receive_queue
        self.max_receive_queue_size = max_receive_queue_size
        # 注意：真实计数在 bridge 中；这里保留字段名以兼容外部直接读该属性
        self.receive_queue_dropped_count = 0

        # 可选组件
        self._physics = PhysicsOptimizerAdapter()

    def connect_as_server(self) -> bool:
        """
        作为服务器连接到Unity客户端
        
        Returns:
            bool: 连接是否成功
        """
        addr, ok = self.transport.listen_and_accept()
        if not ok:
            print("服务器连接失败")
            return False
        print(f"Unity通信服务器启动，监听 {self.host}:{self.port}")
        if addr is not None:
            print(f"Unity客户端已连接: {addr}")
        self._bridge.start()
        return True

    def connect_as_client(self) -> bool:
        """
        作为客户端连接到Unity服务器
        
        Returns:
            bool: 连接是否成功
        """
        ok = self.transport.connect()
        if not ok:
            print("客户端连接失败")
            return False
        print(f"已连接到Unity服务器 {self.host}:{self.port}")
        self._bridge.start()
        return True

    def _handle_disconnect(self):
        """
        处理连接断开事件
        """
        # on_disconnect 回调里尽量避免递归 close；这里只做一次幂等关闭
        print("检测到连接断开，正在关闭连接...")
        self.close()

    def send_data(self, pose_data: List[float], tran_data: List[float],
                  grf_data: Any,
                  tau_data: Optional[List[float]] = None):
        """
        将数据添加到发送队列
        
        Args:
            pose_data: 姿态数据
            tran_data: 位移数据
            grf_data: 地面反作用力数据
            tau_data: 虚拟力数据
        """
        self.send_queue.put((pose_data, tran_data, grf_data, tau_data))

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
        except Exception:
            return None

    @property
    def running(self) -> bool:
        return self._bridge.running

    def close(self):
        """
        关闭连接
        """
        self._bridge.stop()
        self.transport.close()
        print("连接已关闭")

    def get_receive_queue_info(self):
        """
        获取接收队列信息
        
        Returns:
            dict: 包含队列大小、最大容量和丢弃计数的信息
        """
        info = self._bridge.get_receive_queue_info()
        # 同步兼容字段（外部可能直接读 self.receive_queue_dropped_count / self.max_receive_queue_size）
        self.receive_queue_dropped_count = info.dropped_count
        self.max_receive_queue_size = info.max_size
        return {"queue_size": info.queue_size, "max_size": info.max_size, "dropped_count": info.dropped_count}

    def init_physics_optimizer(self, debug=False):
        """
        初始化物理优化器
        
        Args:
            debug: 是否启用调试模式
        """
        self._physics.init(debug=debug)

    def optimize_frame_with_physics(self, pose_data: List[float], jvel: List[float], contact: Any, acc: List[float] = None):
        """
        使用物理优化器优化单帧数据
        
        Args:
            pose: 姿态数据
            jvel: 关节速度数据
            contact: 接触信息数据（兼容多种结构）
                - 旧结构：2个float（左右脚接触程度）
                - 新结构：dict/json（全关节：每关节1个强度c + 4点mask）
            acc: 加速度数据
            
        Returns:
            tuple: (优化后的姿态, 优化后的位移)
        """
        print(f"当前优化帧数{self._physics.current_frame + 1}")
        return self._physics.optimize_frame(pose_data, jvel, contact, acc)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unity <-> Python socket connector (PIP).")
    p.add_argument("--host", default="127.0.0.1", help="Bind/connect host. Use 0.0.0.0 to accept remote clients.")
    p.add_argument("--port", type=int, default=8888, help="TCP port.")
    p.add_argument(
        "--mode",
        choices=["server", "client"],
        default="server",
        help="server: Python listens and Unity connects; client: Python connects to Unity.",
    )
    p.add_argument("--max-recv-queue", type=int, default=100, help="Max receive queue size (drop oldest when full).")
    p.add_argument("--fps", type=float, default=60.0, help="Main loop target FPS when idle.")
    p.add_argument("--physics-debug", action="store_true", help="Enable physics optimizer debug mode.")
    p.add_argument("--disable-physics", action="store_true", help="Run without physics optimizer (echo pose/tran).")
    return p


def _run_forever(args: argparse.Namespace) -> int:
    import articulate as art
    import numpy as np

    connector = UnityConnector(host=args.host, port=args.port, max_receive_queue_size=args.max_recv_queue)

    if not args.disable_physics:
        connector.init_physics_optimizer(debug=bool(args.physics_debug))

    ok = connector.connect_as_server() if args.mode == "server" else connector.connect_as_client()
    if not ok:
        return 2

    dt = 1.0 / max(1.0, float(args.fps))
    try:
        while connector.running:
            received = connector.receive_data(timeout=0.01)
            if not received:
                time.sleep(dt)
                continue

            pose, tran, contact, velocity = received

            if args.disable_physics:
                # Echo back: convert incoming 24x3x3 rotation matrices (216 floats) -> axis-angle (72 floats)
                if len(pose) == 216:
                    mats = np.array(pose, dtype=np.float32).reshape(24, 3, 3)
                    pose_out = art.math.rotation_matrix_to_axis_angle(mats).reshape(-1).tolist()
                else:
                    pose_out = list(pose)
                tran_out = list(tran) if tran is not None else [0.0, 0.0, 0.0]
                grf_out = [0.0] * 24
                tau_out = [0.0] * 6
                connector.send_data(pose_out, tran_out, grf_out, tau_out)
                continue

            result = connector.optimize_frame_with_physics(pose, velocity, contact)

            pose_out = art.math.rotation_matrix_to_axis_angle(result[0]).flatten().tolist()
            tran_out = result[1].tolist()

            if len(result) > 2 and isinstance(result[2], dict):
                grf_out = result[2]
            else:
                grf_out = [0.0] * 24

            tau_out = result[3] if len(result) > 3 else [0.0] * 6
            connector.send_data(pose_out, tran_out, grf_out, tau_out)
    except KeyboardInterrupt:
        print("\n停止通信")
    finally:
        connector.close()
    return 0


# 使用示例
if __name__ == "__main__":
    parser = _build_arg_parser()
    exit_code = _run_forever(parser.parse_args())
    raise SystemExit(exit_code)