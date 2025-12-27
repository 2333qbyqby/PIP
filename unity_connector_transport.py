from __future__ import annotations

import queue
import socket
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

from unity_connector_codec import UnityMessageCodec, UnityOutgoingFrame


OnDisconnect = Callable[[], None]


@dataclass
class ReceiveQueueInfo:
    queue_size: int
    max_size: int
    dropped_count: int


class SocketTransport:
    """
    对 socket 的薄封装，便于单测与替换（例如换成UDP/WebSocket）.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)
        self.socket: Optional[socket.socket] = None
        self.connection: Optional[socket.socket] = None

    def listen_and_accept(self) -> Tuple[Optional[Tuple[str, int]], bool]:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            conn, addr = self.socket.accept()
            self.connection = conn
            return addr, True
        except Exception:
            return None, False

    def connect(self) -> bool:
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.host, self.port))
            return True
        except Exception:
            return False

    def send_text(self, message: str) -> bool:
        try:
            if not self.connection:
                return False
            self.connection.sendall(message.encode("utf8"))
            return True
        except Exception:
            return False

    def recv_text(self, bufsize: int = 4096) -> Optional[str]:
        try:
            if not self.connection:
                return None
            chunk = self.connection.recv(bufsize)
            if not chunk:
                return None
            return chunk.decode("utf-8")
        except Exception:
            return None

    def close(self):
        if self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None


class ThreadedQueueBridge:
    """
    负责：
    - send_queue -> codec.pack -> socket.send
    - socket.recv -> codec.split+parse -> receive_queue（带丢包策略）
    """

    def __init__(
        self,
        transport: SocketTransport,
        codec: UnityMessageCodec,
        max_receive_queue_size: int = 100,
        on_disconnect: Optional[OnDisconnect] = None,
        enable_tau_debug_log: bool = False,
    ):
        self.transport = transport
        self.codec = codec
        self.max_receive_queue_size = int(max_receive_queue_size)
        self.on_disconnect = on_disconnect
        self.enable_tau_debug_log = enable_tau_debug_log

        self.send_queue: "queue.Queue[Union[UnityOutgoingFrame, tuple]]" = queue.Queue()
        # 兼容旧行为：receive_queue 里放 (pose, tran, contact, velocity) 四元组
        self.receive_queue: "queue.Queue[tuple]" = queue.Queue()
        self.receive_queue_dropped_count = 0
        self._dropped_log_every = 10

        self._send_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._send_thread.start()
        self._recv_thread.start()

    def stop(self):
        self._running = False

    def get_receive_queue_info(self) -> ReceiveQueueInfo:
        return ReceiveQueueInfo(
            queue_size=self.receive_queue.qsize(),
            max_size=self.max_receive_queue_size,
            dropped_count=self.receive_queue_dropped_count,
        )

    def _handle_disconnect(self):
        if self._running:
            self._running = False
        if self.on_disconnect:
            try:
                self.on_disconnect()
            except Exception:
                pass

    def _send_loop(self):
        while self._running:
            try:
                item = self.send_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                continue

            try:
                # 兼容旧结构： (pose, tran, grf, tau)
                if isinstance(item, tuple):
                    pose_data, tran_data, grf_data, tau_data = item
                    frame = UnityOutgoingFrame(
                        pose=list(pose_data),
                        tran=list(tran_data),
                        grf=grf_data,
                        tau=list(tau_data) if tau_data is not None else None,
                    )
                else:
                    frame = item

                if self.enable_tau_debug_log and frame.tau is not None:
                    # 避免依赖 numpy，这里只做轻量日志
                    print(f"发送的 tau 数据: {list(frame.tau)}")
                    print(f"tau 数据维度: {len(frame.tau)}")

                message = self.codec.pack_outgoing(frame.pose, frame.tran, frame.grf, tau_data=frame.tau)
                ok = self.transport.send_text(message)
                if not ok:
                    self._handle_disconnect()
                    break
            except Exception:
                self._handle_disconnect()
                break

    def _recv_loop(self):
        buffer = ""
        while self._running:
            chunk = self.transport.recv_text(4096)
            if not chunk:
                self._handle_disconnect()
                break

            buffer += chunk
            packets, buffer = self.codec.split_packets(buffer)
            for packet in packets:
                try:
                    frame = self.codec.parse_incoming(packet)
                except ValueError:
                    continue

                if self.receive_queue.qsize() >= self.max_receive_queue_size:
                    try:
                        self.receive_queue.get_nowait()
                        self.receive_queue_dropped_count += 1
                        # 保持原始语义：每10次丢弃打印一次（第1,11,21...次）
                        if self.receive_queue_dropped_count % self._dropped_log_every == 1:
                            print(f"警告: 接收队列已满，已丢弃 {self.receive_queue_dropped_count} 个数据包")
                    except queue.Empty:
                        pass
                self.receive_queue.put((frame.pose, frame.tran, frame.contact, frame.velocity))


