from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Tuple, Union, Any


GrfPayload = Union[List[float], Dict[str, Any], None]


@dataclass(frozen=True)
class UnityIncomingFrame:
    pose: List[float]
    tran: List[float]
    contact: Any
    velocity: List[float]


@dataclass(frozen=True)
class UnityOutgoingFrame:
    pose: List[float]
    tran: List[float]
    grf: GrfPayload
    tau: Optional[List[float]] = None


class UnityMessageCodec:
    """
    Unity <-> Python 消息编解码（字符串协议）.

    发送（Python -> Unity）:
      pose#tran#grf#tau$

    接收（Unity -> Python）:
      pose#tran#contact#velocity$
    """

    DELIM_END = "$"
    DELIM_SECTION = "#"
    DELIM_FLOAT = ","

    def pack_outgoing(
        self,
        pose_data: List[float],
        tran_data: List[float],
        grf_data: GrfPayload = None,
        tau_data: Optional[List[float]] = None,
    ) -> str:
        pose_str = self.DELIM_FLOAT.join(["%g" % float(v) for v in (pose_data or [])])
        tran_str = self.DELIM_FLOAT.join(["%g" % float(v) for v in (tran_data or [])])

        # GRF：
        # - 旧格式：list[float]（逗号分隔浮点）
        # - 新格式：dict（发送为 JSON 字符串，包含 contacts/left_foot/right_foot 等结构化字段）
        if isinstance(grf_data, dict):
            # 注意：协议分段使用 '#'，结束符使用 '$'，JSON 中不包含这些字符，安全。
            # 为降低带宽，使用紧凑 separators；同时 ensure_ascii=False 保留中文/非ASCII（若有）。
            grf_str = json.dumps(grf_data, ensure_ascii=False, separators=(",", ":"))
        elif grf_data:
            grf_str = self.DELIM_FLOAT.join(["%g" % float(v) for v in grf_data])  # type: ignore[arg-type]
        else:
            grf_str = ""

        tau_str = (
            self.DELIM_FLOAT.join(["%g" % float(v) for v in tau_data])
            if tau_data is not None and len(tau_data) > 0
            else ""
        )

        return f"{pose_str}{self.DELIM_SECTION}{tran_str}{self.DELIM_SECTION}{grf_str}{self.DELIM_SECTION}{tau_str}{self.DELIM_END}"

    def parse_incoming(self, packet: str) -> UnityIncomingFrame:
        # 移除末尾结束符
        data = packet.rstrip(self.DELIM_END)
        parts = data.split(self.DELIM_SECTION)
        if len(parts) != 4:
            raise ValueError(f"数据格式错误：期望4个部分，实际收到{len(parts)}个部分")

        pose_str, tran_str, contact_str, velocity_str = parts
        pose = [float(x) for x in pose_str.split(self.DELIM_FLOAT)] if pose_str else []
        tran = [float(x) for x in tran_str.split(self.DELIM_FLOAT)] if tran_str else []
        # contact：推荐使用 JSON（dict/list）。CSV floats 仅用于极简旧结构（2 floats）。
        contact: Any
        contact_str_stripped = contact_str.strip()
        if contact_str_stripped and (contact_str_stripped[0] == "{" or contact_str_stripped[0] == "["):
            try:
                contact = json.loads(contact_str_stripped)
            except Exception as e:
                raise ValueError(f"contact JSON 解析失败：{e}")
        else:
            contact = [float(x) for x in contact_str.split(self.DELIM_FLOAT)] if contact_str else []
        velocity = [float(x) for x in velocity_str.split(self.DELIM_FLOAT)] if velocity_str else []
        return UnityIncomingFrame(pose=pose, tran=tran, contact=contact, velocity=velocity)

    def split_packets(self, buffer: str) -> Tuple[List[str], str]:
        """
        从拼接缓冲中拆出完整packet（每个以 '$' 结尾），返回 (packets, remaining_buffer).
        packets 中每个字符串都带 '$' 结尾，方便直接 parse_incoming().
        """
        packets: List[str] = []
        while self.DELIM_END in buffer:
            msg, buffer = buffer.split(self.DELIM_END, 1)
            if msg:
                packets.append(msg + self.DELIM_END)
        return packets, buffer


