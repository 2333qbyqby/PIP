from __future__ import annotations

from typing import Any, List, Optional


class PhysicsOptimizerAdapter:
    """
    把 UnityConnector 里“输入整理/容错/帧计数”从主类剥离出来。
    """

    def __init__(self):
        self._optimizer = None
        self.current_frame = 0

    def init(self, debug: bool = False):
        from dynamics import PhysicsOptimizer  # 延迟导入，避免 import unity_connector 时加载重模块

        self._optimizer = PhysicsOptimizer(debug=debug)
        self.current_frame = 0

    @property
    def is_ready(self) -> bool:
        return self._optimizer is not None

    def optimize_frame(
        self,
        pose_data: List[float],
        jvel: List[float],
        contact: Any,
        acc: Optional[List[float]] = None,
    ):
        import numpy as np  # 延迟导入：仅在启用物理优化时需要

        if self._optimizer is None:
            raise RuntimeError("Physics optimizer not initialized. Call init_physics_optimizer() first.")

        self.current_frame += 1

        poses = np.array([])
        velocitys = np.array([])
        # contact：仅支持
        # - 新结构：dict/json（全关节 c+p4），交由 dynamics.py 解析
        contact_payload: Any
        if contact is None or isinstance(contact, dict):
            contact_payload = contact
        else:
            processed_contact = [float(c) for c in contact]
            if len(processed_contact) == 10:
                raise ValueError("已移除旧格式 contact=10个float（2+8）。请改用 dict/json 新结构。")
            contact_payload = processed_contact
        ext_force_payload = None
        acc_payload = acc
        if isinstance(contact, dict):
            ext_force_payload = contact.get('ext_force')
            if acc is None:
                acc_payload = contact.get('acc')
        if len(pose_data) == 216:
            rotation_matrices = []
            for i in range(24):
                matrix = np.array(pose_data[i * 9:(i + 1) * 9]).reshape(3, 3)
                rotation_matrices.append(matrix)
            poses = np.array(rotation_matrices).reshape(1, 24, 3, 3)

        if len(jvel) == 72:
            velocitys = np.array(jvel).reshape(24, 3)

        return self._optimizer.optimize_frame(poses, velocitys, contact_payload, acc)


