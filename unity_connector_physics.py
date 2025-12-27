from __future__ import annotations

from typing import List, Optional


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
        contact: List[float],
        acc: Optional[List[float]] = None,
    ):
        import numpy as np  # 延迟导入：仅在启用物理优化时需要

        if self._optimizer is None:
            raise RuntimeError("Physics optimizer not initialized. Call init_physics_optimizer() first.")

        self.current_frame += 1

        poses = np.array([])
        velocitys = np.array([])
        contacts = np.array([0.0, 0.0], dtype=np.float32)

        processed_contact = [float(c) for c in contact] if contact is not None else []

        if len(pose_data) == 216:
            rotation_matrices = []
            for i in range(24):
                matrix = np.array(pose_data[i * 9:(i + 1) * 9]).reshape(3, 3)
                rotation_matrices.append(matrix)
            poses = np.array(rotation_matrices).reshape(1, 24, 3, 3)

        if len(jvel) == 72:
            velocitys = np.array(jvel).reshape(24, 3)

        if len(processed_contact) == 2:
            contacts = np.array(processed_contact, dtype=np.float32).reshape(2)
        elif len(processed_contact) == 10:
            contacts = np.array(processed_contact, dtype=np.float32).reshape(10)
        elif len(processed_contact) > 0:
            padded = (processed_contact + [0.0, 0.0])[:2]
            contacts = np.array(padded, dtype=np.float32).reshape(2)

        return self._optimizer.optimize_frame(poses, velocitys, contacts, acc)


