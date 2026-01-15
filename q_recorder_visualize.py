"""
可视化回放 q_recorder 记录的 q（q_ref）：
- 读取 data/result/unity_q_ref_frames.csv（CSV）
- 用 pybullet GUI 加载 models/physics.urdf
- 逐帧调用 utils.set_pose() 把机器人摆到该帧的 q

用法：
    python q_recorder_visualize.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p

from config import paths
from q_recorder import DEFAULT_Q_RECORD_PATH, iter_q_records
from utils import set_pose
from articulate.utils.bullet import change_color


# =========================
# 直接在这里改默认配置即可
# =========================
RECORD_PATH = os.environ.get("PIP_Q_RECORD_PATH", DEFAULT_Q_RECORD_PATH)
FPS = 60.0
START_FRAME = 0          # 从第几帧开始回放（按 frame_idx）
END_FRAME: Optional[int] = None  # None 表示回放到文件结束
LOOP = True              # 到末尾后是否从头循环


def _load_scene() -> int:
    p.connect(p.GUI)
    p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
    id_robot = p.loadURDF(
        paths.physics_model_file,
        [0, 0, 0],
        useFixedBase=False,
        flags=p.URDF_MERGE_FIXED_LINKS,
    )
    try:
        change_color(id_robot, [198 / 255, 238 / 255, 0, 1.0])
    except Exception:
        pass

    # 和 dynamics.py 里的 debug 初始化保持一致
    try:
        p.loadURDF(paths.plane_file, [0, -0.881, 0.0], [-0.7071068, 0, 0, 0.7071068])
    except Exception:
        pass
    return int(id_robot)


def _replay_once(id_robot: int) -> None:
    record_file = Path(RECORD_PATH)
    if not record_file.exists():
        raise FileNotFoundError(f"找不到记录文件：{record_file}")

    dt = 1.0 / max(1.0, float(FPS))
    last_time = time.time()

    for rec in iter_q_records(str(record_file)):
        frame_idx = int(rec.get("frame_idx", -1))
        if frame_idx < int(START_FRAME):
            continue
        if END_FRAME is not None and frame_idx > int(END_FRAME):
            break

        q = np.asarray(rec["q_ref"]).reshape(-1)
        set_pose(id_robot, q)

        # 简单的定帧率播放（比 sleep(dt) 更稳一点）
        now = time.time()
        sleep_t = dt - (now - last_time)
        if sleep_t > 0:
            time.sleep(sleep_t)
        last_time = time.time()


def main() -> int:
    id_robot = _load_scene()
    try:
        while True:
            _replay_once(id_robot)
            if not LOOP:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            p.disconnect()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


