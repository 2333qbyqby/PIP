"""
按帧记录 Unity -> Python 送来的“参考姿态” q（本仓库里对应 dynamics.PhysicsOptimizer.optimize_frame() 里的 q_ref）。

特点：
- 追加写入（CSV），不需要把所有帧攒在内存里
- 默认每次进程启动会覆盖同名文件（第一次写入用 wb，后续用 ab）

文件格式：
- CSV：每一行是一帧
  frame_idx,time,q0,q1,...,qN
"""

from __future__ import annotations

import os
import time
import csv
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np


DEFAULT_Q_RECORD_PATH = os.environ.get("PIP_Q_RECORD_PATH", "data/result/unity_q_ref_frames.csv")


class _QFrameRecorder:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fp = None  # lazy open
        self._writer = None
        self._nq: Optional[int] = None
        self._opened = False

    def append(self, frame_idx: int, q_ref: Any) -> None:
        q_arr = np.asarray(q_ref).reshape(-1)
        with self._lock:
            if not self._opened:
                # 每次进程启动第一次写入覆盖旧文件，避免混入上次运行的数据
                # newline="" 防止 Windows 下 csv 出现空行
                self._fp = open(self.path, "w", newline="", encoding="utf-8")
                self._writer = csv.writer(self._fp)
                self._nq = int(q_arr.size)
                header = ["frame_idx", "time"] + [f"q{i}" for i in range(self._nq)]
                self._writer.writerow(header)
                self._opened = True

            # 维度不一致时：尽量写入（padding/truncate），但不抛错卡住主流程
            nq = int(self._nq or q_arr.size)
            row_q = q_arr.tolist()
            if len(row_q) < nq:
                row_q = row_q + [0.0] * (nq - len(row_q))
            elif len(row_q) > nq:
                row_q = row_q[:nq]

            self._writer.writerow([int(frame_idx), float(time.time()), *row_q])
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                finally:
                    self._fp = None
                    self._writer = None
                    self._opened = False
                    self._nq = None


_global_recorder: Optional[_QFrameRecorder] = None


def record_q_frame(q_ref: Any, frame_idx: int, path: str = DEFAULT_Q_RECORD_PATH) -> None:
    """记录一帧 q_ref 到文件（默认：data/result/unity_q_ref_frames.csv）。"""
    global _global_recorder
    if _global_recorder is None or str(_global_recorder.path) != str(Path(path)):
        _global_recorder = _QFrameRecorder(path)
    _global_recorder.append(frame_idx=frame_idx, q_ref=q_ref)


def iter_q_records(path: str = DEFAULT_Q_RECORD_PATH) -> Iterator[Dict[str, Any]]:
    """读取记录文件（CSV）并逐帧 yield dict：{"frame_idx","time","q_ref"}。"""
    p = Path(path)
    with open(p, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        q_cols = [c for c in (reader.fieldnames or []) if c.startswith("q")]
        for row in reader:
            frame_idx = int(float(row.get("frame_idx", 0)))
            t = float(row.get("time", 0.0))
            q = np.array([float(row.get(c, 0.0)) for c in q_cols], dtype=np.float64)
            yield {"frame_idx": frame_idx, "time": t, "q_ref": q}


