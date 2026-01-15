"""
一个“可直接运行”的 Unity 连接器入口脚本：
- 不读取命令行参数（不会因为 IDE 注入参数而 argparse 报错）
- 直接使用下面的默认配置启动（server/client、host、port 等）

用法：
    python unity_connector_run.py
"""

from __future__ import annotations

import argparse

from unity_connector import _run_forever


# =========================
# 直接在这里改默认配置即可
# =========================
HOST = "127.0.0.1"
PORT = 8888
MODE = "server"  # "server"：Python 监听，Unity 连接；"client"：Python 连接 Unity
MAX_RECV_QUEUE = 100
FPS = 60.0
PHYSICS_DEBUG = True
DISABLE_PHYSICS = False  # True 则不跑物理优化，只回传 pose/tran（便于联调）


def _default_args() -> argparse.Namespace:
    return argparse.Namespace(
        host=HOST,
        port=int(PORT),
        mode=str(MODE),
        max_recv_queue=int(MAX_RECV_QUEUE),
        fps=float(FPS),
        physics_debug=bool(PHYSICS_DEBUG),
        disable_physics=bool(DISABLE_PHYSICS),
    )


def main() -> int:
    args = _default_args()
    return int(_run_forever(args))


if __name__ == "__main__":
    raise SystemExit(main())


