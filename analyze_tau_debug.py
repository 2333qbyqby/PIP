import argparse
import csv
from dataclasses import dataclass
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TauSlices:
    # Uses the same slicing convention documented at the bottom of dynamics.py:
    # tau[0:6] root residual; then 3D torque per joint.
    root: Tuple[int, int] = (0, 6)
    # Left arm
    LSHOULDER: Tuple[int, int] = (42, 45)
    LELBOW: Tuple[int, int] = (45, 48)
    LWRIST: Tuple[int, int] = (48, 51)
    LHAND: Tuple[int, int] = (51, 54)
    # Right arm
    RSHOULDER: Tuple[int, int] = (57, 60)
    RELBOW: Tuple[int, int] = (60, 63)
    RWRIST: Tuple[int, int] = (63, 66)
    RHAND: Tuple[int, int] = (66, 69)

    def left_arm(self) -> Tuple[int, int]:
        return (self.LSHOULDER[0], self.LHAND[1])

    def right_arm(self) -> Tuple[int, int]:
        return (self.RSHOULDER[0], self.RHAND[1])


def _safe_ratio(a: float, b: float, eps: float = 1e-9) -> float:
    return float(a / (b + eps))


def read_tau_csv(
    path: str,
    *,
    delimiter: str = ",",
    has_header: bool = False,
    expected_tau_dim: Optional[int] = 75,
) -> Tuple[List[int], List[List[float]]]:
    """
    Returns (frame_ids, taus).
    - frame_ids: list[int]
    - taus: list[list[float]]
    """
    frames: List[int] = []
    taus: List[List[float]] = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for row_i, row in enumerate(reader, start=1):
            if not row:
                continue
            # allow whitespace
            row = [c.strip() for c in row if c.strip() != ""]
            if len(row) < 2:
                continue
            try:
                frame = int(float(row[0]))
            except ValueError as e:
                raise ValueError(f"Failed to parse frame id at line {row_i}: {row[0]!r}") from e
            try:
                tau = [float(x) for x in row[1:]]
            except ValueError as e:
                raise ValueError(f"Failed to parse tau floats at line {row_i}") from e

            if expected_tau_dim is not None and len(tau) != expected_tau_dim:
                raise ValueError(
                    f"Unexpected tau dim at line {row_i}: got {len(tau)}, expected {expected_tau_dim}. "
                    f"Tip: pass --expected-tau-dim to override."
                )
            frames.append(frame)
            taus.append(tau)

    if len(taus) == 0:
        raise ValueError("No valid rows loaded from CSV.")
    return frames, taus


def _l2_norm(seq: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in seq))


def _slice(seq: Sequence[float], a: int, b: int) -> Sequence[float]:
    return seq[a:b]


def compute_metrics(frames: List[int], taus: List[List[float]], slices: TauSlices) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    out["frame"] = [float(x) for x in frames]
    out["tau_l2"] = []
    out["tau_root_l2"] = []

    for name in [
        "LSHOULDER",
        "LELBOW",
        "LWRIST",
        "LHAND",
        "RSHOULDER",
        "RELBOW",
        "RWRIST",
        "RHAND",
        "tau_left_arm_l2",
        "tau_right_arm_l2",
        "arm_delta_l2",
        "arm_ratio_l2",
        "arm_diff_vec_l2",
    ]:
        out[name if name.startswith("tau_") or name.startswith("arm_") else f"tau_{name}_l2"] = []

    La, Lb = slices.left_arm()
    Ra, Rb = slices.right_arm()

    for tau in taus:
        out["tau_l2"].append(_l2_norm(tau))
        out["tau_root_l2"].append(_l2_norm(_slice(tau, slices.root[0], slices.root[1])))

        def add_joint(name: str, a: int, b: int) -> float:
            v = _l2_norm(_slice(tau, a, b))
            out[f"tau_{name}_l2"].append(v)
            return v

        add_joint("LSHOULDER", *slices.LSHOULDER)
        add_joint("LELBOW", *slices.LELBOW)
        add_joint("LWRIST", *slices.LWRIST)
        add_joint("LHAND", *slices.LHAND)
        add_joint("RSHOULDER", *slices.RSHOULDER)
        add_joint("RELBOW", *slices.RELBOW)
        add_joint("RWRIST", *slices.RWRIST)
        add_joint("RHAND", *slices.RHAND)

        left_arm = _l2_norm(_slice(tau, La, Lb))
        right_arm = _l2_norm(_slice(tau, Ra, Rb))
        out["tau_left_arm_l2"].append(left_arm)
        out["tau_right_arm_l2"].append(right_arm)
        out["arm_delta_l2"].append(left_arm - right_arm)
        out["arm_ratio_l2"].append(_safe_ratio(left_arm, right_arm))

        diff_vec = [tau[La + i] - tau[Ra + i] for i in range(Lb - La)]
        out["arm_diff_vec_l2"].append(_l2_norm(diff_vec))

    return out


def _percentile_sorted(xs_sorted: Sequence[float], p: float) -> float:
    if len(xs_sorted) == 0:
        return float("nan")
    if len(xs_sorted) == 1:
        return float(xs_sorted[0])
    p = max(0.0, min(100.0, float(p)))
    pos = (len(xs_sorted) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs_sorted[lo])
    w = pos - lo
    return float(xs_sorted[lo] * (1.0 - w) + xs_sorted[hi] * w)


def _percentiles(x: Sequence[float], ps: Iterable[float]) -> Dict[str, float]:
    xs = sorted(float(v) for v in x)
    return {f"p{int(p)}": _percentile_sorted(xs, p) for p in ps}


def print_summary(metrics: Dict[str, List[float]], top_k: int = 10) -> None:
    arm_delta = metrics["arm_delta_l2"]
    arm_ratio = metrics["arm_ratio_l2"]
    arm_diff = metrics["arm_diff_vec_l2"]

    def _line(name: str, arr: Sequence[float]) -> None:
        stats = _percentiles(arr, [0, 50, 90, 95, 99, 100])
        print(
            f"{name}: "
            f"min={stats['p0']:.3g}, "
            f"p50={stats['p50']:.3g}, "
            f"p90={stats['p90']:.3g}, "
            f"p95={stats['p95']:.3g}, "
            f"p99={stats['p99']:.3g}, "
            f"max={stats['p100']:.3g}"
        )

    print("=== tau_debug summary ===")
    _line("tau_l2", metrics["tau_l2"])
    _line("tau_root_l2", metrics["tau_root_l2"])
    _line("tau_left_arm_l2", metrics["tau_left_arm_l2"])
    _line("tau_right_arm_l2", metrics["tau_right_arm_l2"])
    _line("arm_delta_l2 (L-R)", arm_delta)
    _line("arm_ratio_l2 (L/R)", arm_ratio)
    _line("arm_diff_vec_l2 ||Lseg-Rseg||", arm_diff)

    # Show a few frames where left-right arm difference is most extreme
    idx = sorted(range(len(arm_delta)), key=lambda i: arm_delta[i], reverse=True)[: max(1, int(top_k))]
    print(f"\n=== top {len(idx)} frames by arm_delta_l2 (largest L-R) ===")
    for i in idx:
        print(
            f"frame={int(metrics['frame'][i])}  "
            f"L={metrics['tau_left_arm_l2'][i]:.3g}  "
            f"R={metrics['tau_right_arm_l2'][i]:.3g}  "
            f"delta={arm_delta[i]:.3g}  "
            f"ratio={arm_ratio[i]:.3g}  "
            f"diff_vec={arm_diff[i]:.3g}"
        )


def write_metrics_csv(out_path: str, metrics: Dict[str, List[float]]) -> None:
    keys = list(metrics.keys())
    n = len(metrics[keys[0]])
    for k in keys[1:]:
        if len(metrics[k]) != n:
            raise ValueError(f"Metric length mismatch: {k} has {len(metrics[k])}, expected {n}")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([metrics[k][i] for k in keys])


def main() -> int:
    # 默认路径：不传命令行参数时，直接分析你 Unity 导出的 CSV
    DEFAULT_INPUT = r"E:\SchoolWork\projects\projectX\SIGGRAPH2024Unity\Assets\StreamingAssets\tau_debug.csv"
    DEFAULT_OUT = r"E:\SchoolWork\projects\projectX\tau_metrics.csv"

    ap = argparse.ArgumentParser(description="Analyze tau_debug.csv (frame_id + tau vector per row).")
    ap.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to tau_debug.csv (first col=frame id, remaining cols=tau). Default: {DEFAULT_INPUT}",
    )
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',').")
    ap.add_argument("--has-header", action="store_true", help="Treat the first row as header.")
    ap.add_argument(
        "--expected-tau-dim",
        type=int,
        default=75,
        help="Expected tau dimension (default: 75). Use -1 to disable checking.",
    )
    ap.add_argument(
        "--out-metrics",
        default=DEFAULT_OUT,
        help=f"Write per-frame computed metrics to this CSV path. Default: {DEFAULT_OUT}. "
        f"Use empty string to disable writing.",
    )
    ap.add_argument("--top-k", type=int, default=10, help="How many top frames to show (default: 10).")
    args = ap.parse_args()

    expected = None if args.expected_tau_dim < 0 else int(args.expected_tau_dim)
    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            f"Tip: either put your CSV at the default path above, or run with --input <path> once."
        )
    frames, taus = read_tau_csv(
        args.input, delimiter=args.delimiter, has_header=bool(args.has_header), expected_tau_dim=expected
    )
    slices = TauSlices()
    if len(taus[0]) < slices.RHAND[1]:
        raise ValueError(
            f"tau dim too small for arm slices: got D={len(taus[0])}, needs at least {slices.RHAND[1]}."
        )

    metrics = compute_metrics(frames, taus, slices)
    print_summary(metrics, top_k=args.top_k)

    if args.out_metrics:
        write_metrics_csv(args.out_metrics, metrics)
        print(f"\nWrote metrics CSV: {args.out_metrics}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


