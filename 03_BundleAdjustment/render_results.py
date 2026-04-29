import argparse
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_obj_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    points = []
    colors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            fields = line.strip().split()
            if len(fields) >= 4:
                points.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if len(fields) >= 7:
                colors.append([float(fields[4]), float(fields[5]), float(fields[6])])
    point_array = np.asarray(points, dtype=np.float32)
    color_array = np.asarray(colors, dtype=np.float32) if colors else None
    return point_array, color_array


def load_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY file: {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        vertex_count = 0
        is_binary = False
        for line in header_lines:
            if line.startswith("format "):
                is_binary = "binary_little_endian" in line
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])

        points = []
        colors = []
        if is_binary:
            record_struct = struct.Struct("<fffBBB")
            for _ in range(vertex_count):
                chunk = f.read(record_struct.size)
                if len(chunk) != record_struct.size:
                    break
                x, y, z, r, g, b = record_struct.unpack(chunk)
                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])
        else:
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                fields = line.decode("ascii").strip().split()
                if len(fields) < 3:
                    continue
                points.append([float(fields[0]), float(fields[1]), float(fields[2])])
                if len(fields) >= 6:
                    colors.append([float(fields[3]) / 255.0, float(fields[4]) / 255.0, float(fields[5]) / 255.0])
    point_array = np.asarray(points, dtype=np.float32)
    color_array = np.asarray(colors, dtype=np.float32) if colors else None
    return point_array, color_array


def subsample(points: np.ndarray, colors: np.ndarray | None, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    sampled_points = points[indices]
    sampled_colors = colors[indices] if colors is not None else None
    return sampled_points, sampled_colors


def render_point_cloud(
    points: np.ndarray,
    output_path: Path,
    title: str,
    colors: np.ndarray | None = None,
    elev: float = 20.0,
    azim: float = 35.0,
) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    if colors is None:
        color_arg = "#1f77b4"
        alpha = 0.75
    else:
        color_arg = np.clip(colors, 0.0, 1.0)
        alpha = 0.9

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color_arg,
        s=1.2,
        alpha=alpha,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Assignment 03 point cloud results to PNG files.")
    parser.add_argument(
        "--task1-obj",
        default="outputs/task1_final/reconstructed_points.obj",
        help="Task 1 reconstructed OBJ path, relative to the script directory.",
    )
    parser.add_argument(
        "--task2-ply",
        default="data/colmap/sparse/0/sparse_points.ply",
        help="Task 2 sparse PLY path, relative to the script directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rendered_results",
        help="Directory to save rendered PNG files, relative to the script directory.",
    )
    parser.add_argument("--max-task1-points", type=int, default=12000)
    parser.add_argument("--max-task2-points", type=int, default=5000)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / args.output_dir

    task1_points, task1_colors = load_obj_vertices(base_dir / args.task1_obj)
    task1_points, task1_colors = subsample(task1_points, task1_colors, args.max_task1_points)
    render_point_cloud(
        task1_points,
        output_dir / "task1_point_cloud.png",
        "Task 1 Reconstructed Point Cloud",
        colors=task1_colors,
        elev=18.0,
        azim=32.0,
    )

    task2_points, task2_colors = load_ply_vertices(base_dir / args.task2_ply)
    task2_points, task2_colors = subsample(task2_points, task2_colors, args.max_task2_points)
    render_point_cloud(
        task2_points,
        output_dir / "task2_sparse_point_cloud.png",
        "Task 2 COLMAP Sparse Point Cloud",
        colors=task2_colors,
        elev=24.0,
        azim=48.0,
    )


if __name__ == "__main__":
    main()
