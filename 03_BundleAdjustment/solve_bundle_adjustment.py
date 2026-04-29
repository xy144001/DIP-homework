import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


IMAGE_SIZE = 1024
NUM_VIEWS = 50
NUM_POINTS = 20000


def euler_xyz_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    cx = torch.cos(euler_angles[:, 0])
    sx = torch.sin(euler_angles[:, 0])
    cy = torch.cos(euler_angles[:, 1])
    sy = torch.sin(euler_angles[:, 1])
    cz = torch.cos(euler_angles[:, 2])
    sz = torch.sin(euler_angles[:, 2])

    rx = torch.stack(
        [
            torch.ones_like(cx),
            torch.zeros_like(cx),
            torch.zeros_like(cx),
            torch.zeros_like(cx),
            cx,
            -sx,
            torch.zeros_like(cx),
            sx,
            cx,
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    ry = torch.stack(
        [
            cy,
            torch.zeros_like(cy),
            sy,
            torch.zeros_like(cy),
            torch.ones_like(cy),
            torch.zeros_like(cy),
            -sy,
            torch.zeros_like(cy),
            cy,
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    rz = torch.stack(
        [
            cz,
            -sz,
            torch.zeros_like(cz),
            sz,
            cz,
            torch.zeros_like(cz),
            torch.zeros_like(cz),
            torch.zeros_like(cz),
            torch.ones_like(cz),
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    return rz @ ry @ rx


def load_observations(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    points2d = np.load(data_dir / "points2d.npz")
    views = [points2d[f"view_{i:03d}"] for i in range(NUM_VIEWS)]
    stacked = np.stack(views, axis=0).astype(np.float32)
    points = stacked[:, :, :2]
    visibility = stacked[:, :, 2].astype(np.float32)
    return points, visibility


def initialize_cameras(distance: float, fov_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
    yaw = np.deg2rad(np.linspace(-70.0, 70.0, NUM_VIEWS, dtype=np.float32))
    eulers = np.zeros((NUM_VIEWS, 3), dtype=np.float32)
    eulers[:, 1] = yaw

    translations = np.zeros((NUM_VIEWS, 3), dtype=np.float32)
    translations[:, 2] = -distance

    focal = IMAGE_SIZE / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    return eulers, translations, float(focal)


def initialize_points(
    observations: np.ndarray,
    visibility: np.ndarray,
    eulers: np.ndarray,
    translations: np.ndarray,
    focal: float,
    distance: float,
) -> np.ndarray:
    cx = cy = IMAGE_SIZE / 2.0
    rot = euler_xyz_to_matrix(torch.from_numpy(eulers)).numpy()

    accum = np.zeros((NUM_POINTS, 3), dtype=np.float32)
    counts = np.zeros((NUM_POINTS, 1), dtype=np.float32)

    for view_idx in range(NUM_VIEWS):
        vis = visibility[view_idx] > 0.5
        if not np.any(vis):
            continue

        uv = observations[view_idx, vis]
        x_cam = (uv[:, 0] - cx) * distance / focal
        y_cam = -(uv[:, 1] - cy) * distance / focal
        z_cam = np.full_like(x_cam, -distance)
        cam_points = np.stack([x_cam, y_cam, z_cam], axis=1)

        world_points = (rot[view_idx].T @ (cam_points - translations[view_idx]).T).T
        accum[vis] += world_points
        counts[vis] += 1.0

    counts = np.maximum(counts, 1.0)
    points = accum / counts

    unseen = counts[:, 0] == 0
    if np.any(unseen):
        points[unseen] = np.random.normal(scale=0.05, size=(unseen.sum(), 3)).astype(np.float32)
    points += np.random.normal(scale=0.01, size=points.shape).astype(np.float32)
    return points


def project_points(
    points3d: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    focal: torch.Tensor,
) -> torch.Tensor:
    cam_points = torch.einsum("vij,nj->vni", rotations, points3d) + translations[:, None, :]
    z = cam_points[..., 2].clamp(max=-1e-3)
    cx = cy = IMAGE_SIZE / 2.0
    u = -focal * cam_points[..., 0] / z + cx
    v = focal * cam_points[..., 1] / z + cy
    return torch.stack([u, v], dim=-1)


def save_obj(path: Path, points3d: np.ndarray, colors: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for point, color in zip(points3d, colors):
            f.write(
                f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def run_bundle_adjustment(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_np, vis_np = load_observations(data_dir)
    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32) / 255.0

    init_eulers, init_trans, init_f = initialize_cameras(args.distance, args.init_fov_deg)
    init_points = initialize_points(obs_np, vis_np, init_eulers, init_trans, init_f, args.distance)

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    obs = torch.from_numpy(obs_np).to(device)
    vis = torch.from_numpy(vis_np).to(device)

    fixed_euler = torch.from_numpy(init_eulers[:1]).to(device)
    fixed_trans = torch.from_numpy(init_trans[:1]).to(device)

    free_eulers = torch.nn.Parameter(torch.from_numpy(init_eulers[1:]).to(device))
    free_trans = torch.nn.Parameter(torch.from_numpy(init_trans[1:]).to(device))
    points3d = torch.nn.Parameter(torch.from_numpy(init_points).to(device))
    log_f = torch.nn.Parameter(torch.tensor(math.log(init_f), dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([free_eulers, free_trans, points3d, log_f], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.steps // 3, 1), gamma=0.5)

    loss_history: list[float] = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        eulers = torch.cat([fixed_euler, free_eulers], dim=0)
        translations = torch.cat([fixed_trans, free_trans], dim=0)
        rotations = euler_xyz_to_matrix(eulers)
        focal = torch.exp(log_f)

        pred = project_points(points3d, rotations, translations, focal)
        residual = pred - obs
        reproj = torch.sqrt((residual ** 2).sum(dim=-1) + 1e-8)

        valid_count = vis.sum().clamp(min=1.0)
        reproj_loss = (reproj * vis).sum() / valid_count
        point_reg = 1e-4 * points3d.pow(2).mean()
        trans_reg = 1e-4 * free_trans[:, :2].pow(2).mean()
        loss = reproj_loss + point_reg + trans_reg

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step {step:04d}/{args.steps} "
                f"loss={loss_value:.6f} "
                f"reproj={float(reproj_loss.item()):.6f} "
                f"f={float(focal.item()):.2f}"
            )

    with torch.no_grad():
        eulers = torch.cat([fixed_euler, free_eulers], dim=0)
        translations = torch.cat([fixed_trans, free_trans], dim=0)
        rotations = euler_xyz_to_matrix(eulers)
        focal = torch.exp(log_f)
        pred = project_points(points3d, rotations, translations, focal)
        residual = pred - obs
        reproj = torch.sqrt((residual ** 2).sum(dim=-1) + 1e-8)
        final_reproj = float((reproj * vis).sum().item() / vis.sum().item())

        np.savez(
            output_dir / "bundle_adjustment_result.npz",
            points3d=points3d.cpu().numpy(),
            eulers=eulers.cpu().numpy(),
            translations=translations.cpu().numpy(),
            focal=float(focal.item()),
            final_reprojection_error=final_reproj,
            loss_history=np.array(loss_history, dtype=np.float32),
        )
        save_obj(output_dir / "reconstructed_points.obj", points3d.cpu().numpy(), colors)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Bundle Adjustment Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    summary = (
        f"device: {device}\n"
        f"steps: {args.steps}\n"
        f"final_focal: {float(focal.item()):.6f}\n"
        f"final_reprojection_error: {final_reproj:.6f}\n"
        f"output_dir: {output_dir}\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--init-fov-deg", type=float, default=45.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/task1")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_bundle_adjustment(parse_args())
