#
# Convert a Replica SLAM-format scene (results/frame*.jpg + depth*.png + traj.txt)
# into a hierarchical-3d-gaussians standalone chunk:
#
#   <out>/sparse/0/{cameras.bin, images.bin, points3D.bin, test.txt, depth_params.json}
#   <out>/images/<frameXXXXXX.jpg>   (symlinks into the Replica results dir)
#   <out>/depths/<frameXXXXXX.png>   (16-bit inverse-depth from GT depth)
#
# Poses come from traj.txt (rows are 4x4 c2w, cf. MonoGS ReplicaParser), the init
# point cloud from backprojected GT depth. Everything is written in BINARY COLMAP
# with a 4-param PINHOLE camera because GaussianHierarchyCreator's appearance
# filter only parses that layout (appearance_filter.cpp).
#
# The scene is upscaled by --upscale (default 10): with fx=600 and ~2m median
# cam-to-surface distance this lands at the ~20-unit median distance the h3dgs
# preprocessing (auto_reorient --target_med_dist 20) normalizes to, and keeps
# geometry clear of the rasterizer's 0.2-unit near-plane cull. Synthesized
# inverse depths and depth_params.json scales are expressed in the same frame:
# loader computes png/2^16 * scale = invdepth_metric/upscale exactly.
#
import argparse
import json
import os
import struct

import cv2
import numpy as np


def rotmat2qvec(R):
    # COLMAP convention (qw, qx, qy, qz), from colmap/scripts/python/read_write_model.py
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_bin(path, width, height, fx, fy, cx, cy):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        # camera_id=1, model_id=1 (PINHOLE), width, height, fx fy cx cy
        f.write(struct.pack("<iiQQ", 1, 1, width, height))
        f.write(struct.pack("<dddd", fx, fy, cx, cy))


def write_images_bin(path, images):
    # images: list of (image_id, qvec, tvec, name)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, qvec, tvec, name in images:
            f.write(struct.pack("<i", image_id))
            f.write(struct.pack("<dddd", *qvec))
            f.write(struct.pack("<ddd", *tvec))
            f.write(struct.pack("<i", 1))  # camera_id
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))  # num_points2D


def write_points3d_bin(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", xyz.shape[0]))
        for i in range(xyz.shape[0]):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<ddd", *xyz[i]))
            f.write(struct.pack("<BBB", *rgb[i]))
            f.write(struct.pack("<d", 1.0))
            f.write(struct.pack("<Q", 0))  # track length


def load_depth_m(path, depth_scale):
    d = cv2.imread(path, -1)
    if d is None:
        raise FileNotFoundError(path)
    dm = d.astype(np.float32) / depth_scale
    if (d == 0).any():
        # GT depth has a handful of invalid pixels; inpaint so that invdepth=0
        # ("infinitely far") never supervises the depth loss with garbage.
        mask = (d == 0).astype(np.uint8)
        dm = cv2.inpaint(dm, mask, 3, cv2.INPAINT_NS)
    return dm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replica_dir", required=True, help="Replica scene dir (contains results/ and traj.txt)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--stride", type=int, default=5, help="use every Nth frame")
    p.add_argument("--test_hold", type=int, default=8, help="every Nth used frame goes to test.txt")
    p.add_argument("--upscale", type=float, default=10.0)
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=680)
    p.add_argument("--fx", type=float, default=600.0)
    p.add_argument("--fy", type=float, default=600.0)
    p.add_argument("--cx", type=float, default=599.5)
    p.add_argument("--cy", type=float, default=339.5)
    p.add_argument("--depth_scale", type=float, default=6553.5)
    p.add_argument("--points_frame_stride", type=int, default=10, help="every Nth used frame feeds the init point cloud")
    p.add_argument("--points_pixel_stride", type=int, default=6)
    p.add_argument("--voxel", type=float, default=0.02, help="init cloud voxel size in meters (pre-upscale)")
    p.add_argument("--copy_images", action="store_true", help="copy instead of symlink")
    args = p.parse_args()

    res_dir = os.path.join(args.replica_dir, "results")
    traj = np.loadtxt(os.path.join(args.replica_dir, "traj.txt")).reshape(-1, 4, 4)
    n_total = traj.shape[0]
    used = list(range(0, n_total, args.stride))
    print(f"{n_total} frames total, using {len(used)} (stride {args.stride})")

    out = args.output_dir
    sparse = os.path.join(out, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "depths"), exist_ok=True)

    u = args.upscale
    images = []
    for k, idx in enumerate(used):
        c2w = traj[idx]
        w2c = np.linalg.inv(c2w)
        qvec = rotmat2qvec(w2c[:3, :3])
        tvec = w2c[:3, 3] * u
        name = f"frame{idx:06d}.jpg"
        images.append((k + 1, qvec, tvec, name))
        dst = os.path.join(out, "images", name)
        src = os.path.abspath(os.path.join(res_dir, name))
        if not os.path.exists(dst):
            if args.copy_images:
                import shutil
                shutil.copy(src, dst)
            else:
                os.symlink(src, dst)

    write_cameras_bin(os.path.join(sparse, "cameras.bin"),
                      args.width, args.height, args.fx, args.fy, args.cx, args.cy)
    write_images_bin(os.path.join(sparse, "images.bin"), images)

    test_names = [images[k][3] for k in range(len(images)) if k % args.test_hold == 0]
    with open(os.path.join(sparse, "test.txt"), "w") as f:
        f.write("\n".join(test_names) + "\n")
    print(f"{len(test_names)} test / {len(images) - len(test_names)} train")

    # --- pass 1: global inverse-depth normalization constant K ---
    K_inv = 0.0
    for idx in used:
        dm = load_depth_m(os.path.join(res_dir, f"depth{idx:06d}.png"), args.depth_scale)
        K_inv = max(K_inv, float((1.0 / dm[dm > 0]).max()))
    print(f"global max inverse depth K = {K_inv:.4f} (min depth {1.0/K_inv:.3f} m)")

    # --- pass 2: write 16-bit inverse-depth PNGs ---
    for idx in used:
        dm = load_depth_m(os.path.join(res_dir, f"depth{idx:06d}.png"), args.depth_scale)
        inv = np.zeros_like(dm)
        np.divide(1.0, dm, out=inv, where=dm > 0)
        png = np.round(65535.0 * inv / K_inv).astype(np.uint16)
        cv2.imwrite(os.path.join(out, "depths", f"frame{idx:06d}.png"), png)

    # loader: invdepth_used = png/2^16 * scale + offset  ==  invdepth_metric / upscale
    scale = K_inv * (65536.0 / 65535.0) / u
    depth_params = {f"frame{idx:06d}": {"scale": scale, "offset": 0.0} for idx in used}
    with open(os.path.join(sparse, "depth_params.json"), "w") as f:
        json.dump(depth_params, f, indent=2)

    # --- init point cloud: backprojected GT depth, voxel-downsampled ---
    vox = {}
    ys, xs = np.mgrid[0:args.height:args.points_pixel_stride,
                      0:args.width:args.points_pixel_stride]
    ys, xs = ys.ravel(), xs.ravel()
    for k, idx in enumerate(used[::args.points_frame_stride]):
        dm = load_depth_m(os.path.join(res_dir, f"depth{idx:06d}.png"), args.depth_scale)
        bgr = cv2.imread(os.path.join(res_dir, f"frame{idx:06d}.jpg"))
        d = dm[ys, xs]
        valid = d > 0
        x = (xs[valid] - args.cx) / args.fx * d[valid]
        y = (ys[valid] - args.cy) / args.fy * d[valid]
        pts_cam = np.stack([x, y, d[valid]], axis=1)
        c2w = traj[idx]
        pts_w = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
        cols = bgr[ys[valid], xs[valid]][:, ::-1]  # BGR -> RGB
        keys = np.floor(pts_w / args.voxel).astype(np.int64)
        for key, pt, col in zip(map(tuple, keys), pts_w, cols):
            if key not in vox:
                vox[key] = (pt, col)
    xyz = np.array([v[0] for v in vox.values()]) * u
    rgb = np.array([v[1] for v in vox.values()], dtype=np.uint8)
    print(f"init point cloud: {xyz.shape[0]} points (voxel {args.voxel} m)")
    write_points3d_bin(os.path.join(sparse, "points3D.bin"), xyz, rgb)

    with open(os.path.join(out, "conversion_meta.json"), "w") as f:
        json.dump({"replica_dir": os.path.abspath(args.replica_dir),
                   "stride": args.stride, "test_hold": args.test_hold,
                   "upscale": u, "K_inv_depth": K_inv,
                   "n_used": len(used), "n_test": len(test_names),
                   "n_points": int(xyz.shape[0])}, f, indent=2)
    print("done:", out)


if __name__ == "__main__":
    main()
