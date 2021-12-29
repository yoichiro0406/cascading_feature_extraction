import open3d as o3d
import numpy as np

from matplotlib import pyplot as plt
from rich.progress import track
import argparse

import os


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pcd_root', type=str, default='/home/dataset/odometry/dataset/poses')
    parser.add_argument('--poses_root', type=str, default='/home/dataset/odometry/sequences')
    parser.add_argument('--new_root', type=str, default='/home/dataset/kitti_aligned')
    return parser


R = np.array([
    7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
    -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
]).reshape(3, 3)
T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
velo2cam = np.hstack([R, T])
velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T


def get_kitti_scan(path):
    with open (path, "rb") as f:
        scan = np.fromfile(f, dtype=np.float32)
        scan = scan.reshape((-1, 4))
    return scan[:, :3]


def scan2pcd(scan, estimate_normal=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan)
    if estimate_normal:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(3.0, max_nn=30))
    return pcd


def get_kitti_pcd(path, max_points=12000, voxel_size=0.3):
    scan = get_kitti_scan(path)
    pcd = scan2pcd(scan, estimate_normal=True)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def icp_p2l(pcd_a, pcd_b, threshold=1.0, fitness_th=0.65):
    reg_p2l = o3d.registration.registration_icp(
        pcd_a, pcd_b, threshold, np.eye(4),
        o3d.registration.TransformationEstimationPointToPlane(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    res = reg_p2l.fitness > fitness_th
    return reg_p2l.transformation, res


def get_nearest_over_Nm(poses, i, min_dist=10.0):
    dists = poses[i].reshape(1, 1, 3) - poses.reshape(1, -1, 3)
    dists = np.sum(dists ** 2, axis=-1)[0]
    try:
        nearest_id = np.where(dists[i:] > min_dist ** 2)[0][0]
        return nearest_id + i
    except:
        return None


def allign_for_init(pcd_a, pcd_b, pose_a, pose_b):
    T_w_a = pose_a.reshape(3, 4)
    T_w_a = np.vstack((T_w_a, [0, 0, 0, 1]))
    T_w_b = pose_b.reshape(3, 4)
    T_w_b = np.vstack((T_w_b, [0, 0, 0, 1]))

    M = (velo2cam @ T_w_a.T @ np.linalg.inv(T_w_b.T) @ np.linalg.inv(velo2cam)).T
    pcd_a.transform(M)
    M_icp_aux, res = icp_p2l(pcd_a, pcd_b)
    M = M @ M_icp_aux
    pcd_a.transform(M_icp_aux)

    return pcd_a, pcd_b, M, res


def plt_3d(pcd_a, pcd_b):
    fig = plt.figure(figsize=(50,50))
    ax = fig.add_subplot( 111 , projection='3d')
    pcd_a_np = np.asarray(pcd_a.points)
    pcd_b_np = np.asarray(pcd_b.points)
    ax.scatter(pcd_a_np[..., 0], pcd_a_np[..., 1], pcd_a_np[..., 2], s=20)
    ax.scatter(pcd_b_np[..., 0], pcd_b_np[..., 1], pcd_b_np[..., 2], s=20)


def plt_2d(pcd_a, pcd_b):
    fig = plt.figure(figsize=(50,50))
    ax = fig.add_subplot()
    pcd_a_np = np.asarray(pcd_a.points)
    pcd_b_np = np.asarray(pcd_b.points)
    ax.scatter(pcd_a_np[..., 0], pcd_a_np[..., 1], s=20)
    ax.scatter(pcd_b_np[..., 0], pcd_b_np[..., 1], s=20)
    plt.savefig('pics/{}.png'.format(np.random.rand()))
    plt.close(fig)

    
def main():
    drive_ids = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    poses_root = args.poses_root
    pcd_root = args.pcd_root

    new_root = args.new_root

    min_dist = 10

    for drive_id in drive_ids:
        if not os.path.exists(os.path.join(new_root, drive_id)):
            os.makedirs(os.path.join(new_root, drive_id))

        pcd_bin_dir = os.path.join(pcd_root, drive_id, 'velodyne')
        pcd_bin_paths = os.listdir(pcd_bin_dir)
        pcd_bin_paths.sort()

        poses_path = os.path.join(poses_root, drive_id + '.txt')
        poses = np.genfromtxt(poses_path).reshape(-1, 3, 4)
        
        for i, pcd_bin_path in track(enumerate(pcd_bin_paths), total=len(pcd_bin_paths), description=drive_id):
            pcd_bin_path = os.path.join(pcd_bin_dir, pcd_bin_path)
            pcd_src = get_kitti_pcd(pcd_bin_path)
            pose_src = poses[i]

            pcd_ref_id = get_nearest_over_Nm(poses[..., 3], i, min_dist=min_dist)
            if pcd_ref_id is None:
                continue
            pcd_ref = get_kitti_pcd(os.path.join(pcd_bin_dir, pcd_bin_paths[pcd_ref_id]))
            pose_ref = poses[pcd_ref_id]

            pcd_src, pcd_ref, trans, icp_res = allign_for_init(pcd_src, pcd_ref, pose_src, pose_ref)
            
            if icp_res:
                filename = os.path.basename(pcd_bin_path)
                new_filename = filename.replace('.bin', '_src.pcd')
                save_path_src = os.path.join(new_root, drive_id, new_filename)
                save_path_ref = save_path_src.replace('src', 'ref')
                save_path_trans = save_path_src.replace('_src.pcd', '_trans.npy')
                o3d.io.write_point_cloud(save_path_src, pcd_src)
                o3d.io.write_point_cloud(save_path_ref, pcd_ref)
                np.save(save_path_trans, trans[:3])


if __name__ == '__main__':
    args = get_args()
    main()

