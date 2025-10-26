import cv2
import numpy as np
from scipy.linalg import lstsq
from scipy.spatial.transform import Rotation as R
from normalflow.utils import height2pointcloud, get_J

class InsufficientOverlapError(Exception):
    """Exception raised when there is insufficient shared contact region between frames."""
    
    def __init__(
        self,
        message="Insufficient shared contact regions between frames for reliable NormalFlow registration.",
    ):
        super().__init__(message)

def gauss_newton_with_shared_region(N_ref, C_ref, N_tar, C_tar, H_ref, H_tar,
                                   tar_T_ref_init, ppmm, n_samples=5000,
                                   max_iters=50):
    # Apply mask to pointcloud and normals on the reference
    tar_T_ref_init = tar_T_ref_init.astype(np.float32)
    pointcloud_ref = height2pointcloud(H_ref, ppmm).astype(np.float32)
    masked_pointcloud_ref = pointcloud_ref[C_ref.reshape(-1)]
    masked_N_ref = N_ref.reshape(-1, 3)[C_ref.reshape(-1)]
    
    # Randomly sample the points to speed up
    if n_samples is not None and n_samples < masked_N_ref.shape[0]:
        sample_mask = np.random.choice(masked_N_ref.shape[0], n_samples, replace=False)
    else:
        sample_mask = np.arange(masked_N_ref.shape[0])
    masked_pointcloud_ref = masked_pointcloud_ref[sample_mask]
    masked_N_ref = masked_N_ref[sample_mask]
    J = get_J(N_ref, C_ref, masked_pointcloud_ref, sample_mask, ppmm)

    # Apply Gauss-Newton optimization
    tar_T_ref = tar_T_ref_init.copy()
    final_shared_C = None
    overlap_history = []
    
    for i in range(max_iters):
        # Remap the pointcloud
        remapped_pointcloud_ref = (
            np.dot(tar_T_ref[:3, :3], masked_pointcloud_ref.T).T + tar_T_ref[:3, 3]
        )
        remapped_xx_ref = (
            remapped_pointcloud_ref[:, 0] * 1000.0 / ppmm + N_ref.shape[1] / 2 - 0.5
        )
        remapped_yy_ref = (
            remapped_pointcloud_ref[:, 1] * 1000.0 / ppmm + N_ref.shape[0] / 2 - 0.5
        )
        
        # Get the shared contact map
        remapped_C_tar = (
            cv2.remap(
                C_tar.astype(np.float32),
                remapped_xx_ref,
                remapped_yy_ref,
                cv2.INTER_LINEAR,
            )[:, 0]
            > 0.5
        )
        xx_region = np.logical_and(
            remapped_xx_ref >= 0, remapped_xx_ref < C_ref.shape[1]
        )
        yy_region = np.logical_and(
            remapped_yy_ref >= 0, remapped_yy_ref < C_ref.shape[0]
        )
        xy_region = np.logical_and(xx_region, yy_region)
        shared_C = np.logical_and(remapped_C_tar, xy_region)
        
        # Record overlap information
        overlap_ratio = np.sum(shared_C) / max(1, len(masked_pointcloud_ref))
        overlap_history.append(overlap_ratio)
        
        if np.sum(shared_C) < 10:
            raise InsufficientOverlapError()

        # Least square estimation
        remapped_N_tar = cv2.remap(
            N_tar, remapped_xx_ref, remapped_yy_ref, cv2.INTER_LINEAR
        )[:, 0, :]
        b = (remapped_N_tar @ np.linalg.inv(tar_T_ref[:3, :3]).T - masked_N_ref)[
            shared_C
        ].reshape(-1)
        A = np.transpose(J, (2, 0, 1))[shared_C].reshape(-1, 5)
        dp = lstsq(A, b, lapack_driver="gelsy")[0]

        # Update matrix T by transformation composition
        dR = R.from_euler("xyz", dp[:3], degrees=False).as_matrix()
        dT = np.identity(4, dtype=np.float32)
        dT[:3, :3] = dR
        dT[:2, 3] = dp[3:]
        tar_T_ref = np.dot(tar_T_ref, np.linalg.inv(dT))
        tar_T_ref[2, 3] = 0.0
        
        # Save final shared region
        final_shared_C = shared_C.copy()

        # Convergence check or reaching maximum iterations
        if np.linalg.norm(dp[:3]) < 1e-4 and np.linalg.norm(dp[3:]) < 1e-5 and i > 5:
            break

    # Calculate z translation by height difference
    remapped_pointcloud_ref = (
        np.dot(tar_T_ref[:3, :3], masked_pointcloud_ref.T).T + tar_T_ref[:3, 3]
    )
    remapped_xx_ref = (
        remapped_pointcloud_ref[:, 0] * 1000.0 / ppmm + N_ref.shape[1] / 2 - 0.5
    )
    remapped_yy_ref = (
        remapped_pointcloud_ref[:, 1] * 1000.0 / ppmm + N_ref.shape[0] / 2 - 0.5
    )
    remapped_C_tar = (
        cv2.remap(
            C_tar.astype(np.float32), remapped_xx_ref, remapped_yy_ref, cv2.INTER_LINEAR
        )[:, 0]
        > 0.5
    )
    xx_region = np.logical_and(remapped_xx_ref >= 0, remapped_xx_ref < C_ref.shape[1])
    yy_region = np.logical_and(remapped_yy_ref >= 0, remapped_yy_ref < C_ref.shape[0])
    xy_region = np.logical_and(xx_region, yy_region)
    remapped_C_tar = np.logical_and(remapped_C_tar, xy_region)
    remapped_H_tar = cv2.remap(
        H_tar, remapped_xx_ref, remapped_yy_ref, cv2.INTER_LINEAR
    )[:, 0]
    tar_T_ref[2, 3] = np.mean(
        remapped_H_tar[remapped_C_tar] * ppmm / 1000.0
        - remapped_pointcloud_ref[:, 2][remapped_C_tar],
        axis=0,
    )
    
    # Build overlap information
    final_remapped_pointcloud_ref = (
        np.dot(tar_T_ref[:3, :3], masked_pointcloud_ref.T).T + tar_T_ref[:3, 3]
    )
    final_remapped_xx_ref = (
        final_remapped_pointcloud_ref[:, 0] * 1000.0 / ppmm + N_ref.shape[1] / 2 - 0.5
    )
    final_remapped_yy_ref = (
        final_remapped_pointcloud_ref[:, 1] * 1000.0 / ppmm + N_ref.shape[0] / 2 - 0.5
    )
    
    overlap_info = {
        'shared_points': np.sum(final_shared_C) if final_shared_C is not None else 0,
        'total_points': len(masked_pointcloud_ref),
        'overlap_ratio': overlap_history[-1] if overlap_history else 0.0,
        'remapped_coords': (final_remapped_xx_ref, final_remapped_yy_ref),
        'masked_pointcloud_ref': masked_pointcloud_ref
    }
    
    return tar_T_ref, final_shared_C, overlap_info