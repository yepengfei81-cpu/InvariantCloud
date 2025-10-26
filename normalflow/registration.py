import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R2
import open3d as o3d
from normalflow.benchmark import gauss_newton_with_shared_region
from normalflow.marker_interpolator import (
    interpolate_markers_to_475_grid, 
    reset_interpolator)
from normalflow.grid_line_assigner import (
    assign_marker_indices_7x9_traditional, 
    reset_grid_assigner)
from normalflow.flat_frame_rotation import FlatFrameRotationCalculator
from normalflow.point_reconstruction import (
        initialize_multiframe_fusion, 
        add_frame_to_fusion, 
        start_new_fusion_session,
        generate_fusion_mesh)
from normalflow.contact_reconnection import (
    save_contact_for_reconnection,
    match_contact_for_reconnection
)

class InsufficientOverlapError(Exception):
    """Exception raised when there is insufficient shared contact region between frames."""

    def __init__(
        self,
        message="Insufficient shared contact regions between frames for reliable NormalFlow registration.",
    ):
        super().__init__(message)

# Global variables
keyframe_pts_2d  = None
keyframe_image   = None
keyframe_H       = None
keyframe_C       = None
cumulative_xy    = np.array([0.0, 0.0])
cumulative_xyz   = np.array([0.0, 0.0, 0.0])
cumulative_rot   = np.array([0.0, 0.0, 0.0])
prev_centroid    = np.zeros(3)
geometric_center = np.zeros(3)
keyframe_indices = None
reconstruction_paused = False
last_disconnect_cumulative_xy = np.array([0.0, 0.0])
last_disconnect_cumulative_z = 0.0
last_disconnect_cumulative_rot = np.array([0.0, 0.0, 0.0])
reconnection_alignment_result = None 
first_contact_motion_direction = None
motion_projection_enabled = False
flat_frame_calculator = FlatFrameRotationCalculator()
flat_frame_image_for_vis = None
flat_frame_pts_2d_for_vis = None
flat_frame_indices_for_vis = None

def detect_markers(image, expect_num=63, verbose=False, min_accept_ratio=0.7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Parameter sets from strict to relaxed
    param_sets = [
        # Standard parameters
        {
            'minThreshold': 15, 'maxThreshold': 200,
            'minArea': 35, 'maxArea': 300,
            'minCircularity': 0.4, 'minConvexity': 0.5
        },
        # Relaxed area constraints
        {
            'minThreshold': 10, 'maxThreshold': 220,
            'minArea': 20, 'maxArea': 500,
            'minCircularity': 0.3, 'minConvexity': 0.4
        },
        # Further relaxed shape requirements
        {
            'minThreshold': 5, 'maxThreshold': 240,
            'minArea': 15, 'maxArea': 800,
            'minCircularity': 0.2, 'minConvexity': 0.3
        },
        # Most relaxed parameters
        {
            'minThreshold': 5, 'maxThreshold': 250,
            'minArea': 10, 'maxArea': 1000,
            'minCircularity': 0.1, 'minConvexity': 0.1
        }
    ]
    
    best_keypoints = []
    best_score = 0
    
    for i, params_dict in enumerate(param_sets):
        params = cv2.SimpleBlobDetector_Params()
        
        params.minThreshold = params_dict['minThreshold']
        params.maxThreshold = params_dict['maxThreshold']
        
        params.filterByArea = True
        params.minArea = params_dict['minArea']
        params.maxArea = params_dict['maxArea']
        
        params.filterByCircularity = True
        params.minCircularity = params_dict['minCircularity']
        
        params.filterByConvexity = True
        params.minConvexity = params_dict['minConvexity']
        
        params.filterByInertia = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        num_detected = len(keypoints)
        coverage_score = min(1.0, num_detected / expect_num)
        
        if num_detected >= expect_num * 0.9:
            bonus = 0.2
        elif num_detected >= expect_num * 0.8:
            bonus = 0.1
        else:
            bonus = 0.0
            
        total_score = coverage_score + bonus
        
        if verbose:
            print(f"Param set {i+1}: Detected {num_detected} points, score: {total_score:.3f}")
        
        if total_score > best_score:
            best_score = total_score
            best_keypoints = keypoints
            
        if num_detected >= expect_num * 0.95:
            break
    
    min_required = int(expect_num * min_accept_ratio)
    if len(best_keypoints) < min_required:
        if verbose:
            print(f"Warning: Only detected {len(best_keypoints)} points, less than minimum required {min_required}")
        return np.zeros((0, 2), dtype=np.float32)
    
    if verbose:
        print(f"Final result: Detected {len(best_keypoints)}/{expect_num} markers")
    
    if len(best_keypoints) > 0:
        pts = np.array([kp.pt for kp in best_keypoints], dtype=np.float32)
        return pts[np.argsort(pts[:, 0])]
    else:
        return np.zeros((0, 2), dtype=np.float32)

def detect_markers_adaptive(image, expect_num=63, verbose=False):
    # Try standard detection first
    standard_result = detect_markers(image, expect_num, verbose, min_accept_ratio=0.8)
    
    if len(standard_result) >= expect_num * 0.8:
        return standard_result
    
    # Try image enhancement if standard detection fails
    if verbose:
        print("Standard detection poor, trying image enhancement...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    enhanced_result = detect_markers(enhanced_bgr, expect_num, verbose, min_accept_ratio=0.6)
    
    if len(enhanced_result) > len(standard_result):
        if verbose:
            print(f"Enhancement improved results: {len(enhanced_result)} vs {len(standard_result)}")
        return enhanced_result
    
    return standard_result if len(standard_result) >= len(enhanced_result) else enhanced_result

def visualize_indexed_features(image, all_pts_2d, all_indices, title="Indexed Features (13x17)"):
    disp = image.copy()
    
    for i, (x, y) in enumerate(all_pts_2d):
        cv2.circle(disp, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imshow(title, disp)
    return disp

def visualize_marker_indices(image, markers_2d, markers_indices, all_pts_2d=None, title="Marker Indices"):
    disp = image.copy()
    # Draw interpolated points (green, no labels)
    if all_pts_2d is not None:
        for (x, y) in all_pts_2d:
            cv2.circle(disp, (int(x), int(y)), 3, (0, 255, 0), -1)
    # Draw marker points (red, with labels)
    for i, (x, y) in enumerate(markers_2d):
        cv2.circle(disp, (int(x), int(y)), 4, (0, 0, 255), -1)
        text = str(markers_indices[i])
        cv2.putText(disp, text, (int(x) - 8, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow(title, disp)
    return disp

def pixels_to_world(pts_2d, height_map, ppmm):
    # Get image center coordinates
    height, width = height_map.shape[:2]
    center_x = width / 2.0 - 0.5
    center_y = height / 2.0 - 0.5
    
    # Get Z values from height map (bilinear interpolation)
    z_values = []
    for pt in pts_2d:
        x, y = pt
        x = np.clip(x, 0, width-1)
        y = np.clip(y, 0, height-1)
        
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
        
        dx, dy = x - x1, y - y1
        z11 = height_map[y1, x1]
        z12 = height_map[y1, x2]
        z21 = height_map[y2, x1]
        z22 = height_map[y2, x2]
        
        z_value = (1 - dx) * (1 - dy) * z11 + \
                  dx * (1 - dy) * z12 + \
                  (1 - dx) * dy * z21 + \
                  dx * dy * z22
        z_values.append(z_value)
    
    # Convert to 3D world coordinates
    pts_3d = np.zeros((len(pts_2d), 3), dtype=np.float32)
    
    # X: (pixel - center) * ppmm / 1000 -> meters
    pts_3d[:, 0] = (pts_2d[:, 0] - center_x) * ppmm / 1000.0
    
    # Y: negative because image Y-axis is downward
    pts_3d[:, 1] = -(pts_2d[:, 1] - center_y) * ppmm / 1000.0
    
    # Z: height * ppmm / 1000 -> meters
    pts_3d[:, 2] = np.array(z_values) * ppmm / 1000.0
    return pts_3d

def world_to_pixels(pts_3d, height_map, ppmm):
    H, W = height_map.shape[:2]
    center_x = (W - 1) * 0.5
    center_y = (H - 1) * 0.5

    pts_2d = np.empty((len(pts_3d), 2), np.float32)

    pts_2d[:, 0] = pts_3d[:, 0] * 1000.0 / ppmm + center_x
    pts_2d[:, 1] = -pts_3d[:, 1] * 1000.0 / ppmm + center_y

    return pts_2d

global_cumulative_z = 0.0
last_pca_z_angle = 0.0

def normalflow(
    N_ref,
    C_ref,
    H_ref,
    N_tar,
    C_tar,
    H_tar,
    image_init,
    image_ref,
    image_tar,
    tar_T_ref_init=np.eye(4),
    ppmm=0.0634,
    n_samples=5000,
    verbose=False,
    save_contact=False,
    match_contact=False, 
    has_reset=False,
    useful_info=None
):
    global keyframe_indices, flat_frame_calculator, first_contact_motion_direction
    global global_cumulative_z, last_pca_z_angle, motion_projection_enabled
    global cumulative_xy, cumulative_xyz, cumulative_rot, prev_centroid, geometric_center
    global keyframe_pts_2d, keyframe_image, keyframe_H, keyframe_C, reconnection_alignment_result
    global reconstruction_paused, last_disconnect_cumulative_xy, last_disconnect_cumulative_rot, last_disconnect_cumulative_z
    global flat_frame_image_for_vis, flat_frame_pts_2d_for_vis, flat_frame_indices_for_vis
    
    if not hasattr(normalflow, 'prev_extremes'):
        normalflow.prev_extremes = None   
    if not hasattr(normalflow, 'prev_contour'):
        normalflow.prev_contour = None     
    if not hasattr(normalflow, 'transformation_chain'):
        normalflow.transformation_chain = []
        normalflow.current_segment_base = np.eye(4)
    
    # Keyframe initialization
    if keyframe_image is None:
        keyframe_image = image_init.copy()
        keyframe_H = H_ref.copy()
        keyframe_C = C_ref.copy()
        
        reset_grid_assigner()
        reset_interpolator()
        markers = detect_markers_adaptive(keyframe_image, verbose=verbose)
        keyframe_pts_2d, keyframe_indices = assign_marker_indices_7x9_traditional(markers)
        keyframe_pts_2d_full, keyframe_indices_full = interpolate_markers_to_475_grid(
            keyframe_pts_2d, keyframe_indices, keyframe_H, ppmm
        )        
        keyframe_pts_2d = keyframe_pts_2d_full
        keyframe_indices = keyframe_indices_full        
        flat_frame_calculator.initialize_flat_frame(
            keyframe_pts_2d, keyframe_indices, keyframe_H, keyframe_C, verbose=verbose
        ) 
        flat_frame_image_for_vis = keyframe_image.copy()
        flat_frame_pts_2d_for_vis = keyframe_pts_2d.copy()
        flat_frame_indices_for_vis = keyframe_indices.copy()        
    elif has_reset:
        print("[Keyframe] Updating keyframe...")
        keyframe_image = image_tar.copy()
        keyframe_H = H_tar.copy()
        keyframe_C = C_tar.copy()  
    
    # 1. Compute XYZ rotation
    _, theta_xyz, _ = flat_frame_calculator.compute_xy_rotation_kabsch_by_flat_frame_points(
        keyframe_C, C_tar, keyframe_H, H_tar, ppmm, min_points=8, verbose=verbose
    )
    cumulative_rot += theta_xyz
    cumulative_rot[2] -= theta_xyz[2]    
    z_rotation_deg, geometric_center = flat_frame_calculator.visualize_prev_and_current_frame_contact_points(
        keyframe_C, C_tar, keyframe_H, H_tar, ppmm)
    if last_pca_z_angle == 0.0:
        last_pca_z_angle = z_rotation_deg
    delta_z = z_rotation_deg - last_pca_z_angle
    cumulative_rot[2] += delta_z
    last_pca_z_angle = z_rotation_deg
    
    # 2. Compute XYZ translation
    if np.allclose(prev_centroid, np.zeros(3)) or has_reset:
        prev_centroid = geometric_center.copy()
        delta_xy_mm = np.array([0.0, 0.0])
    else:
        delta_xy_mm = (geometric_center[:2] - prev_centroid[:2]) * 1000
    cumulative_xy += delta_xy_mm  
    z_curr_mm = geometric_center[2] * 1000
    cumulative_xyz = np.concatenate([cumulative_xy, [z_curr_mm]]) 
    # benchmark method
    tar_T_ref, _, _ = gauss_newton_with_shared_region(
        N_ref, C_ref,      # 普通关键帧
        N_tar, C_tar,      # 当前帧
        H_ref, H_tar,      # 普通关键帧和当前帧
        tar_T_ref_init, ppmm, 
        n_samples
    )    
    # 3. Long-sequence tracking and matching
    if save_contact or match_contact:
        _, current_contact_info, _ = flat_frame_calculator._get_contact_points_3d_with_matches(
            keyframe_C, C_tar, keyframe_H, H_tar, ppmm)
    
    if save_contact:
        if not reconstruction_paused:
            print("🛑 Pausing reconstruction, saving disconnect state...")
            reconstruction_paused = True
            last_disconnect_cumulative_xy = cumulative_xy.copy()
            last_disconnect_cumulative_z = -z_curr_mm
            last_disconnect_cumulative_rot = cumulative_rot.copy()     
        if current_contact_info is not None:
            success = save_contact_for_reconnection(
                current_contact_info, cumulative_rot[2], geometric_center, H_tar
            )
            if success:
                print("✅ Feature points saved successfully")
            else:
                print("❌ Failed to save feature points")
        else:
            print("❌ Cannot get contact point info")    
    
    if match_contact:
        print("\n🔵 Starting feature-based matching with previous contact...")
        if current_contact_info is not None:
            flat_frame_vis_data = None
            if flat_frame_image_for_vis is not None:
                flat_frame_vis_data = {
                    'image': flat_frame_image_for_vis,
                    'all_pts_2d': flat_frame_pts_2d_for_vis,
                    'all_indices': flat_frame_indices_for_vis
                }            
            is_match, _, match_info, match_details = match_contact_for_reconnection(
                current_contact_info, cumulative_rot[2], geometric_center, H_tar,
                flat_frame_vis_data=flat_frame_vis_data
            )
            
            if is_match:
                print(f"🎯 Reconnection successful! {match_info}")
                if reconstruction_paused:
                    reconstruction_paused = False 
                    normalflow.reconnection_reset_needed = True
                    motion_projection_enabled = True
                if match_details:
                    reconnection_alignment_result = match_details.copy()

                    last_segment_transform = np.eye(4)
                    last_disconnect_rot_no_x = last_disconnect_cumulative_rot.copy()
                    last_disconnect_rot_no_x[1] = -last_disconnect_rot_no_x[1]
                    last_disconnect_rot_no_x[2] = 0.0
                    last_segment_transform[:3, :3] = R2.from_euler('xyz', last_disconnect_rot_no_x, degrees=True).as_matrix()                    
                    last_segment_transform[:3, 3] = np.concatenate([[last_disconnect_cumulative_xy[0], 0.0], [last_disconnect_cumulative_z]]) / 1000.0 

                    z_angle_diff = reconnection_alignment_result['z_angle_correction']
                    xy_translation = reconnection_alignment_result['xy_translation']
                    xy_rotation_correction = reconnection_alignment_result['xy_rotation_correction']
                    
                    inverse_transform = np.eye(4)
                    inverse_xy_rot = R2.from_euler('xy', -xy_rotation_correction, degrees=True).as_matrix()
                    inverse_z_rot = R2.from_euler('z', -z_angle_diff, degrees=True).as_matrix()
                    inverse_transform[:3, :3] = inverse_z_rot @ inverse_xy_rot
                    inverse_transform[:3, 3] = np.array([-xy_translation[0], -xy_translation[1], 0.0])
                    
                    corrected_last_transform = last_segment_transform
                    
                    normalflow.transformation_chain.append(corrected_last_transform)
                    
                    cumulative_xy = np.array([0.0, 0.0])
                    cumulative_rot = np.array([0.0, 0.0, 0.0])
                    try:
                        start_new_fusion_session()
                    except Exception as e:
                        print(f"[Fusion] Failed to start new session: {e}")                    
                    
                    normalflow.current_segment_base = np.eye(4)
                    for T in normalflow.transformation_chain:
                        normalflow.current_segment_base = normalflow.current_segment_base @ T                             
            else:
                print(f"❌ Reconnection failed: {match_info}")
                reconnection_alignment_result = None
        else:
            print("❌ Cannot get current contact point info") 
    
    # 4. Output results
    print(f"[Cumulative] θ_X={cumulative_rot[0]:+.2f}°, θ_Y={cumulative_rot[1]:+.2f}°, θ_Z={cumulative_rot[2]:+.2f}°")
    print(f"[ΔX] Current={delta_xy_mm[0]:+.3f} mm, Cumulative={cumulative_xy[0]:+.3f} mm")
    print(f"[ΔY] Current={delta_xy_mm[1]:+.3f} mm, Cumulative={cumulative_xy[1]:+.3f} mm")
    
    R_mat = R2.from_euler('xyz', cumulative_rot, degrees=True).as_matrix()
    R_mat_rec = R2.from_euler('xyz', [cumulative_rot[0], -cumulative_rot[1], 0.0], degrees=True).as_matrix()
    T_ours = np.eye(4)       
    T_ours[:3, :3] = R_mat
    T_ours[:3, 3] = cumulative_xyz / 1000.0
    T_current_segment = np.eye(4)
    T_current_segment[:3, :3] = R_mat_rec
    T_current_segment[:3, 3] = np.array([cumulative_xy[0], 0.0, -z_curr_mm]) / 1000.0
    T_translation_only = np.eye(4)
    T_translation_only[:2, 3] = np.array([0.0, 0.0]) / 1000.0     
    
    if hasattr(normalflow, 'current_segment_base'):
        T_reconstruct = normalflow.current_segment_base @ T_current_segment
    else:
        T_reconstruct = T_current_segment
        normalflow.current_segment_base = np.eye(4)
    
    # 5. Reconstruction
    if not reconstruction_paused:
        _, current_contact_info, _ = flat_frame_calculator._get_contact_points_3d_with_matches(
            keyframe_C, C_tar, keyframe_H, H_tar, ppmm)    
        if not hasattr(normalflow, 'fusion_initialized'):
            initialize_multiframe_fusion(max_frames=200, voxel_size=0.00045, accumulate_all=True)
            normalflow.fusion_initialized = True

        if not hasattr(normalflow, 'frame_counter'):
            normalflow.frame_counter = 0

        if useful_info is True:
            add_frame_to_fusion(T_reconstruct, normalflow.frame_counter, current_contact_info, geometric_center=geometric_center)
            normalflow.frame_counter += 1    
        
        if normalflow.frame_counter % 500 == 0:
            mesh = generate_fusion_mesh(method='ball_pivoting')
            if mesh is not None:
                o3d.io.write_triangle_mesh(f"mesh_frame_{normalflow.frame_counter}.ply", mesh)
    
    # 6. Keyframe update
    update_threshold = 0.5
    combined = np.linalg.norm(delta_xy_mm * 1000) + np.linalg.norm(theta_xyz[:2])    
    if combined > update_threshold:
        print("[Keyframe] Updating keyframe...")
        keyframe_image = image_tar.copy()
        keyframe_H = H_tar.copy()
        keyframe_C = C_tar.copy()    
        prev_centroid = geometric_center.copy()

    # tar_T_ref = T_ours.copy()
    return tar_T_ref, T_ours, geometric_center