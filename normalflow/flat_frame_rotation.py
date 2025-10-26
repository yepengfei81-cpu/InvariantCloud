import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

class FlatFrameRotationCalculator:
    def __init__(self):
        self.flat_frame_pts_2d = None
        self.flat_frame_indices = None
        self.flat_frame_H = None
        self.flat_frame_C = None
        self.initialized = False
        
        self.prev_primary_axis = None
        self.filtered_z_angle = None
        self.filter_xy_angle = None
        
    @staticmethod
    def rot_to_euler_xyz(R):
        return R_scipy.from_matrix(R).as_euler('xyz', degrees=True)

    def initialize_flat_frame(self, pts_2d, indices, H_map, C_map, verbose=False):
        self.flat_frame_pts_2d = pts_2d.copy()
        self.flat_frame_indices = indices.copy()
        self.flat_frame_H = H_map.copy()
        self.flat_frame_C = C_map.copy()
        self.initialized = True
       
        if verbose:
            print(f"[FlatFrame] Initialized with {len(pts_2d)} points")

    def visualize_prev_and_current_frame_contact_points(self, C_prev, C_curr, 
                                                    H_prev, H_curr, ppmm,
                                                    title="Prev vs Current Frame Contact Points",
                                                    verbose=False):
        if not self.initialized:
            return 0.0

        h, w = self.flat_frame_C.shape
        
        # Create side-by-side canvas
        canvas = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
        
        # Get 3D point information
        prev_3d_info, curr_3d_info, matched_pairs = self._get_contact_points_3d_with_matches(C_prev, C_curr, H_prev, H_curr, ppmm)
        
        # Draw left side (previous frame)
        left_display = canvas[:, :w]
        left_display[C_prev > 0] = [100, 50, 0]
        prev_angle = None
        if prev_3d_info:
            prev_angle = self._draw_points_and_axes_stable(left_display, prev_3d_info, H_prev, ppmm, is_current=False)
        
        # Draw right side (current frame)
        right_display = canvas[:, w+20:]
        right_display[C_curr > 0] = [0, 100, 200]
        curr_angle = None
        stable_centroid = None
        if curr_3d_info:
            curr_angle, stable_centroid = self._draw_points_and_axes_stable(right_display, curr_3d_info, H_curr, ppmm, is_current=True)
        
        if matched_pairs:
            self._draw_matched_point_connections(canvas, matched_pairs, w)
    
        cv2.putText(canvas, "Prev Frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, "Current Frame", (w + 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(title, canvas)
        return curr_angle, stable_centroid if curr_angle is not None else 0.0

    def _draw_points_and_axes_stable(self, display, info_3d, H, ppmm, is_current=True):
        from normalflow.registration import world_to_pixels
        
        # Draw contact points
        for x, y in info_3d['points_2d']:
            cv2.circle(display, (int(x), int(y)), 4, (255, 100, 0), -1)
        
        points_3d = info_3d['points_3d']
        if len(points_3d) < 3:
            return None
        
        # Stable PCA computation
        primary_axis, secondary_axis, primary_angle = self._compute_stable_pca(points_3d, is_current)
        
        if primary_axis is None:
            return None
        
        # Compute centroid
        centroid_xy = np.mean(points_3d[:, :2], axis=0)
        centroid_z = np.min(points_3d[:, 2])
        centroid = np.array([centroid_xy[0], centroid_xy[1], centroid_z])
        
        # Compute axis length
        x_range = points_3d[:, 0].max() - points_3d[:, 0].min()
        y_range = points_3d[:, 1].max() - points_3d[:, 1].min()
        axis_length = max(x_range, y_range) * 0.6
        
        # Convert coordinates and draw
        centroid_2d = world_to_pixels(np.array([[centroid[0], centroid[1], 0]]), H, ppmm)[0]
        
        # Compute axis endpoints
        axes_3d = np.array([
            [centroid[0] + primary_axis[0] * axis_length, centroid[1] + primary_axis[1] * axis_length, 0],
            [centroid[0] - primary_axis[0] * axis_length, centroid[1] - primary_axis[1] * axis_length, 0],
            [centroid[0] + secondary_axis[0] * axis_length, centroid[1] + secondary_axis[1] * axis_length, 0],
            [centroid[0] - secondary_axis[0] * axis_length, centroid[1] - secondary_axis[1] * axis_length, 0]
        ])
        axes_2d = world_to_pixels(axes_3d, H, ppmm)
        
        # Draw primary axis (red)
        cv2.line(display, tuple(axes_2d[1].astype(int)), tuple(axes_2d[0].astype(int)), (0, 0, 255), 2)
        # Draw secondary axis (green)
        cv2.line(display, tuple(axes_2d[3].astype(int)), tuple(axes_2d[2].astype(int)), (0, 255, 0), 2)
        # Draw centroid (white)
        cv2.circle(display, tuple(centroid_2d.astype(int)), 5, (255, 255, 255), -1)
        
        return primary_angle, centroid
    
    def _compute_stable_pca(self, points_3d, is_current):
        # 1. Data preprocessing: remove outliers
        points_2d = points_3d[:, :2]
        centroid = np.mean(points_2d, axis=0)
        
        # Calculate distance from each point to centroid
        distances = np.linalg.norm(points_2d - centroid, axis=1)
        # Remove points beyond 2 standard deviations
        threshold = np.mean(distances) + 2 * np.std(distances)
        valid_mask = distances <= threshold
        
        if np.sum(valid_mask) < 3:
            valid_mask = np.ones(len(points_2d), dtype=bool)
        
        filtered_points = points_2d[valid_mask]
        centroid = np.mean(filtered_points, axis=0)
        centered_points = filtered_points - centroid
        
        # 2. Compute covariance matrix using SVD (more stable than eigenvalue decomposition)
        if len(centered_points) < 2:
            return None, None, None
            
        U, s, Vt = np.linalg.svd(centered_points, full_matrices=False)
        primary_axis = Vt[0]
        secondary_axis = Vt[1] if len(Vt) > 1 else np.array([-primary_axis[1], primary_axis[0]])
        
        # 3. Direction consistency check
        if self.prev_primary_axis is not None and is_current:
            dot_product = np.dot(primary_axis, self.prev_primary_axis)
            if dot_product < 0:
                primary_axis = -primary_axis
                secondary_axis = -secondary_axis
        
        primary_angle = np.degrees(np.arctan2(primary_axis[1], primary_axis[0]))
        alpha = 0.5  # Filter coefficient
        if is_current:
            if self.filtered_z_angle is None:
                self.filtered_z_angle = primary_angle
            else:
                # Handle angle wrap-around (±180° crossing)
                delta = primary_angle - self.filtered_z_angle
                if delta > 180:
                    primary_angle -= 360
                elif delta < -180:
                    primary_angle += 360
                self.filtered_z_angle = alpha * primary_angle + (1 - alpha) * self.filtered_z_angle
            self.prev_primary_axis = primary_axis.copy()
            return primary_axis, secondary_axis, self.filtered_z_angle
        else:
            return primary_axis, secondary_axis, primary_angle

    def _get_contact_points_3d_with_matches(self, C_prev, C_curr, H_prev, H_curr, ppmm):
        from normalflow.registration import pixels_to_world

        h, w = self.flat_frame_C.shape

        # Use vectorized operations
        x_coords = self.flat_frame_pts_2d[:, 0].astype(int)
        y_coords = self.flat_frame_pts_2d[:, 1].astype(int)
        valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)

        # Previous frame contact points
        prev_contact_mask = valid_mask & (C_prev[y_coords, x_coords] > 0)
        prev_3d_info = None
        prev_indices = None
        if np.any(prev_contact_mask):
            prev_points_2d = self.flat_frame_pts_2d[prev_contact_mask]
            prev_points_3d = pixels_to_world(prev_points_2d, H_prev, ppmm)
            prev_indices = self.flat_frame_indices[prev_contact_mask]
            prev_3d_info = {
                'points_2d': prev_points_2d,
                'points_3d': prev_points_3d,
                'indices': prev_indices,
                'count': len(prev_points_2d)
            }

        # Current frame contact points
        curr_contact_mask = valid_mask & (C_curr[y_coords, x_coords] > 0)
        curr_3d_info = None
        curr_indices = None

        if np.any(curr_contact_mask):
            curr_points_2d = self.flat_frame_pts_2d[curr_contact_mask]
            curr_points_3d = pixels_to_world(curr_points_2d, H_curr, ppmm)
            curr_indices = self.flat_frame_indices[curr_contact_mask]
            curr_3d_info = {
                'points_2d': curr_points_2d,
                'points_3d': curr_points_3d,
                'indices': curr_indices,
                'count': len(curr_points_2d)
            }

        # Find matched point pairs
        matched_pairs = []
        if prev_3d_info and curr_3d_info:
            prev_idx_set = set(prev_indices)
            curr_idx_set = set(curr_indices)
            common_indices = prev_idx_set & curr_idx_set

            for common_idx in common_indices:
                prev_pos = np.where(prev_indices == common_idx)[0][0]
                curr_pos = np.where(curr_indices == common_idx)[0][0]
                matched_pairs.append({
                    'index': common_idx,
                    'prev_pt_2d': prev_points_2d[prev_pos],
                    'curr_pt_2d': curr_points_2d[curr_pos],
                    'prev_pt_3d': prev_points_3d[prev_pos],
                    'curr_pt_3d': curr_points_3d[curr_pos]
                })

        return prev_3d_info, curr_3d_info, matched_pairs

    def _draw_matched_point_connections(self, canvas, matched_pairs, left_width):
        gap = 20
        
        for pair in matched_pairs:
            left_pt = (int(pair['prev_pt_2d'][0]), int(pair['prev_pt_2d'][1]))
            right_pt = (int(pair['curr_pt_2d'][0]) + left_width + gap, int(pair['curr_pt_2d'][1]))
            
            cv2.line(canvas, left_pt, right_pt, (0, 255, 255), 1)
            cv2.circle(canvas, left_pt, 2, (0, 255, 255), 1)
            cv2.circle(canvas, right_pt, 2, (0, 255, 255), 1)
        
    def compute_xy_rotation_kabsch_by_flat_frame_points(self, C_prev, C_curr, H_prev, H_curr, ppmm, min_points=8, verbose=False):
        _, _, matched_pairs = self._get_contact_points_3d_with_matches(
            C_prev, C_curr, H_prev, H_curr, ppmm
        )
        if not matched_pairs or len(matched_pairs) < min_points:
            if verbose:
                print(f"[Kabsch] Insufficient matched points: {len(matched_pairs)}")
            return np.eye(3), np.zeros(3), 0

        # Collect points in order
        src_pts = []
        dst_pts = []
        for pair in matched_pairs:
            src_pts.append(pair['prev_pt_3d'])
            dst_pts.append(pair['curr_pt_3d'])
        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)

        # Use full XYZ points
        src_mean = src_pts.mean(axis=0)
        dst_mean = dst_pts.mean(axis=0)
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean

        # Kabsch algorithm (SVD)
        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        filter_alpha = 0.9
        theta_xyz = self.rot_to_euler_xyz(np.array(R, copy=True))
        self.filter_xy_angle = filter_alpha * theta_xyz + (1 - filter_alpha) * (self.filter_xy_angle if self.filter_xy_angle is not None else theta_xyz)

        return R, theta_xyz, len(matched_pairs)