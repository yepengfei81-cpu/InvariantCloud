import numpy as np

class MarkerInterpolator:
    def __init__(self, marker_shape=(7, 9), target_shape=(19, 25)):
        self.marker_rows, self.marker_cols = marker_shape
        self.target_rows, self.target_cols = target_shape
        # Auto-calculate step size
        self.row_step = (self.target_rows - 1) // (self.marker_rows - 1)
        self.col_step = (self.target_cols - 1) // (self.marker_cols - 1)
        # Precompute interpolation relationships
        self._precompute_interpolation_map()
    
    def _precompute_interpolation_map(self):
        self.interpolation_map = {}  # {global_idx: (interpolation_type, source_indices, weights)}
        for r in range(self.target_rows):
            for c in range(self.target_cols):
                global_idx = r * self.target_cols + c + 1  # Index starts from 1
                # Check if this is a marker position
                if r % self.row_step == 0 and c % self.col_step == 0:
                    marker_r = r // self.row_step
                    marker_c = c // self.col_step
                    if marker_r < self.marker_rows and marker_c < self.marker_cols:
                        marker_idx = marker_r * self.marker_cols + marker_c
                        self.interpolation_map[global_idx] = ('marker', [marker_idx], [1.0])
                    continue
                # Calculate interpolation relationships
                interp_type, source_indices, weights = self._compute_interpolation(r, c)
                if interp_type is not None:
                    self.interpolation_map[global_idx] = (interp_type, source_indices, weights)
    
    def _compute_interpolation(self, r, c):
        # Find nearest marker grid position
        marker_r_base = (r // self.row_step) * self.row_step
        marker_c_base = (c // self.col_step) * self.col_step
        # Calculate relative position within sub-grid
        sub_r = r % self.row_step
        sub_c = c % self.col_step
        # Get surrounding marker indices
        corners = self._get_marker_corners(marker_r_base, marker_c_base)
        if not corners:
            return None, None, None
        # Determine interpolation method based on position type
        if sub_r == 0:
            # On horizontal line
            if self.col_step > 1 and 0 < sub_c < self.col_step:
                left_idx, right_idx = corners['left'], corners['right']
                if left_idx is not None and right_idx is not None:
                    w_left = (self.col_step - sub_c) / self.col_step
                    w_right = sub_c / self.col_step
                    return 'horizontal', [left_idx, right_idx], [w_left, w_right]
        elif sub_c == 0:
            # On vertical line
            if self.row_step > 1 and 0 < sub_r < self.row_step:
                top_idx, bottom_idx = corners['top'], corners['bottom']
                if top_idx is not None and bottom_idx is not None:
                    w_top = (self.row_step - sub_r) / self.row_step
                    w_bottom = sub_r / self.row_step
                    return 'vertical', [top_idx, bottom_idx], [w_top, w_bottom]
        else:
            # Interior point, bilinear interpolation
            tl, tr, bl, br = corners['tl'], corners['tr'], corners['bl'], corners['br']
            if all(idx is not None for idx in [tl, tr, bl, br]):
                u = sub_c / self.col_step  # [0, 1]
                v = sub_r / self.row_step  # [0, 1]
                w_tl = (1-u) * (1-v)
                w_tr = u * (1-v)
                w_bl = (1-u) * v
                w_br = u * v
                return 'bilinear', [tl, tr, bl, br], [w_tl, w_tr, w_bl, w_br]
        return None, None, None
    
    def _get_marker_corners(self, base_r, base_c):
        corners = {}
        marker_r = base_r // self.row_step
        marker_c = base_c // self.col_step
        # Top-left corner
        if marker_r < self.marker_rows and marker_c < self.marker_cols:
            corners['tl'] = marker_r * self.marker_cols + marker_c
        else:
            corners['tl'] = None
        # Top-right corner
        if marker_r < self.marker_rows and marker_c + 1 < self.marker_cols:
            corners['tr'] = marker_r * self.marker_cols + (marker_c + 1)
        else:
            corners['tr'] = None
        # Bottom-left corner
        if marker_r + 1 < self.marker_rows and marker_c < self.marker_cols:
            corners['bl'] = (marker_r + 1) * self.marker_cols + marker_c
        else:
            corners['bl'] = None
        # Bottom-right corner
        if marker_r + 1 < self.marker_rows and marker_c + 1 < self.marker_cols:
            corners['br'] = (marker_r + 1) * self.marker_cols + (marker_c + 1)
        else:
            corners['br'] = None
        # Simplified aliases
        corners['left'] = corners['tl']
        corners['right'] = corners['tr']
        corners['top'] = corners['tl']
        corners['bottom'] = corners['bl']
        return corners
    
    def interpolate_markers_to_grid(self, marker_pts_2d, marker_indices, H, ppmm):
        from .registration import pixels_to_world, world_to_pixels
        
        # Convert markers to 3D coordinates
        marker_pts_3d = pixels_to_world(marker_pts_2d, H, ppmm)
        # Create marker index mapping
        marker_dict = {idx: i for i, idx in enumerate(marker_indices)}
        # Store all grid points
        all_points_3d = []
        all_indices = []
        # Generate all grid points in index order
        for global_idx in range(1, self.target_rows * self.target_cols + 1):
            if global_idx in self.interpolation_map:
                interp_type, source_indices, weights = self.interpolation_map[global_idx]
                if interp_type == 'marker':
                    marker_idx = source_indices[0]
                    point_3d = marker_pts_3d[marker_idx].copy()
                else:
                    point_3d = self._interpolate_3d_point(
                        marker_pts_3d, source_indices, weights
                    )
                all_points_3d.append(point_3d)
                all_indices.append(global_idx)
            else:
                # Boundary case, use nearest marker
                nearest_marker_idx = self._find_nearest_marker_index(global_idx)
                if nearest_marker_idx < len(marker_pts_3d):
                    point_3d = marker_pts_3d[nearest_marker_idx].copy()
                else:
                    point_3d = marker_pts_3d[0].copy()  # Fallback to first point
                all_points_3d.append(point_3d)
                all_indices.append(global_idx)
        # Convert back to 2D coordinates
        all_points_3d = np.array(all_points_3d)
        all_pts_2d = world_to_pixels(all_points_3d, H, ppmm)
        all_indices = np.array(all_indices, dtype=np.int32)
        return all_pts_2d, all_indices
    
    def _interpolate_3d_point(self, marker_pts_3d, source_indices, weights):
        weights = np.array(weights)
        source_points = marker_pts_3d[source_indices]  # (n, 3)
        interpolated_point = np.sum(source_points * weights[:, np.newaxis], axis=0)
        return interpolated_point
    
    def _find_nearest_marker_index(self, global_idx):
        r = (global_idx - 1) // self.target_cols
        c = (global_idx - 1) % self.target_cols
        marker_r = min(round(r / self.row_step), self.marker_rows - 1)
        marker_c = min(round(c / self.col_step), self.marker_cols - 1)
        return marker_r * self.marker_cols + marker_c

# Global interpolator instance
_global_interpolator = None

def get_interpolator():
    global _global_interpolator
    if _global_interpolator is None:
        _global_interpolator = MarkerInterpolator(marker_shape=(7, 9), target_shape=(31, 41))
    return _global_interpolator

def interpolate_markers_to_475_grid(marker_pts_2d, marker_indices, H, ppmm):
    interpolator = get_interpolator()
    return interpolator.interpolate_markers_to_grid(marker_pts_2d, marker_indices, H, ppmm)

def reset_interpolator():
    global _global_interpolator
    _global_interpolator = None