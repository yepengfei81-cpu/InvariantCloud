import numpy as np
from scipy.spatial import cKDTree
import cv2

class GridLineAssigner:
    """Grid line-based marker index assigner"""
    
    def __init__(self, grid_shape=(7, 9)):
        self.rows, self.cols = grid_shape
        self.horizontal_lines = None  # Y coordinates of horizontal lines
        self.vertical_lines = None    # X coordinates of vertical lines
        self.is_initialized = False
        self.grid_indices = self._precompute_grid_indices()
    
    def _precompute_grid_indices(self):
        indices = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                global_row = r * 5  # Markers at rows 0,3,6,9,12,15,18
                global_col = c * 5  # Markers at cols 0,3,6,9,12,15,18,21,24
                global_475_idx = global_row * 41 + global_col + 1
                indices[r, c] = global_475_idx
        return indices
    
    def initialize_grid_lines(self, markers):
        if len(markers) != self.rows * self.cols:
            print(f"Warning: marker count mismatch, expected {self.rows * self.cols}, got {len(markers)}")
            return False
        
        # Use traditional method to get ideal grid layout
        markers = np.array(markers, dtype=np.float32)
        
        # Sort by Y coordinate and group
        y_sorted_indices = np.argsort(markers[:, 1])
        y_sorted_pts = markers[y_sorted_indices]
        
        # Calculate average Y coordinate per row (vectorized)
        row_points = y_sorted_pts.reshape(self.rows, self.cols, 2)
        self.horizontal_lines = np.mean(row_points[:, :, 1], axis=1)
        
        # Sort each row by X coordinate
        for r in range(self.rows):
            row_x_indices = np.argsort(row_points[r, :, 0])
            row_points[r] = row_points[r][row_x_indices]
        
        # Calculate average X coordinate per column (vectorized)
        self.vertical_lines = np.mean(row_points[:, :, 0], axis=0)
        
        self.is_initialized = True
        print(f"[Grid lines initialized] {len(self.horizontal_lines)} horizontal lines and {len(self.vertical_lines)} vertical lines established")
        return True
    
    def find_best_grid_positions(self, points):
        points = np.array(points, dtype=np.float32)
        n_points = len(points)
        
        h_distances = np.abs(points[:, 1:2] - self.horizontal_lines[np.newaxis, :])
        best_rows = np.argmin(h_distances, axis=1)
        
        v_distances = np.abs(points[:, 0:1] - self.vertical_lines[np.newaxis, :])
        best_cols = np.argmin(v_distances, axis=1)
        
        best_h_distances = h_distances[np.arange(n_points), best_rows]
        best_v_distances = v_distances[np.arange(n_points), best_cols]
        total_distances = best_h_distances + best_v_distances
        
        return best_rows, best_cols, total_distances
    
    def assign_indices_optimized(self, current_markers):
        if not self.is_initialized:
            print("Error: Grid lines not initialized")
            return None, None
        
        current_markers = np.array(current_markers, dtype=np.float32)
        n_markers = len(current_markers)
        
        best_rows, best_cols, distances = self.find_best_grid_positions(current_markers)
        
        assigned_grid = np.full((self.rows, self.cols), -1, dtype=np.int32)  # -1 means unassigned
        point_to_grid = {}
        
        sorted_indices = np.argsort(distances)
        
        for point_idx in sorted_indices:
            row, col = best_rows[point_idx], best_cols[point_idx]
            
            if assigned_grid[row, col] != -1:
                row, col = self._find_nearest_empty_position(
                    assigned_grid, current_markers[point_idx], row, col
                )
            
            if row != -1 and col != -1:
                assigned_grid[row, col] = point_idx
                point_to_grid[point_idx] = (row, col)
        
        assignments = []
        for point_idx, (row, col) in point_to_grid.items():
            grid_index = self.grid_indices[row, col]
            assignments.append((grid_index, current_markers[point_idx]))
        
        assignments.sort(key=lambda x: x[0])
        
        n_assigned = len(assignments)
        marker_pts_2d = np.zeros((n_assigned, 2), dtype=np.float32)
        marker_indices = np.zeros(n_assigned, dtype=np.int32)
        
        for i, (grid_index, point_coords) in enumerate(assignments):
            marker_indices[i] = grid_index
            marker_pts_2d[i] = point_coords
        
        return marker_pts_2d, marker_indices
    
    def _find_nearest_empty_position(self, assigned_grid, point, preferred_row, preferred_col):
        row_indices, col_indices = np.meshgrid(
            np.arange(self.rows), np.arange(self.cols), indexing='ij'
        )
        
        empty_mask = (assigned_grid == -1)
        
        if not np.any(empty_mask):
            return -1, -1
        
        h_distances = np.abs(self.horizontal_lines[row_indices] - point[1])
        v_distances = np.abs(self.vertical_lines[col_indices] - point[0])
        total_distances = h_distances + v_distances
        
        total_distances[~empty_mask] = np.inf
        
        min_pos = np.unravel_index(np.argmin(total_distances), total_distances.shape)
        return min_pos[0], min_pos[1]

def assign_marker_indices_7x9_traditional(markers, grid_shape=(7, 9)):
    rows, cols = grid_shape
    expected_markers = rows * cols  # 63
    
    markers = np.array(markers, dtype=np.float32)
    
    if len(markers) != expected_markers:
        if len(markers) < expected_markers:
            shortage = expected_markers - len(markers)
            repeated_pts = np.tile(markers[-1:], (shortage, 1))
            markers = np.vstack([markers, repeated_pts])
        else:
            markers = markers[:expected_markers]
    
    y_sorted_indices = np.argsort(markers[:, 1])
    y_sorted_pts = markers[y_sorted_indices]
    
    grid_points = y_sorted_pts.reshape(rows, cols, 2)
    
    for r in range(rows):
        x_sorted_indices = np.argsort(grid_points[r, :, 0])
        grid_points[r] = grid_points[r][x_sorted_indices]
    
    row_indices = np.arange(rows)[:, np.newaxis] * 3
    col_indices = np.arange(cols)[np.newaxis, :] * 3
    global_indices = row_indices * 25 + col_indices + 1
    
    marker_pts_2d = grid_points.reshape(-1, 2)
    marker_indices = global_indices.reshape(-1)
    
    return marker_pts_2d.astype(np.float32), marker_indices.astype(np.int32)

def assign_marker_indices_with_grid_lines(current_markers, grid_shape=(7, 9)):
    if not hasattr(assign_marker_indices_with_grid_lines, 'grid_assigner'):
        assign_marker_indices_with_grid_lines.grid_assigner = GridLineAssigner(grid_shape)
    
    grid_assigner = assign_marker_indices_with_grid_lines.grid_assigner
    
    rows, cols = grid_shape
    expected_markers = rows * cols  # 63
    
    if len(current_markers) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32)
    
    current_markers = np.array(current_markers, dtype=np.float32)
    
    if len(current_markers) != expected_markers:
        print(f"Warning: marker count mismatch, expected {expected_markers}, got {len(current_markers)}")
        if len(current_markers) < expected_markers:
            shortage = expected_markers - len(current_markers)
            repeated_pts = np.tile(current_markers[-1:], (shortage, 1))
            current_markers = np.vstack([current_markers, repeated_pts])
        else:
            current_markers = current_markers[:expected_markers]
    
    if not grid_assigner.is_initialized:
        if grid_assigner.initialize_grid_lines(current_markers):
            return assign_marker_indices_7x9_traditional(current_markers, grid_shape)
        else:
            print("Error: Grid lines initialization failed")
            return assign_marker_indices_7x9_traditional(current_markers, grid_shape)
    
    marker_pts_2d, marker_indices = grid_assigner.assign_indices_optimized(current_markers)
    
    if marker_pts_2d is None:
        print("Warning: Grid line assignment failed, falling back to traditional method")
        return assign_marker_indices_7x9_traditional(current_markers, grid_shape)
    
    return marker_pts_2d, marker_indices

def reset_grid_assigner():
    """Reset grid line assigner (call when starting new sequence)"""
    if hasattr(assign_marker_indices_with_grid_lines, 'grid_assigner'):
        assign_marker_indices_with_grid_lines.grid_assigner = GridLineAssigner((7, 9))