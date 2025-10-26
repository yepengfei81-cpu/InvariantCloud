import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class ContactRegionMatcher:
    def __init__(self, icp_threshold=0.001, rotation_threshold=5.0, translation_threshold=0.002,
                 grid_cols=41, grid_rows=31):
        self.icp_threshold = icp_threshold
        self.rotation_threshold = rotation_threshold
        self.translation_threshold = translation_threshold
        self.last_contact_info = None
        self.grid_cols = int(grid_cols)
        self.grid_rows = int(grid_rows)        
        print("🔧 Contact reconnection matcher initialized")

    def _idx_to_grid(self, idx):
        """Index (1-based) -> (col, row)"""
        i0 = int(idx) - 1
        col = i0 % self.grid_cols
        row = i0 // self.grid_cols
        return col, row

    def _grid_to_idx(self, col, row):
        """(col, row) -> Index, return None if out of bounds"""
        if col < 0 or col >= self.grid_cols or row < 0 or row >= self.grid_rows:
            return None
        return row * self.grid_cols + col + 1

    def _estimate_grid_offset(self, matched_pairs, verbose=False):
        if len(matched_pairs) < 3:
            if verbose:
                print(f"   [Grid offset estimation] Too few pairs ({len(matched_pairs)}), cannot estimate")
            return None, None, 0.0, 999.0

        deltas = []
        for prev_idx, curr_idx in matched_pairs:
            col_p, row_p = self._idx_to_grid(prev_idx)
            col_c, row_c = self._idx_to_grid(curr_idx)
            delta = (col_p - col_c, row_p - row_c)
            deltas.append(delta)
            if verbose and len(deltas) <= 5:
                print(f"      Pair {prev_idx}->{curr_idx}: ({col_p},{row_p})-({col_c},{row_c}) = Δ{delta}")

        # Median per dimension
        dx_med = int(round(np.median([d[0] for d in deltas])))
        dy_med = int(round(np.median([d[1] for d in deltas])))

        # Support ratio: proportion matching median (dx_med, dy_med)
        support = sum(1 for d in deltas if d == (dx_med, dy_med))
        support_ratio = support / len(deltas)

        # Residual: index difference after prediction using (dx_med, dy_med)
        residuals = []
        for prev_idx, curr_idx in matched_pairs:
            col_c, row_c = self._idx_to_grid(curr_idx)
            pred_idx = self._grid_to_idx(col_c + dx_med, row_c + dy_med)
            if pred_idx is not None:
                residuals.append(abs(pred_idx - prev_idx))
        median_res = np.median(residuals) if residuals else 999.0

        return dx_med, dy_med, support_ratio, median_res

    def _remap_indices_by_offset(self, current_indices, dx, dy, prev_indices_set, 
                                threshold_support=0.6, verbose=False):
        remapping = {}
        out_of_bound = 0
        not_in_prev = 0

        for curr_idx in current_indices:
            if isinstance(curr_idx, np.ndarray):
                curr_idx = int(curr_idx.item())
            else:
                curr_idx = int(curr_idx)
            
            col_c, row_c = self._idx_to_grid(curr_idx)
            pred_idx = self._grid_to_idx(col_c + dx, row_c + dy)
            
            if pred_idx is None:
                out_of_bound += 1
                continue
            
            remapping[curr_idx] = int(pred_idx)

        return remapping
                
    def save_contact_with_info(self, contact_info, z_angle, geometric_center, H_tar):
        if contact_info is None or contact_info['count'] < 10:
            print("❌ Too few feature points in contact region, cannot save")
            return False
        
        self.last_contact_info = {
            'flat_frame_points_2d': contact_info['points_2d'].copy(),
            'flat_frame_points_3d': contact_info['points_3d'].copy(),
            'flat_frame_indices': contact_info['indices'].copy(),
            'center_3d': np.mean(contact_info['points_3d'], axis=0),
            'z_angle': z_angle,
            'geometric_center': geometric_center.copy(),
            'H_tar': H_tar.copy(),
            'timestamp': np.datetime64('now')
        }
        return True

    def _icp_matched_index_pairs(self, source_points, target_points,
                                 source_indices, target_indices,
                                 icp_result, max_dist=0.001, mutual_check=True):
        src = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
        tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
        src_ids = np.asarray(source_indices).reshape(-1)
        tgt_ids = np.asarray(target_indices).reshape(-1)

        if src.shape[0] == 0 or tgt.shape[0] == 0:
            return []
        if src_ids.shape[0] != src.shape[0] or tgt_ids.shape[0] != tgt.shape[0]:
            return []

        # Filter NaN/Inf
        src_finite = np.isfinite(src).all(axis=1)
        tgt_finite = np.isfinite(tgt).all(axis=1)
        src = src[src_finite]
        src_ids = src_ids[src_finite]
        tgt = tgt[tgt_finite]
        tgt_ids = tgt_ids[tgt_finite]

        if src.shape[0] == 0 or tgt.shape[0] == 0:
            return []

        # Apply ICP transformation to align source to target coordinate system
        T = np.asarray(icp_result.transformation, dtype=np.float64)
        src_h = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float64)])
        src_aligned = (T @ src_h.T).T[:, :3]

        src_finite2 = np.isfinite(src_aligned).all(axis=1)
        src_aligned = src_aligned[src_finite2]
        src_ids = src_ids[src_finite2]
        if src_aligned.shape[0] == 0:
            return []

        # Build KD tree using PointCloud
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(tgt))
        kdt_tgt = o3d.geometry.KDTreeFlann(tgt_pcd)

        if mutual_check:
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(src_aligned))
            kdt_src = o3d.geometry.KDTreeFlann(src_pcd)

        max_d2 = float(max_dist) * float(max_dist)
        pairs = []
        used_tgt = set()

        for i, p in enumerate(src_aligned):
            q = np.asarray(p, dtype=np.float64).reshape(3)
            try:
                k, idxs, d2 = kdt_tgt.search_knn_vector_3d(q, 1)
            except Exception:
                continue

            if k == 1 and d2[0] <= max_d2:
                j = int(idxs[0])
                if j in used_tgt:
                    continue
                if mutual_check:
                    try:
                        k2, idxs2, d22 = kdt_src.search_knn_vector_3d(tgt[j], 1)
                        if not (k2 == 1 and int(idxs2[0]) == i):
                            continue
                    except Exception:
                        continue
                pairs.append((int(tgt_ids[j]), int(src_ids[i])))
                used_tgt.add(j)

        return pairs
    
    def match_contact_with_info(self, current_contact_info, z_angle, geometric_center, H_tar, flat_frame_vis_data=None):
        if self.last_contact_info is None:
            print("❌ No saved contact information")
            return False, np.eye(4), "No saved contact information", None
        
        if current_contact_info is None or current_contact_info['count'] < 10:
            print("❌ Too few feature points in current contact region")
            return False, np.eye(4), "Too few feature points in current contact region", None
        
        last_info = self.last_contact_info
        try:
            # Step 1: Z-axis angle alignment - rotate previous saved point cloud to current angle
            z_angle_diff = z_angle - last_info['z_angle']
            print(f"   Z angle difference: {z_angle_diff:.2f}° (current {z_angle:.2f}° - previous {last_info['z_angle']:.2f}°)")
            
            last_points_3d = last_info['flat_frame_points_3d']
            last_center_3d = np.array([last_info['geometric_center'][0], last_info['geometric_center'][1], 0.0])
            
            z_rotation_matrix = self._create_z_rotation_matrix(z_angle_diff)
            
            last_points_centered = last_points_3d - last_center_3d
            last_points_rotated = self._apply_rotation(last_points_centered, z_rotation_matrix)
            last_points_z_aligned = last_points_rotated + last_center_3d
            
            print(f"   Step 1: Z-axis alignment complete, rotated previous point cloud by {z_angle_diff:.2f}°")
            
            # Step 2: XY translation alignment
            last_center_after_rotation = np.mean(last_points_z_aligned, axis=0)
            current_center_3d = np.array([geometric_center[0], geometric_center[1], 0.0])
            current_center_xy = current_center_3d[:2]
            last_rotated_center_xy = last_center_after_rotation[:2]
            
            xy_translation = current_center_xy - last_rotated_center_xy
            print(f"   XY translation offset: [{xy_translation[0]*1000:.2f}, {xy_translation[1]*1000:.2f}] mm")
            
            translation_3d = np.array([xy_translation[0], xy_translation[1], 0.0])
            last_points_transformed = last_points_z_aligned + translation_3d
            print(f"   Step 2: XY translation alignment complete")
            
            # Step 3: ICP fine alignment
            current_points_3d = current_contact_info['points_3d']
            icp_result = self._perform_icp(current_points_3d, last_points_transformed)
            # matched_pairs = self._icp_matched_index_pairs(
            #     source_points=current_points_3d,
            #     target_points=last_points_transformed,
            #     source_indices=current_contact_info['indices'],
            #     target_indices=last_info['flat_frame_indices'],
            #     icp_result=icp_result,
            #     max_dist=0.001,
            #     mutual_check=True
            # )
            
            # # ==== New: Step 4 - Estimate integer grid offset ====
            # dx, dy, support_ratio, med_res = self._estimate_grid_offset(matched_pairs, verbose=True)
            
            # index_remapping = {}
            # remapping_quality = "none"

            # if dx is not None and support_ratio >= 0.6 and med_res < 2.0:
            #     print(f"   ✅ Grid offset reliable, using (dx={dx}, dy={dy}) for batch remapping")
            #     prev_idx_set = set(np.asarray(last_info['flat_frame_indices']).astype(int))
            #     index_remapping = self._remap_indices_by_offset(
            #         current_indices=current_contact_info['indices'],
            #         dx=dx, dy=dy,
            #         prev_indices_set=prev_idx_set,
            #         verbose=True
            #     )
            #     remapping_quality = "grid_offset"
            # else:
            #     print(f"   ⚠️  Grid offset unreliable (support rate={support_ratio:.2%}), using ICP seed mapping only")
            #     index_remapping = {int(c): int(p) for p, c in matched_pairs}
            #     remapping_quality = "icp_seed_only"

            # # ==== New: Step 5 - Visualize remapping verification ====
            # if flat_frame_vis_data and len(index_remapping) > 0:
            #     self._visualize_remapping_comparison(
            #         flat_frame_image=flat_frame_vis_data['image'],
            #         all_pts_2d=flat_frame_vis_data['all_pts_2d'],
            #         all_indices=flat_frame_vis_data['all_indices'],
            #         current_indices=current_contact_info['indices'],
            #         remapping_dict=index_remapping,
            #         prev_indices_highlight=last_info['flat_frame_indices'],
            #         remapping_quality=remapping_quality,
            #         title="Index Remapping: Original vs Remapped"
            #     )              
            # final_matched_pairs = matched_pairs 
            # if flat_frame_vis_data and len(final_matched_pairs) > 0:
            #     self._visualize_matched_indices_on_grid(
            #         flat_frame_image=flat_frame_vis_data['image'],
            #         all_pts_2d=flat_frame_vis_data['all_pts_2d'],
            #         all_indices=flat_frame_vis_data['all_indices'],
            #         matched_pairs=final_matched_pairs,
            #         title="Matched Indices on Full Grid"
            #     )                       
       
            transform_matrix = self._build_geometric_transform_matrix(
                z_rotation_matrix, translation_3d, last_center_3d, icp_result.transformation
            )

            # Step 4 (original): Overlap ratio evaluation
            last_H_map = last_info['H_tar']
            current_H_map = H_tar
            overlap_ratio = self._calculate_contour_overlap(last_points_transformed, current_points_3d, last_H_map, current_H_map)
            print(f"   Step 4 (original): Overlap evaluation - Contour overlap ratio: {overlap_ratio:.3f}")

            alignment_result = {
                'z_angle_diff': z_angle_diff,
                'z_rotation_matrix': z_rotation_matrix,
                'xy_translation': xy_translation,
                'translation_3d': translation_3d,
                'last_points_transformed': last_points_transformed,
                'icp_result': icp_result,
                'transform_matrix': transform_matrix,
                'contour_overlap_ratio': overlap_ratio
            }
        
            is_match = self._evaluate_geometric_match_quality(alignment_result)
            
            if is_match:
                icp_rotation_matrix = icp_result.transformation[:3, :3]
                xy_euler = R.from_matrix(icp_rotation_matrix.copy()).as_euler('xyz', degrees=True)
                
                match_details = {
                    'z_angle_correction': z_angle_diff,
                    'xy_translation': xy_translation,
                    'xy_rotation_correction': xy_euler[:2],
                    'icp_fitness': icp_result.fitness,
                    'icp_rmse': icp_result.inlier_rmse,
                    'last_contact_points': len(last_points_3d),
                    'current_contact_points': len(current_points_3d)
                }
                
                match_info = (f"Geometric alignment success - Z rotation: {z_angle_diff:.2f}°, "
                            f"XY translation: [{xy_translation[0]*1000:.2f}, {xy_translation[1]*1000:.2f}] mm, "
                            f"ICP: {icp_result.fitness:.3f}")
                
                print(f"✅ Match success")
                
                self._visualize_matching_result(
                    last_points_3d,
                    last_points_transformed,
                    current_points_3d,
                    match_details
                )
                
                return True, transform_matrix, match_info, match_details
            else:
                icp_rotation_matrix = icp_result.transformation[:3, :3]
                xy_euler = R.from_matrix(icp_rotation_matrix.copy()).as_euler('xyz', degrees=True)
                
                match_details = {
                    'z_angle_correction': z_angle_diff,
                    'xy_translation': xy_translation,
                    'xy_rotation_correction': xy_euler[:2],
                    'icp_fitness': icp_result.fitness,
                    'icp_rmse': icp_result.inlier_rmse,
                    'last_contact_points': len(last_points_3d),
                    'current_contact_points': len(current_points_3d),
                    'match_success': False
                }
                
                no_match_info = (f"Geometric alignment failed - Z rotation: {z_angle_diff:.2f}°, "
                            f"XY translation: [{xy_translation[0]*1000:.2f}, {xy_translation[1]*1000:.2f}] mm, "
                            f"ICP: {icp_result.fitness:.3f}, RMSE: {icp_result.inlier_rmse:.4f}")
                print(f"❌ {no_match_info}")
                
                self._visualize_matching_result(
                    last_points_3d,
                    last_points_transformed,
                    current_points_3d,
                    match_details
                )
                return False, np.eye(4), no_match_info, None
                
        except Exception as e:
            error_info = f"Geometric alignment error: {str(e)}"
            print(f"❌ {error_info}")
            return False, np.eye(4), error_info, None

    def _visualize_remapping_comparison(self, flat_frame_image, all_pts_2d, all_indices,
                                        current_indices, remapping_dict, 
                                        prev_indices_highlight=None,
                                        remapping_quality="none",
                                        title="Index Remapping Comparison"):
        h, w, _ = flat_frame_image.shape
        canvas = np.zeros((h, w * 3 + 40, 3), dtype=np.uint8)
        left_disp = canvas[:, :w]
        middle_disp = canvas[:, w+20:w*2+20]
        right_disp = canvas[:, w*2+40:]
        
        base = flat_frame_image.copy()
        for pt in all_pts_2d:
            cv2.circle(base, tuple(pt.astype(int)), 3, (0, 180, 0), -1)
        left_disp[:] = base.copy()
        middle_disp[:] = base.copy()
        right_disp[:] = base.copy()

        idx_to_pt = {}
        for idx, pt in zip(all_indices, all_pts_2d):
            if isinstance(idx, np.ndarray):
                idx = int(idx.item())
            else:
                idx = int(idx)
            idx_to_pt[idx] = pt

        curr_set = set()
        for idx in current_indices:
            if isinstance(idx, np.ndarray):
                curr_set.add(int(idx.item()))
            else:
                curr_set.add(int(idx))
        
        for idx in curr_set:
            if idx in idx_to_pt:
                pt = idx_to_pt[idx]
                cv2.circle(left_disp, tuple(pt.astype(int)), 4, (0, 255, 255), -1)

        prev_set = set()
        if prev_indices_highlight is not None:
            for idx in prev_indices_highlight:
                if isinstance(idx, np.ndarray):
                    prev_set.add(int(idx.item()))
                else:
                    prev_set.add(int(idx))
            
            for idx in prev_set:
                if idx in idx_to_pt:
                    pt = idx_to_pt[idx]
                    cv2.circle(middle_disp, tuple(pt.astype(int)), 4, (0, 0, 255), -1)

        remapped_count = 0
        for curr_idx in curr_set:
            new_idx = remapping_dict.get(curr_idx, None)
            if new_idx is not None:
                if new_idx in idx_to_pt:
                    pt = idx_to_pt[new_idx]
                    cv2.circle(right_disp, tuple(pt.astype(int)), 5, (255, 255, 0), -1)
                    remapped_count += 1

        # Annotate corner indices
        if 1 in idx_to_pt:
            top_left_pt = idx_to_pt[1]
            cv2.circle(left_disp, tuple(top_left_pt.astype(int)), 6, (255, 255, 255), 2)
            cv2.putText(left_disp, "1", (int(top_left_pt[0])+10, int(top_left_pt[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(middle_disp, tuple(top_left_pt.astype(int)), 6, (255, 255, 255), 2)
            cv2.putText(middle_disp, "1", (int(top_left_pt[0])+10, int(top_left_pt[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if 1271 in idx_to_pt:
            bottom_right_pt = idx_to_pt[1271]
            cv2.circle(left_disp, tuple(bottom_right_pt.astype(int)), 6, (255, 255, 255), 2)
            cv2.putText(left_disp, "1271", (int(bottom_right_pt[0])-50, int(bottom_right_pt[1])+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(middle_disp, tuple(bottom_right_pt.astype(int)), 6, (255, 255, 255), 2)
            cv2.putText(middle_disp, "1271", (int(bottom_right_pt[0])-50, int(bottom_right_pt[1])+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if remapping_quality == "grid_offset" and len(remapping_dict) > 0:
            sample_curr = list(remapping_dict.keys())[0]
            sample_prev = remapping_dict[sample_curr]
            
            col_c, row_c = self._idx_to_grid(sample_curr)
            col_p, row_p = self._idx_to_grid(sample_prev)
            dx = col_p - col_c
            dy = row_p - row_c
            
            col_1, row_1 = 0, 0
            new_col_1, new_row_1 = col_1 + dx, row_1 + dy
            theoretical_idx_1 = new_row_1 * self.grid_cols + new_col_1 + 1
            
            col_1271, row_1271 = self.grid_cols - 1, self.grid_rows - 1
            new_col_1271, new_row_1271 = col_1271 + dx, row_1271 + dy
            theoretical_idx_1271 = new_row_1271 * self.grid_cols + new_col_1271 + 1
            
            if 1 in idx_to_pt and 1271 in idx_to_pt:
                top_left_pt = idx_to_pt[1]
                cv2.circle(right_disp, tuple(top_left_pt.astype(int)), 6, (255, 255, 255), 2)
                cv2.putText(right_disp, f"~{theoretical_idx_1}", 
                        (int(top_left_pt[0])+10, int(top_left_pt[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                bottom_right_pt = idx_to_pt[1271]
                cv2.circle(right_disp, tuple(bottom_right_pt.astype(int)), 6, (255, 255, 255), 2)
                cv2.putText(right_disp, f"~{theoretical_idx_1271}", 
                        (int(bottom_right_pt[0])-80, int(bottom_right_pt[1])+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            info_text = f"Theoretical Range: {theoretical_idx_1} ~ {theoretical_idx_1271} (offset: dx={dx}, dy={dy})"
            cv2.putText(right_disp, info_text, 
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            print(f"   [Visualization] Right panel theoretical index range: {theoretical_idx_1} ~ {theoretical_idx_1271} (offset dx={dx}, dy={dy})")
            print(f"   [Visualization] Actual visible cyan points: {remapped_count}/{len(curr_set)} (only showing remapped points still within 1-1271 range)")
            
        else:
            if 1 in idx_to_pt:
                top_left_pt = idx_to_pt[1]
                cv2.circle(right_disp, tuple(top_left_pt.astype(int)), 6, (255, 255, 255), 2)
                cv2.putText(right_disp, "1", (int(top_left_pt[0])+10, int(top_left_pt[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if 1271 in idx_to_pt:
                bottom_right_pt = idx_to_pt[1271]
                cv2.circle(right_disp, tuple(bottom_right_pt.astype(int)), 6, (255, 255, 255), 2)
                cv2.putText(right_disp, "1271", (int(bottom_right_pt[0])-50, int(bottom_right_pt[1])+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.putText(canvas, f"Current (Original, {len(curr_set)} pts)", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(canvas, f"Previous ({len(prev_set)} pts)", 
                    (w + 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if remapping_quality == "grid_offset":
            right_title = f"Current (Remapped, {remapped_count}/{len(curr_set)} visible) - Grid Offset"
        else:
            right_title = f"Current (Remapped, {remapped_count} pts) - ICP Only"
        cv2.putText(canvas, right_title, 
                    (w*2 + 50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow(title, canvas)
        cv2.waitKey(1)

    def _visualize_matching_result(self, last_points_3d, transformed_points_3d, current_points_3d, match_details):
        from normalflow.registration import world_to_pixels
        
        img_width, img_height = 400, 400

        transformed_center_3d = np.mean(transformed_points_3d, axis=0)
        current_center_3d = np.mean(current_points_3d, axis=0)

        all_points = np.vstack([last_points_3d, transformed_points_3d, current_points_3d])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        margin = 0.002
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        virtual_height_map = np.zeros((img_height, img_width))
        ppmm = 0.0634
        
        img1 = self._draw_points_2d_with_world_to_pixels(
            last_points_3d, "Last Saved Points", 
            virtual_height_map, ppmm, (0, 255, 0)
        )
        
        img2 = self._draw_points_2d_with_world_to_pixels(
            transformed_points_3d, "Transformed Points", 
            virtual_height_map, ppmm, (255, 0, 0)
        )
        
        img3 = self._draw_points_2d_with_world_to_pixels(
            current_points_3d, "Current Points", 
            virtual_height_map, ppmm, (0, 0, 255)
        )
        
        img_compare = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        trans_pts_2d = world_to_pixels(transformed_points_3d, virtual_height_map, ppmm)
        for pt in trans_pts_2d:
            cv2.circle(img_compare, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)
        
        current_pts_2d = world_to_pixels(current_points_3d, virtual_height_map, ppmm)
        for pt in current_pts_2d:
            cv2.circle(img_compare, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

        current_center_2d = world_to_pixels(np.array([current_center_3d]), virtual_height_map, ppmm)[0]
        transformed_center_2d = world_to_pixels(np.array([transformed_center_3d]), virtual_height_map, ppmm)[0]
        
        cv2.circle(img_compare, (int(current_center_2d[0]), int(current_center_2d[1])), 8, (255, 255, 0), -1)
        cv2.circle(img_compare, (int(transformed_center_2d[0]), int(transformed_center_2d[1])), 8, (0, 255, 255), -1)
                
        cv2.putText(img_compare, "Red: Current | Blue: Transformed", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        top_row = cv2.hconcat([img1, img2])
        bottom_row = cv2.hconcat([img3, img_compare])
        final_img = cv2.vconcat([top_row, bottom_row])
        
        info_height = 100
        info_img = np.zeros((info_height, final_img.shape[1], 3), dtype=np.uint8)
        
        is_success = match_details.get('match_success', True)
        title_color = (0, 255, 0) if is_success else (0, 0, 255)
        title_text = "Contact Reconnection - MATCH SUCCESS" if is_success else "Contact Reconnection - MATCH FAILED"
        
        info_text = [
            title_text,
            f"Z Rotation: {match_details['z_angle_correction']:.2f}°",
            f"XY Translation: [{match_details['xy_translation'][0] * 1000:.2f}, {match_details['xy_translation'][1] * 1000:.2f}] mm",
            f"ICP Fitness: {match_details['icp_fitness']:.3f} | RMSE: {match_details['icp_rmse']:.4f}"
        ]
        
        for i, text in enumerate(info_text):
            color = title_color if i == 0 else (255, 255, 255)
            font_scale = 0.6 if i == 0 else 0.4
            cv2.putText(info_img, text, (10, 20 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        final_display = cv2.vconcat([info_img, final_img])
        
        window_title = "Contact Reconnection - SUCCESS" if is_success else "Contact Reconnection - FAILED"
        cv2.imshow(window_title, final_display)
        cv2.waitKey(3000)
        cv2.destroyWindow(window_title)
        
        result_emoji = "✅" if is_success else "❌"
        print(f"🖼️  Matching visualization displayed {result_emoji}")

    def _visualize_matched_indices_on_grid(self, flat_frame_image, all_pts_2d, all_indices, matched_pairs, title="Matched Indices"):
        h, w, _ = flat_frame_image.shape
        
        canvas = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        left_disp = canvas[:, :w]
        right_disp = canvas[:, w+20:]

        base_left = flat_frame_image.copy()
        base_right = flat_frame_image.copy()
        for pt in all_pts_2d:
            cv2.circle(base_left, tuple(pt.astype(int)), 3, (0, 180, 0), -1)
            cv2.circle(base_right, tuple(pt.astype(int)), 3, (0, 180, 0), -1)

        left_disp[:] = base_left
        right_disp[:] = base_right

        idx_to_pt2d = {idx: pt for idx, pt in zip(all_indices, all_pts_2d)}

        prev_matched_indices = {p for p, c in matched_pairs}
        curr_matched_indices = {c for p, c in matched_pairs}

        for idx in prev_matched_indices:
            if idx in idx_to_pt2d:
                pt = idx_to_pt2d[idx]
                cv2.circle(left_disp, tuple(pt.astype(int)), 4, (0, 0, 255), -1)

        for idx in curr_matched_indices:
            if idx in idx_to_pt2d:
                pt = idx_to_pt2d[idx]
                cv2.circle(right_disp, tuple(pt.astype(int)), 4, (0, 255, 255), -1)

        cv2.putText(canvas, f"Previous Matched ({len(prev_matched_indices)} pts)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(canvas, f"Current Matched ({len(curr_matched_indices)} pts)", (w + 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(title, canvas)
        cv2.waitKey(1)

    def _draw_points_2d_with_world_to_pixels(self, points_3d, title, height_map, ppmm, color):
        from normalflow.registration import world_to_pixels
        img_height, img_width = height_map.shape
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        pts_2d = world_to_pixels(points_3d, height_map, ppmm)
        
        for pt in pts_2d:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img_width and 0 <= y < img_height:
                cv2.circle(img, (x, y), 3, color, -1)
        
        cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"{len(points_3d)} points", (10, img_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return img

    def _build_geometric_transform_matrix(self, z_rotation_matrix, translation_3d, rotation_center, icp_transform):
        R_z = np.eye(4)
        R_z[:3, :3] = z_rotation_matrix
        
        T3 = np.eye(4)
        T3[:3, 3] = translation_3d
        
        geometric_transform = T3 @ R_z
        full_transform = icp_transform @ geometric_transform
        
        return full_transform
    
    def _evaluate_geometric_match_quality(self, alignment_result):
        """Evaluate geometric alignment match quality"""
        icp_result = alignment_result['icp_result']
        
        fitness_good = icp_result.fitness > 0.98
        
        z_angle_diff = abs(alignment_result['z_angle_diff'])
        z_rotation_reasonable = z_angle_diff < 90.0

        overlap_ratio = abs(alignment_result.get('contour_overlap_ratio', 0.0))
        
        is_match = (fitness_good) and z_rotation_reasonable and (overlap_ratio > 0.8)
        
        return is_match

    def _create_z_rotation_matrix(self, angle_deg):
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    
    def _apply_rotation(self, points, rotation_matrix):
        return (rotation_matrix @ points.T).T
    
    def _perform_icp(self, source_points, target_points):
        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()
        
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 
            max_correspondence_distance=0.001,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=100
            )
        )
        
        return result

    def _calculate_contour_overlap(self, points_a, points_b, current_H_tar, last_H_tar, ppmm=0.0634):
        from normalflow.registration import world_to_pixels
        from scipy.spatial import ConvexHull
        
        if len(points_a) < 3 or len(points_b) < 3:
            return 0.0
        
        pixels_a = world_to_pixels(points_a, current_H_tar, ppmm)
        pixels_b = world_to_pixels(points_b, current_H_tar, ppmm)
        
        h, w = current_H_tar.shape
        pixels_a = np.clip(pixels_a.astype(int), [0, 0], [w-1, h-1])
        pixels_b = np.clip(pixels_b.astype(int), [0, 0], [h-1, w-1])
        
        try:
            hull_a = ConvexHull(pixels_a)
            hull_b = ConvexHull(pixels_b)
            
            boundary_a = pixels_a[hull_a.vertices]
            boundary_b = pixels_b[hull_b.vertices]
            
            area_a = hull_a.volume
            area_b = hull_b.volume
            
            mask_a = np.zeros((h, w), dtype=np.uint8)
            mask_b = np.zeros((h, w), dtype=np.uint8)
            
            cv2.fillPoly(mask_a, [boundary_a], 255)
            cv2.fillPoly(mask_b, [boundary_b], 255)
            
            intersection = cv2.bitwise_and(mask_a, mask_b)
            intersection_area = np.sum(intersection > 0)
            if intersection_area > min(area_a, area_b):
                intersection_area = min(area_a, area_b)
            
            max_area = max(area_a, area_b)
            overlap_ratio = intersection_area / max_area if max_area > 0 else 0.0

            print(f"     Contour area: A={area_a:.1f}px², B={area_b:.1f}px², overlap={intersection_area:.1f}px², ratio={overlap_ratio:.3f}")
            
            return overlap_ratio
            
        except Exception as e:
            print(f"     Contour calculation failed: {e}")
            return 0.0

    def clear_last_contact(self):
        """Clear last contact information"""
        self.last_contact_info = None
        print("🗑️  Last contact information cleared")

# Global instance
contact_reconnection_matcher = ContactRegionMatcher()

# ============= Public interface functions =============
def save_contact_for_reconnection(contact_info, z_angle, geometric_center, H_tar):
    global contact_reconnection_matcher
    return contact_reconnection_matcher.save_contact_with_info(
        contact_info, z_angle, geometric_center, H_tar
    )

def match_contact_for_reconnection(current_contact_info, z_angle, geometric_center, H_tar, flat_frame_vis_data=None):
    global contact_reconnection_matcher
    return contact_reconnection_matcher.match_contact_with_info(
        current_contact_info, z_angle, geometric_center, H_tar, flat_frame_vis_data
    )

def clear_reconnection_data():
    global contact_reconnection_matcher
    contact_reconnection_matcher.clear_last_contact()

def reset_reconnection_matcher(icp_threshold=0.001, rotation_threshold=5.0, translation_threshold=0.002):
    global contact_reconnection_matcher
    contact_reconnection_matcher = ContactRegionMatcher(icp_threshold, rotation_threshold, translation_threshold)
    print("🔄 Reconnection matcher reset")

def has_saved_contact():
    global contact_reconnection_matcher
    return contact_reconnection_matcher.last_contact_info is not None