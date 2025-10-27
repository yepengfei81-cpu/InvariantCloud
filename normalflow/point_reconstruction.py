import numpy as np
import open3d as o3d
import cv2
from collections import deque

class MultiFramePointCloudFusion:
    def __init__(self, max_frames=200, voxel_size=0.0005, accumulate_all=True):
        self.max_frames = max_frames
        self.voxel_size = voxel_size
        self.accumulate_all = accumulate_all

        self.frame_poses = deque(maxlen=max_frames)
        self.frame_ids = deque(maxlen=max_frames)

        # Global accumulated point cloud
        self.world_points = {}  # key: (session_id, index) -> np.array([x,y,z])
        self.current_session_id = 0
        self.global_pcd = o3d.geometry.PointCloud()   
        self.session_base_translation = {0: np.zeros(3)}
        self.translation_history = deque(maxlen=5)

        # Point repository
        self.point_repo = {}     
        self.world_origin_pose = None

        # Visualization
        self.vis = None
        self.coordinate_frame = None
        self.trajectory_points = []
        self.color_mode = 'depth'  # 'world_z' or 'depth'

        self.frame_count = 0
        self.debug_live_only = False         

    def start_new_session(self):
        self.current_session_id += 1
        self.world_origin_pose = None
        print(f"🟢 New session started: session_id={self.current_session_id}")

    def _update_repo_and_world_points(self, pose_matrix, indices, local_pts, geometric_center=None):
        indices = np.asarray(indices).reshape(-1)
        local_pts = np.asarray(local_pts).reshape(-1, 3)
        if len(indices) == 0 or len(local_pts) == 0:
            return

        min_len = min(len(indices), len(local_pts))
        indices = indices[:min_len]
        local_pts = local_pts[:min_len]

        current_translation_global = pose_matrix[:3, 3]
        geometric_center = np.asarray(geometric_center).reshape(3)
        
        # Record global cumulative baseline at start of each contact
        if not hasattr(self, 'session_initial_centers'):
            self.session_initial_centers = {}
        if not hasattr(self, 'session_base_translations_global'):
            self.session_base_translations_global = {}            
        
        if self.current_session_id not in self.session_initial_centers:
            self.session_initial_centers[self.current_session_id] = geometric_center.copy()
            self.session_base_translations_global[self.current_session_id] = current_translation_global.copy()
            
        initial_center = self.session_initial_centers[self.current_session_id]
        cumulative_displacement = geometric_center - initial_center
        base_translation_global = self.session_base_translations_global[self.current_session_id]

        # Update repository: lock T and XY; update Z to deeper value
        for idx, p in zip(indices, local_pts):
            key = (self.current_session_id, int(idx))
            if key not in self.point_repo:
                self.point_repo[key] = {
                    'T': pose_matrix[:3, :3].copy(),
                    'xy': p[:2].copy(),
                    'depth': float(p[2]),
                    'first_geometric_center': geometric_center.copy(),
                    'first_cumulative_displacement': cumulative_displacement.copy(),
                    'base_translation_global': base_translation_global.copy()
                }
            else:
                if float(p[2]) < self.point_repo[key]['depth']:
                    self.point_repo[key]['depth'] = float(p[2])
                    self.point_repo[key]['T'] = pose_matrix[:3, :3].copy()   

        # Rebuild world_points from repository
        self.world_points.clear()
        
        print(f"\nFrame {self.frame_count} - Coordinate transformation:")
        print(f"   Input points: {len(indices)}")
        print(f"   Total repo points: {len(self.point_repo)}")

        for key, rec in self.point_repo.items():
            x, y = rec['xy']
            z = rec['depth']
            P_local = np.array([x, y, z], dtype=np.float64)
            P_relative = P_local - rec['first_geometric_center']
            P_rotated = rec['T'] @ P_relative
            displacement_rotated = rec['T'] @ rec['first_cumulative_displacement']
            P_in_session = P_rotated + displacement_rotated
            P_world = P_in_session + rec['base_translation_global']
            self.world_points[key] = P_world

        self._rebuild_global_pcd_from_world_points()

    def _rebuild_global_pcd_from_world_points(self):
        if len(self.world_points) == 0:
            self.global_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.global_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            return

        ordered_keys = list(self.world_points.keys())
        pts = np.vstack([self.world_points[k] for k in ordered_keys])
        self.global_pcd.points = o3d.utility.Vector3dVector(pts)

        # Choose scalar field for coloring
        if self.color_mode == 'world_z':
            scalars = pts[:, 2]
        else:
            scalars = np.array([self.point_repo[k]['depth'] for k in ordered_keys], dtype=np.float64)

        s_min, s_max = float(scalars.min()), float(scalars.max())
        if s_max > s_min:
            ns = (scalars - s_min) / (s_max - s_min)
            colors = np.zeros((len(pts), 3), dtype=np.float64)
            colors[:, 0] = ns
            colors[:, 1] = 1 - np.abs(ns - 0.5) * 2
            colors[:, 2] = 1 - ns
        else:
            colors = np.tile(self._get_frame_color(self.frame_count), (len(pts), 1))

        self.global_pcd.colors = o3d.utility.Vector3dVector(colors)

    def _get_frame_color(self, frame_id):
        # Generate different colors using HSV color space
        hue = (frame_id * 137.508) % 360  # Golden angle distribution
        saturation = 0.8
        value = 0.9

        import colorsys
        rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
        return np.array(rgb)
    
    def add_frame(self, pose_matrix, frame_id=None, contact_points_3d=None, geometric_center=None):
        if frame_id is None:
            frame_id = self.frame_count

        print(f"\n🔵 Adding frame {frame_id}...")

        points_3d_array = np.asarray(contact_points_3d['points_3d'])
        indices = np.asarray(contact_points_3d['indices']).reshape(-1)
        
        translation = pose_matrix[:3, 3]

        if getattr(self, 'debug_live_only', False):
            local = points_3d_array.astype(np.float64)
            if local.size == 0:
                self.global_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                self.global_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            else:
                ones = np.ones((local.shape[0], 1), dtype=np.float64)
                Pw = (pose_matrix @ np.hstack([local, ones]).T).T[:, :3]
                self.global_pcd.points = o3d.utility.Vector3dVector(Pw)

                if self.color_mode == 'world_z':
                    scalars = Pw[:, 2]
                else:
                    scalars = local[:, 2]
                s_min, s_max = float(np.min(scalars)), float(np.max(scalars))
                if s_max > s_min:
                    ns = (scalars - s_min) / (s_max - s_min)
                    colors = np.zeros((len(Pw), 3), dtype=np.float64)
                    colors[:, 0] = ns
                    colors[:, 1] = 1 - np.abs(ns - 0.5) * 2
                    colors[:, 2] = 1 - ns
                else:
                    colors = np.tile(self._get_frame_color(self.frame_count), (len(Pw), 1))
                self.global_pcd.colors = o3d.utility.Vector3dVector(colors)

            self.frame_poses.append(pose_matrix.copy())
            self.frame_ids.append(frame_id)
            self.trajectory_points.append(translation.copy())
            # self.update_visualization_improved()
            self.frame_count += 1
            return True
        
        # Update repository and rebuild global point cloud
        self._update_repo_and_world_points(pose_matrix, indices, points_3d_array, geometric_center)

        self.frame_poses.append(pose_matrix.copy())
        self.frame_ids.append(frame_id)
        self.trajectory_points.append(translation.copy())

        # self.update_visualization_improved()

        self.frame_count += 1
        return True

    def _matrix_to_euler(self, rotation_matrix):
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rotation_matrix)
        return r.as_euler('xyz', degrees=True)

    def initialize_visualization(self):
        if self.vis is None:
            self.global_pcd_added = False
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("Multi-frame Point Cloud Fusion - 3D Surface Reconstruction", 1200, 800)

            self.vis.add_geometry(self.global_pcd)
            try:
                ro = self.vis.get_render_option()
                ro.line_width = 5.0
            except Exception:
                pass
            print("🖥️ Visualization window initialized")

    def update_visualization_improved(self):
        if self.vis is None:
            self.initialize_visualization()

        try:          
            if hasattr(self, 'global_pcd_added') and self.global_pcd_added:
                self.vis.remove_geometry(self.global_pcd, reset_bounding_box=False)

            self.vis.add_geometry(self.global_pcd, reset_bounding_box=False)
            self.global_pcd_added = True

            # Add trajectory line
            if len(self.trajectory_points) > 1:
                if hasattr(self, 'trajectory_line'):
                    self.vis.remove_geometry(self.trajectory_line, reset_bounding_box=False)

                self.trajectory_line = o3d.geometry.LineSet()
                self.trajectory_line.points = o3d.utility.Vector3dVector(self.trajectory_points)

                lines = [[i, i+1] for i in range(len(self.trajectory_points)-1)]
                self.trajectory_line.lines = o3d.utility.Vector2iVector(lines)

                colors = [[1, 0, 0] for _ in range(len(lines))]
                self.trajectory_line.colors = o3d.utility.Vector3dVector(colors)

                self.vis.add_geometry(self.trajectory_line, reset_bounding_box=False)

            # Current camera position
            if len(self.frame_poses) > 0:
                if hasattr(self, 'current_camera'):
                    self.vis.remove_geometry(self.current_camera, reset_bounding_box=False)

                self.current_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.008)
                self.current_camera.transform(self.frame_poses[-1])
                self.vis.add_geometry(self.current_camera, reset_bounding_box=False)

            self.vis.poll_events()
            self.vis.update_renderer()

        except Exception as e:
            print(f"❌ Visualization update error: {e}")

    def save_reconstruction(self, output_path="contact_surface_reconstruction.ply"):
        if len(self.global_pcd.points) > 0:
            success = o3d.io.write_point_cloud(output_path, self.global_pcd)
            if success:
                print(f"✅ 3D reconstruction saved: {output_path}")
                print(f"   Total points: {len(self.global_pcd.points)}")
                return True
            else:
                print(f"❌ Save failed: {output_path}")
        else:
            print("❌ No point cloud data to save")
        return False

    def generate_mesh(self, method='poisson'):
        if len(self.global_pcd.points) < 500:
            print("Insufficient points for mesh generation")
            return None

        try:
            pcd_copy = o3d.geometry.PointCloud(self.global_pcd)
            
            points = np.asarray(pcd_copy.points)
            pcd_copy.points = o3d.utility.Vector3dVector(points)

            # Remove statistical outliers
            pcd_copy, outlier_mask = pcd_copy.remove_statistical_outlier(
                nb_neighbors=30, std_ratio=2.5
            )
            
            # Adaptive normal estimation
            distances = pcd_copy.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            normal_radius = avg_dist * 10
            
            pcd_copy.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius,
                    max_nn=60
                )
            )
            normals = np.asarray(pcd_copy.normals)
            pcd_copy.normals = o3d.utility.Vector3dVector(normals)      

            pcd_copy.orient_normals_consistent_tangent_plane(k=40)
            
            if method == 'ball_pivoting':
                # Adaptive radius adjustment based on density
                if std_dist / avg_dist > 0.3:
                    base_radius = avg_dist
                    radii = [
                        base_radius * 0.5,
                        base_radius * 0.8,
                        base_radius * 1.0,
                        base_radius * 1.5,
                        base_radius * 2.0,
                        base_radius * 3.0,
                        base_radius * 4.0,
                        base_radius * 6.0,
                        base_radius * 8.0,
                        base_radius * 12.0,
                        base_radius * 15.0,
                        base_radius * 25.0
                    ]
                else:
                    base_radius = avg_dist
                    radii = [
                        base_radius * 1.5,
                        base_radius * 3.0,
                        base_radius * 6.0,
                        base_radius * 12.0,
                        base_radius * 20.0
                    ]
                
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd_copy, o3d.utility.DoubleVector(radii)
                )
                
                if len(mesh.triangles) > 0:
                    mesh_vertices = np.asarray(mesh.vertices)
                    points_original = np.asarray(pcd_copy.points)
                    
                    vertex_count = len(mesh_vertices)
                    point_count = len(points_original)
                    
                    print(f"Mesh generated: {vertex_count} vertices, {len(mesh.triangles)} triangles")
                    
                    if vertex_count < point_count * 0.3:
                        print(f"⚠️ Low vertex count ({vertex_count}/{point_count}), possible connection issues")
                    else:
                        print(f"✅ Mesh quality good")
                else:
                    smaller_radii = [avg_dist * i for i in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.5, 5.0, 8.0]]
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pcd_copy, o3d.utility.DoubleVector(smaller_radii)
                    )

            print("Cleaning mesh...")
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            if method in ['ball_pivoting', 'alpha_shape']:
                mesh = mesh.filter_smooth_simple(number_of_iterations=2)

            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            mesh.compute_vertex_normals()

            print(f"✅ Mesh generation successful: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            return mesh

        except Exception as e:
            print(f"❌ Mesh generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# Global fusion system instance
multiframe_fusion = None

def initialize_multiframe_fusion(max_frames=200, voxel_size=0.0005, accumulate_all=True):
    global multiframe_fusion
    multiframe_fusion = MultiFramePointCloudFusion(
        max_frames=max_frames, 
        voxel_size=voxel_size,
        accumulate_all=accumulate_all
    )
    mode = "Accumulate mode" if accumulate_all else "Sliding window mode"
    print(f"🚀 Multi-frame fusion system initialized ({mode})")

def start_new_fusion_session():
    global multiframe_fusion
    if multiframe_fusion is not None:
        multiframe_fusion.start_new_session()
        
def add_frame_to_fusion(T_ours, frame_id=None, contact_points_3d=None, geometric_center=None):
    global multiframe_fusion
    if multiframe_fusion is not None:
        return multiframe_fusion.add_frame(T_ours, frame_id, contact_points_3d, geometric_center)
    else:
        print("❌ Fusion system not initialized, call initialize_multiframe_fusion() first")
    return False

def generate_fusion_mesh(method='poisson'):
    global multiframe_fusion
    if multiframe_fusion is not None:
        return multiframe_fusion.generate_mesh(method)
    return None