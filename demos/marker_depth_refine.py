"""
marker_depth_refinement.py
离线标定一次 -> 在线光流跟踪 -> TPS 插值精修深度
"""
import cv2
import numpy as np
from scipy.interpolate import Rbf

class MarkerDepthRefiner:
    def __init__(self, ref_img, detector_params=None):
        """
        ref_img : uint8 BGR 背景图（无接触）
        detector_params : cv2.SimpleBlobDetector params（可留 None）
        """
        if detector_params is None:
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = 50
            params.maxThreshold = 255
            params.filterByArea = True
            params.minArea = 15
            params.maxArea = 200
            params.filterByCircularity = True
            params.minCircularity = 0.7
            detector_params = params
        self.detector = cv2.SimpleBlobDetector_create(detector_params)
        # 标定：提取 63 个标记点像素坐标
        keypoints = self.detector.detect(ref_img)
        # if len(keypoints) != 63:
        #     raise RuntimeError(f"Expect 63 markers, got {len(keypoints)}")
        # 按 x 排序，使顺序固定
        self.ref_uv = np.array([kp.pt for kp in keypoints], dtype=np.float32)  # (63,2)
        self.num = len(self.ref_uv)
        # 初始深度将在第一次调用 update_depth0 时填充
        self.ref_z = None
        # 光流用
        self.lk_param = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    def update_depth0(self, depth0):
        """把第一帧网络输出的深度图采样到标记点上"""
        h, w = depth0.shape
        u, v = self.ref_uv[:, 0], self.ref_uv[:, 1]
        u = np.clip(u.astype(int), 0, w - 1)
        v = np.clip(v.astype(int), 0, h - 1)
        self.ref_z = depth0[v, u]

    def refine(self, curr_bgr, coarse_depth):
        """
        输入：
            curr_bgr     : 当前帧 BGR
            coarse_depth : 网络输出的深度图 (h,w)
        返回：
            refined_depth: 精修后的深度图 (h,w)
        """
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        next_uv, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                  curr_gray,
                                                  self.ref_uv,
                                                  None,
                                                  **self.lk_param)
        # 只保留成功跟踪的点
        good = st.squeeze() == 1
        uv0 = self.ref_uv[good]
        uv1 = next_uv[good]
        z0 = self.ref_z[good]

        # 假设局部线性：dz = kx*dx + ky*dy，用最小二乘估计 kx,ky
        dx = uv1[:, 0] - uv0[:, 0]
        dy = uv1[:, 1] - uv0[:, 1]
        A = np.stack([dx, dy], axis=1)
        k, *_ = np.linalg.lstsq(A, z0 * 0.5, rcond=None)  # 比例系数经验值 0.01
        dz = k[0] * dx + k[1] * dy
        z1 = z0 + dz

        # 用 TPS 把稀疏修正量插值到整幅图
        h, w = coarse_depth.shape
        yy, xx = np.mgrid[0:h, 0:w]
        rbf = Rbf(uv1[:, 0], uv1[:, 1], z1, function='thin_plate')
        corr = rbf(xx, yy)  # (h,w)
        refined = coarse_depth + corr
        # 平滑一下
        refined = cv2.bilateralFilter(refined.astype(np.float32), 5, 1.0, 1.0)
        # 更新缓存
        self.prev_gray = curr_gray
        return refined