# marker_detector.py
import cv2
import numpy as np

class MarkerDetector:
    def __init__(self, bg_img, expect_num=63):
        """
        bg_img   : 无接触背景图 (BGR)
        expect_num: 期望检测到的标记点数量
        """
        self.expect_num = expect_num
        self.detector = self._build_blob_detector()
        keypoints = self.detector.detect(bg_img)
        if len(keypoints) != expect_num:
            raise RuntimeError(f"期望 {expect_num} 个标记点，实际 {len(keypoints)} 个")
        # 按 x 排序，固定顺序
        self.ref_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        self.prev_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def _build_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 15
        params.maxArea = 200
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        return cv2.SimpleBlobDetector_create(params)

    def draw(self, img_bgr):
        """在输入图上画点并返回标注图"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        next_pts, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray,
                                                   self.ref_pts, None,
                                                   **self.lk_params)
        good = st.squeeze() == 1
        next_pts = next_pts[good]
        self.prev_gray = gray
        self.ref_pts = next_pts  # 更新参考点

        out = img_bgr.copy()
        for idx, (x, y) in enumerate(next_pts, 1):
            cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(out, str(idx), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return out