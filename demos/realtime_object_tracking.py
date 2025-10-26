import argparse
import os
import sys
gs_sdk_path = "/home/ypf/gs_sdk"
sys.path.insert(0, gs_sdk_path)

import cv2
import numpy as np
import yaml

from gs_sdk.gs_device import Camera, FastCamera
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import normalflow, InsufficientOverlapError
from normalflow.utils import erode_contact_mask, gxy2normal, transform2pose
from normalflow.viz_utils import annotate_coordinate_system
from marker_detector import MarkerDetector
from marker_depth_refine import MarkerDepthRefiner
"""
This script demonstrates real-time using normalflow to track objects in contact.
The implementation is using the long-horizon tracking algorithm discussed in our paper.

Prerequisite:
    - Calibrate your GelSight sensor if you are not using the GelSight Mini sensor.
    - Connect the sensor to your computer.
Instruction:
    - Run this script, wait two seconds for background image collection.
    - Press whatever object to the sensor for tracking. You may switch objects in between.
    - Press any key to quit the streaming session.
Important Note:
    - If you are using the GelSight Mini, setting args.streamer to 'opencv' will result in a lower frame rate 
      (~10 Hz instead of the expected 25 Hz). Setting args.streamer to 'ffmpeg' resolves this frame rate issue;
      however, on some systems, it may introduce significant delays and duplicate frames.
    - If you are not using GelSight Mini, please calibrate your sensor using the gs_sdk package.
    - The whole calibration process only requires a metal ball with a known diameter and less than half hour.
    - Provide the trained calibration model and the sensor configuration file as arguments when running.

Usage:
    python realtime_object_tracking.py [--calib_model_path CALIB_MODEL_PATH] [--config_path CONFIG_PATH] [--device {cpu, cuda}]

Arguments:
    --calib_model_path: (Optional) The directory where the calibration model are stored.
            The calibration model can be easily trained using the gs_sdk package.
            The default is our trained calibration model for GelSight Mini.
    --config_path: (Optional) The path of the configuration file for the GelSight sensor.
            The configuration file specifies the specifications of the sensor.
            The default is the configuration file for GelSight Mini.
    --streamer: (Optional) The sensor streamer.
            Can be either 'opencv' for 'gs_device.Camera' or 'ffmpeg' for 'gs_device.FastCamera'.
    --device: (Optional) The device to run the neural network model that predicts the normal map.
            Can be either 'cpu' or 'cuda'. The default is 'cpu'.

Press any key to quit the streaming session.
"""


calib_model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")

def visualize_gradient(gradient, channel_name):
    vis = np.uint8(127 + 127 * np.clip(gradient, -1, 1))
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

def draw_sensor_coord(image):
    h, w = image.shape[:2]
    cv2.arrowedLine(image, (w//2, h//2), (w//2+50, h//2), (0,0,255), 2)
    cv2.arrowedLine(image, (w//2, h//2), (w//2, h//2+50), (0,255,0), 2)
    cv2.putText(image, "X", (w//2+55, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(image, "Y", (w//2, h//2+55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return image

def realtime_object_tracking():
    # Argument Parser
    parser = argparse.ArgumentParser(
        description="Real-time tracking the object using tactile sensors."
    )
    parser.add_argument(
        "-b",
        "--calib_model_path",
        type=str,
        help="place where the calibration data and model is stored",
        default=calib_model_path,
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of the configuration file for the GelSight sensor",
        default=config_path,
    )
    parser.add_argument(
        "-s",
        "--streamer",
        type=str,
        choices=["opencv", "ffmpeg"],
        help="The sensor streamer. 'opencv' for 'gs_device.Camera' and 'ffmpeg' for 'gs_device.FastCamera'.",
        default="opencv",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="the device to run the neural network model that predicts the normal map",
        default="cpu",
    )
    args = parser.parse_args()

    # Read the configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        device_name = config["device_name"]
        ppmm = config["ppmm"]
        imgh = config["imgh"]
        imgw = config["imgw"]
        raw_imgh = config["raw_imgh"]
        raw_imgw = config["raw_imgw"]
        framerate = config["framerate"]

    # Connect to the sensor and the reconstructor
    if args.streamer == "opencv":
        device = Camera(device_name, imgh, imgw)
    elif args.streamer == "ffmpeg":
        device = FastCamera(device_name, imgh, imgw, raw_imgh, raw_imgw, framerate)
    device.connect()
    recon = Reconstructor(args.calib_model_path, device="cpu")

    # Collect background images
    print("Collecting 10 background images, please wait ...")
    bg_images = []
    for _ in range(10):
        image = device.get_image()
        bg_images.append(image)
    bg_image = np.mean(bg_images, axis=0).astype(np.uint8)
    recon.load_bg(bg_image)
    marker_refiner = MarkerDepthRefiner(bg_image)
    # add new code to record init mark image
    # marker_vis = MarkerDetector(bg_image)
    print("Done with background collection.")

    # Real-time object tracking
    print("\nStart object tracking, Press any key to quit.\n")
    is_running = True
    print("Waiting for flat surface ...")
    image_flat = None
    while True:
        img = device.get_image()
        _, _, C_tmp = recon.get_surface_info(img, ppmm)
        if np.sum(C_tmp == 0) > 0.95 * C_tmp.size:
            image_flat = img.copy()
            print("Collect flat surface success")
            break      

    tracking_mode = "normal"   
    cloud_reset = False
    while is_running:
        image = device.get_image()
        G, H, C = recon.get_surface_info(image, ppmm)
        C = erode_contact_mask(C)
        contact_area = np.sum(C)
        if contact_area < 500:
            cv2.imshow("frame", image)
            key = cv2.waitKey(1)
            if key != -1:
                is_running = False
            continue
        else:
            # Tracking a new object, wait 2 frames for the contact to stabilize
            for _ in range(2):
                image = device.get_image()
                cv2.imshow("frame", image)
                key = cv2.waitKey(1)

            # Get the surface information of the reference frame (key frame)
            image_start = device.get_image()
            G_start, H_start, C_start = recon.get_surface_info(image_start, ppmm)
            # add code to init marker refiner
            marker_refiner.update_depth0(H_start)
            C_start = erode_contact_mask(C_start)
            N_start = gxy2normal(G_start)
            # For display purpose, get the largest contour and its center
            contours_start, _ = cv2.findContours(
                (C_start * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            M_start = cv2.moments(max(contours_start, key=cv2.contourArea))
            cx_start, cy_start = int(M_start["m10"] / M_start["m00"]), int(
                M_start["m01"] / M_start["m00"]
            )

            # Start tracking this object relative to the reference frame (key frame)
            G_ref = G_start.copy()
            C_ref = C_start.copy()
            H_ref = H_start.copy()
            N_ref = N_start.copy()
            G_prev = None
            C_prev = None
            H_prev = None
            N_prev = None
            prev_T_ref = np.eye(4)
            start_T_ref = np.eye(4)
            is_tracking = True   
            while is_tracking:
                if key != -1:
                    # 检查键盘输入以切换模式
                    if key == ord('a') or key == ord('A'):
                        tracking_mode = "save_once"
                        print("🔄 切换到保存模式（一次性）")
                    elif key == ord('b') or key == ord('B'):
                        tracking_mode = "match_once"
                        print("🔄 切换到匹配模式（一次性）")                
                # Get the surface information of the current frame
                image_curr = device.get_image()
                G_curr, H_curr, C_curr = recon.get_surface_info(image_curr, ppmm)
                C_curr = erode_contact_mask(C_curr)
                # add new code to refine H_curr
                H_refined = marker_refiner.refine(image_curr, H_curr)
                # add code to evaluate difference between refined and original
                if np.sum(C_curr) < 500:
                    is_tracking = False
                    cloud_reset = True
                    break
                N_curr = gxy2normal(G_curr)

                save_contact = (tracking_mode == "save_once")
                match_contact = (tracking_mode == "match_once")
                # Use NormalFlow to estimate the transformation
                try:
                    curr_T_ref, T_ours, geometric_center = normalflow(
                        N_ref,
                        C_ref,
                        H_ref,
                        N_curr,
                        C_curr,
                        H_curr,
                        image_flat,
                        image_start,
                        image_curr,
                        prev_T_ref,
                        ppmm,
                        save_contact=save_contact,
                        match_contact=match_contact,     
                        has_reset=cloud_reset,
                        useful_info=True                   
                    )
                    if tracking_mode == "save_once" or tracking_mode == "match_once":
                        tracking_mode = "normal"  # reset to normal mode
                    else:
                        tracking_mode = "normal"
                        
                    is_reset = False
                except InsufficientOverlapError:
                    is_reset = True
                # Reset reference frame (set a new keyframe) if needed.
                if N_prev is not None:
                    curr_T_prev, T_ours, geometric_center = normalflow(
                        N_prev,
                        C_prev,
                        H_prev,
                        N_curr,
                        C_curr,
                        H_curr,
                        image_flat,
                        image_start,
                        image_curr,                        
                        np.eye(4),
                        ppmm,
                        has_reset=cloud_reset,
                        useful_info=False 
                    )
                    cloud_reset = False
                    if not is_reset:
                        T_error = np.linalg.inv(curr_T_ref) @ curr_T_prev @ prev_T_ref
                        pose_error = transform2pose(T_error)
                        rot_error = np.linalg.norm(pose_error[3:])
                        trans_error = np.linalg.norm(pose_error[:3])
                        is_reset = rot_error > 3.0 or trans_error > 1.0
                    if is_reset:
                        G_ref = G_prev.copy()
                        C_ref = C_prev.copy()
                        H_ref = H_prev.copy()
                        N_ref = N_prev.copy()
                        start_T_ref = start_T_ref @ np.linalg.inv(prev_T_ref)
                        curr_T_ref = curr_T_prev.copy()
                # Update states
                G_prev = G_curr.copy()
                C_prev = C_curr.copy()
                H_prev = H_curr.copy()
                N_prev = N_curr.copy()
                prev_T_ref = curr_T_ref.copy()

                R_mat_ours = T_ours[:3, :3]
                R_mat_ours[:, 1] *= -1
                axes = np.eye(3)
                axes_rot = R_mat_ours @ axes.T  # shape (3,3)

                # Display the object tracking result
                image_l = image_start.copy()
                cv2.putText(
                    image_l,
                    "Initial Frame",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                center_start = np.array([cx_start, cy_start]).astype(np.int32)
                unit_vectors_start = np.eye(3)[:, :2]
                annotate_coordinate_system(image_l, center_start, unit_vectors_start)
                # Annotate the transformation on the target frame
                image_ours = image_curr.copy()
                cv2.putText(
                    image_ours,
                    "Our Method (InvariantCloud)",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )     
                from normalflow.registration import world_to_pixels
                center_pixel = world_to_pixels(
                    np.array([[geometric_center[0], geometric_center[1], 0]]),  # 如果只有xy
                    H_curr,  # 当前帧的单应性矩阵
                    ppmm
                )[0]
                center = tuple(np.round(center_pixel).astype(int))
                cv2.circle(image_ours, center, 5, (255, 255, 255), -1)      
                axis_length = 50
                axes_colors = [(0,0,255), (0,255,0), (255,0,0)]             
                for i in range(3):
                    dx = axes_rot[0, i] * axis_length
                    dy = -axes_rot[1, i] * axis_length  # 注意Y轴像素方向向下
                    end_point = (int(center[0] + dx), int(center[1] + dy))
                    cv2.arrowedLine(image_ours, center, end_point, axes_colors[i], 2, tipLength=0.2)                                      
               
                image_r = image_curr.copy()
                cv2.putText(
                    image_r,
                    "Baseline Method (NormalFlow)",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                center_3d_start = (
                    np.array(
                        [(cx_start - imgw / 2 + 0.5), (cy_start - imgh / 2 + 0.5), 0]
                    )
                    * ppmm
                    / 1000.0
                )
                unit_vectors_3d_start = np.eye(3) * ppmm / 1000.0
                curr_T_start = curr_T_ref @ np.linalg.inv(start_T_ref)
                R_mat = curr_T_start[:3, :3]
                from scipy.spatial.transform import Rotation as R
                euler_angles = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
                z_angle_deg = euler_angles[2] 
                remapped_center_3d_start = (
                    np.dot(curr_T_start[:3, :3], center_3d_start) + curr_T_start[:3, 3]
                )
                remapped_cx_start = (
                    remapped_center_3d_start[0] * 1000 / ppmm + imgw / 2 - 0.5
                )
                remapped_cy_start = (
                    remapped_center_3d_start[1] * 1000 / ppmm + imgh / 2 - 0.5
                )
                remapped_center_start = np.array(
                    [remapped_cx_start, remapped_cy_start]
                ).astype(np.int32)
                remapped_unit_vectors_start = (
                    np.dot(curr_T_start[:3, :3], unit_vectors_3d_start.T).T
                    * 1000
                    / ppmm
                )[:, :2]
                annotate_coordinate_system(
                    image_r, remapped_center_start, remapped_unit_vectors_start
                )

                # Display
                cv2.imshow("frame", cv2.hconcat([image_l, image_r, image_ours]))
                key = cv2.waitKey(1)
                # if key != -1:
                #     is_tracking = False
                #     is_running = False                                 

    device.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_object_tracking()
