#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *

TRIANGULATION_PARAMS = TriangulationParameters(
    max_reprojection_error=1.0,
    min_triangulation_angle_deg=3.0,
    min_depth=0.1)


def _init_on_two_frames(frame_corners_0, frame_corners_1, intrinsic_mat) \
        -> Tuple[Pose, np.ndarray, np.ndarray]:
    correspondences = build_correspondences(frame_corners_0, frame_corners_1)
    pts_0 = correspondences.points_1
    pts_1 = correspondences.points_2
    # Check if the E matrix will be well-defined based on homography matrix
    H, h_mask = cv2.findHomography(pts_0, pts_1, method=cv2.RANSAC, ransacReprojThreshold=1.0, confidence=0.999)
    if np.nonzero(h_mask)[0].size / len(pts_0) > 0.9:
        return None, None, None
    E, e_mask = cv2.findEssentialMat(pts_0, pts_1, cameraMatrix=intrinsic_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    correspondences = remove_correspondences_with_ids(correspondences, np.where(e_mask == 0)[0])
    R1, R2, t_d = cv2.decomposeEssentialMat(E)
    best_pose, best_pts, best_ids = None, None, None
    for R, t in [(R1, t_d), (R1, -t_d), (R2, t_d), (R2, -t_d)]:
        pose_1 = Pose(r_mat=R, t_vec=t)
        view_0 = eye3x4()
        view_1 = pose_to_view_mat3x4(pose_1)
        pts, ids = triangulate_correspondences(
            correspondences, view_0, view_1, intrinsic_mat, TRIANGULATION_PARAMS)
        if best_pose is None or len(ids) > len(best_ids):
            best_pose, best_pts, best_ids = pose_1, pts, ids
    return best_pose, best_pts, best_ids


def _find_best_pair(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) -> int:
    best_index, largest_size = None, None
    for frame_index in range(1, len(corner_storage)):
        _, _, ids = _init_on_two_frames(corner_storage[0], corner_storage[frame_index], intrinsic_mat)
        if ids is None:
            continue
        if (ids is not None) and (best_index is None or len(ids) > largest_size):
            best_index = frame_index
            largest_size = len(ids)
    print('The best paired frame is {}. The initial cloud size is {}'.format(best_index, largest_size))
    return best_index


def _track_camera(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    paired_frame = _find_best_pair(corner_storage, intrinsic_mat)
    # Init the camera position based on the first and paired frames
    pose, pts, ids = _init_on_two_frames(corner_storage[0], corner_storage[paired_frame], intrinsic_mat)
    view_mats = [eye3x4()]
    builder = PointCloudBuilder()
    builder.add_points(ids, pts)

    # Process the rest of the frames
    for frame_index in range(1, len(corner_storage)):
        print('Processing frame {}/{}'.format(frame_index, len(corner_storage)))
        if frame_index == paired_frame:
            view_mats.append(pose_to_view_mat3x4(pose))
            print('paired frame')
        else:
            frame_corners = corner_storage[frame_index]
            ids = []
            objectPoints = []
            imagePoints = []
            for id, point in zip(frame_corners.ids, frame_corners.points):
                indices_x, _ = np.nonzero(builder.ids == id)
                if len(indices_x) == 0:
                    continue
                paired_frame = indices_x[0]
                ids.append(id)
                objectPoints.append(builder.points[paired_frame])
                imagePoints.append(point)
            objectPoints = np.array(objectPoints)
            imagePoints = np.array(imagePoints)
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints,
                imagePoints,
                cameraMatrix=intrinsic_mat,
                distCoeffs=np.array([]),
                flags=cv2.SOLVEPNP_EPNP)
            view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            view_mats.append(view_mat)
            print('inliers count={} new triangulated points={} cloud size={}'.format(len(inliers), 0, len(builder.ids)))
    return view_mats, builder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
