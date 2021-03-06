#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple
from collections import namedtuple

import cv2
import numpy as np

from ba import run_bundle_adjustment
from corners import CornerStorage, without_long_jump_corners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


def _init_on_two_frames(frame_corners_0, frame_corners_1, intrinsic_mat, triang_params) \
        -> Tuple[Pose, np.ndarray, np.ndarray]:
    best_pose, best_pts, best_ids = None, np.array([]), np.array([])
    correspondences = build_correspondences(frame_corners_0, frame_corners_1)
    if len(correspondences.ids) < 5:
        return best_pose, best_pts, best_ids
    pts_0 = correspondences.points_1
    pts_1 = correspondences.points_2
    # Check if the E matrix will be well-defined based on homography matrix
    H, h_mask = cv2.findHomography(pts_0, pts_1, method=cv2.RANSAC, ransacReprojThreshold=1.0, confidence=0.999)
    if np.nonzero(h_mask)[0].size / len(pts_0) > 0.9:
        return best_pose, best_pts, best_ids
    E, e_mask = cv2.findEssentialMat(
        pts_0, pts_1, cameraMatrix=intrinsic_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return best_pose, best_pts, best_ids
    correspondences = remove_correspondences_with_ids(correspondences, np.where(e_mask == 0)[0])
    R1, R2, t_d = cv2.decomposeEssentialMat(E)
    for R, t in [(R1, t_d), (R1, -t_d), (R2, t_d), (R2, -t_d)]:
        pose_1 = Pose(R.T, R.T @ -t)
        view_1 = pose_to_view_mat3x4(pose_1)
        view_0 = eye3x4()
        pts, ids = triangulate_correspondences(correspondences, view_0, view_1, intrinsic_mat, triang_params)
        if len(best_ids) < len(ids):
            best_pose, best_pts, best_ids = pose_1, pts, ids
    return best_pose, best_pts, best_ids


def _find_best_pair(corner_storage: CornerStorage, intrinsic_mat: np.ndarray, triang_params: TriangulationParameters) \
        -> int:
    best_index, largest_size = None, None
    for frame_index in range(1, len(corner_storage)):
        _, _, ids = _init_on_two_frames(
            corner_storage[0], corner_storage[frame_index], intrinsic_mat, triang_params)
        if len(ids) == 0:
            continue
        print('Best pose for frame {} contains {} points'.format(frame_index, len(ids)))
        if (ids is not None) and (best_index is None or len(ids) > largest_size):
            best_index = frame_index
            largest_size = len(ids)
    return best_index


def _try_to_update_cloud(
        frame_corners_0, frame_corners_1, view_0, view_1, intrinsic_mat, builder, triang_params) -> int:
    correspondences = build_correspondences(frame_corners_0, frame_corners_1, ids_to_remove=builder.ids)
    if len(correspondences.ids) == 0:
        return 0
    pts, ids = triangulate_correspondences(
        correspondences, view_0, view_1, intrinsic_mat, triang_params)
    builder.add_points(ids, pts)
    return len(ids)


TrackingResult = namedtuple(
    'TrackingResult',
    ('is_successful', 'view_mats', 'builder'))


def _track_camera_with_params(
        corner_storage: CornerStorage, intrinsic_mat: np.ndarray, triang_params: TriangulationParameters) \
        -> TrackingResult:
    # Init the camera position based on the first and best paired frames
    paired_frame = _find_best_pair(corner_storage, intrinsic_mat, triang_params)
    if paired_frame is None:
        return TrackingResult(is_successful=False, view_mats=None, builder=None)
    pose, pts, ids = _init_on_two_frames(corner_storage[0], corner_storage[paired_frame], intrinsic_mat, triang_params)
    print('The best paired frame is {}. The initial cloud size is {}'.format(paired_frame, len(pts)))
    view_mats = [eye3x4()]
    builder = PointCloudBuilder()
    builder.add_points(ids, pts)
    ba_frames_in_adjustment = 20
    ba_run_every_frames = 5
    # Process the rest of the frames
    for frame_index in range(1, len(corner_storage)):
        print('Processing frame {}/{}'.format(frame_index, len(corner_storage)))
        if frame_index == paired_frame:
            view_mat = pose_to_view_mat3x4(pose)
            print('paired frame. skipping PnP')
        else:
            frame_corners = corner_storage[frame_index]
            ids = []
            object_points = []
            image_points = []
            for corner_id, point in zip(frame_corners.ids, frame_corners.points):
                indices_x, _ = np.nonzero(builder.ids == corner_id)
                if len(indices_x) == 0:
                    continue
                paired_frame = indices_x[0]
                ids.append(corner_id)
                object_points.append(builder.points[paired_frame])
                image_points.append(point)
            if len(object_points) < 4:
                return TrackingResult(is_successful=False, view_mats=None, builder=None)
            object_points = np.array(object_points, dtype=np.float64).reshape((len(object_points), 1, 3))
            image_points = np.array(image_points, dtype=np.float64).reshape((len(object_points), 1, 2))
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                cameraMatrix=intrinsic_mat,
                distCoeffs=np.array([]),
                flags=cv2.SOLVEPNP_EPNP)
            if not retval:
                return TrackingResult(is_successful=False, view_mats=None, builder=None)
            view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            print('computed camera position based on {} inliers'.format(len(inliers)))
        view_mats.append(view_mat)
        total_new_points = 0
        for another_frame in range(frame_index):
            total_new_points += _try_to_update_cloud(
                corner_storage[another_frame], corner_storage[frame_index],
                view_mats[another_frame], view_mat,
                intrinsic_mat, builder, triang_params)
        print('new △ points={} cloud size={}'.format(total_new_points, len(builder.ids)))
        # Try to improve parameters with bundle adjustment
        M = len(view_mats)
        if M >= ba_frames_in_adjustment and M % ba_run_every_frames == 0:
            view_mats[M-ba_frames_in_adjustment:] = run_bundle_adjustment(
                intrinsic_mat=intrinsic_mat,
                list_of_corners=list(corner_storage)[M-ba_frames_in_adjustment:],
                max_inlier_reprojection_error=triang_params.max_reprojection_error,
                view_mats=view_mats[M-ba_frames_in_adjustment:],
                pc_builder=builder)
    return TrackingResult(is_successful=True, view_mats=view_mats, builder=builder)


def _track_camera(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    # 2-D metrics to filter untrustworthy corners
    max_dst_optical_flow = 0.1
    corner_storage = without_long_jump_corners(corner_storage, max_dst_optical_flow)
    # Choose the strictest tracking parameters that lead to a successful tracking
    strict_constraints = TriangulationParameters(
        max_reprojection_error=1.0,
        min_triangulation_angle_deg=5.0,
        min_depth=0.1)
    medium_constraints = TriangulationParameters(
        max_reprojection_error=1.0,
        min_triangulation_angle_deg=3.0,
        min_depth=0.1)
    weak_constraints = TriangulationParameters(
        max_reprojection_error=1.0,
        min_triangulation_angle_deg=1.0,
        min_depth=0.1)
    for triangulation_parameters in [strict_constraints, medium_constraints, weak_constraints]:
        result = _track_camera_with_params(corner_storage, intrinsic_mat, triangulation_parameters)
        if result.is_successful:
            print('Tracking succeeded.')
            break
        else:
            print('Unsuccessful tracking. Restarting with weaker constraints...')
    if result.is_successful:
        view_mats = result.view_mats
        pc_builder = result.builder
        return view_mats, pc_builder
    else:
        return [], PointCloudBuilder()


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
