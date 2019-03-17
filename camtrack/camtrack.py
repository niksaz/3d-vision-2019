#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple
from collections import namedtuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


def _init_on_two_frames(frame_corners_0, frame_corners_1, intrinsic_mat, triang_params) \
        -> Tuple[Pose, np.ndarray, np.ndarray]:
    correspondences = build_correspondences(frame_corners_0, frame_corners_1)
    if len(correspondences.ids) < 5:
        return None, None, None
    pts_0 = correspondences.points_1
    pts_1 = correspondences.points_2
    # Check if the E matrix will be well-defined based on homography matrix
    H, h_mask = cv2.findHomography(pts_0, pts_1, method=cv2.RANSAC, ransacReprojThreshold=1.0, confidence=0.999)
    if np.nonzero(h_mask)[0].size / len(pts_0) > 0.9:
        return None, None, None
    E, e_mask = cv2.findEssentialMat(pts_0, pts_1, cameraMatrix=intrinsic_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None, None, None
    correspondences = build_correspondences(frame_corners_0, frame_corners_1, ids_to_remove=np.where(e_mask == 0)[0])
    R1, R2, t_d = cv2.decomposeEssentialMat(E)
    best_pose, best_pts, best_ids = None, None, None
    for R, t in [(R1, t_d), (R1, -t_d), (R2, t_d), (R2, -t_d)]:
        pose_1 = Pose(r_mat=R, t_vec=t)
        view_0 = eye3x4()
        view_1 = pose_to_view_mat3x4(pose_1)
        pts, ids = triangulate_correspondences(correspondences, view_0, view_1, intrinsic_mat, triang_params)
        if best_pose is None or len(ids) > len(best_ids):
            best_pose, best_pts, best_ids = pose_1, pts, ids
    return best_pose, best_pts, best_ids


def _find_best_pair(corner_storage: CornerStorage, intrinsic_mat: np.ndarray, triang_params: TriangulationParameters) \
        -> int:
    best_index, largest_size = None, None
    for frame_index in range(1, len(corner_storage)):
        _, _, ids = _init_on_two_frames(
            corner_storage[0], corner_storage[frame_index], intrinsic_mat, triang_params)
        if ids is None or len(ids) == 0:
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
    # Process the rest of the frames
    for frame_index in range(1, len(corner_storage)):
        print('Processing frame {}/{}'.format(frame_index, len(corner_storage)))
        if frame_index == paired_frame:
            view_mat = pose_to_view_mat3x4(pose)
            print('paired frame. skipping PnP')
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
            if len(objectPoints) < 4:
                return TrackingResult(is_successful=False, view_mats=None, builder=None)
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(objectPoints),
                np.array(imagePoints),
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
    return TrackingResult(is_successful=True, view_mats=view_mats, builder=builder)


def _track_camera(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
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
            return result.view_mats, result.builder
        else:
            print('Unsuccessful tracking. Restarting with weaker constraints...')
    return None, None


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
