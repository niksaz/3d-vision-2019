__all__ = [
    'Correspondences',
    'PointCloudBuilder',
    'TriangulationParameters',
    'build_correspondences',
    'remove_correspondences_with_ids',
    'calc_point_cloud_colors',
    'calc_inlier_indices',
    'check_inliers_mask',
    'check_baseline',
    'compute_reprojection_errors',
    'create_cli',
    'draw_residuals',
    'eye3x4',
    'project_points',
    'view_mat3x4_to_rodrigues_and_translation',
    'rodrigues_and_translation_to_view_mat3x4',
    'to_opencv_camera_mat3x3',
    'triangulate_correspondences',
    'view_mat3x4_to_pose',
    'pose_to_view_mat3x4'
]

from collections import namedtuple
from typing import List, Tuple

import click
import cv2
import numpy as np
import pims
from sklearn.preprocessing import normalize
import sortednp as snp

import frameseq
from corners import CornerStorage, FrameCorners, build, load
from data3d import CameraParameters, PointCloud, Pose
from data3d import read_camera_parameters, write_point_cloud, write_poses


def to_opencv_camera_mat3x3(camera_parameters: CameraParameters,
                            image_height: int) -> np.ndarray:
    h = image_height
    w = h * camera_parameters.aspect_ratio
    h_to_f = 2.0 * np.tan(camera_parameters.fov_y / 2.0)
    f = h / h_to_f
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]])


_IDENTITY_3x4 = np.hstack(
    (np.eye(3, 3, dtype=np.float32),
     np.zeros((3, 1), dtype=np.float32))
)


def eye3x4() -> np.ndarray:
    return _IDENTITY_3x4


def view_mat3x4_to_pose(view_mat: np.ndarray) -> Pose:
    r_mat = view_mat[:, :3]
    t_vec = view_mat[:, 3]
    return Pose(r_mat.T, r_mat.T @ -t_vec)


def pose_to_view_mat3x4(pose: Pose) -> np.ndarray:
    return np.hstack((
        pose.r_mat.T,
        pose.r_mat.T @ -pose.t_vec.reshape(-1, 1)
    ))


def _to_homogeneous(points):
    return np.pad(points, ((0, 0), (0, 1)), 'constant', constant_values=(1,))


def project_points(points3d: np.ndarray, proj_mat: np.ndarray) -> np.ndarray:
    points3d = _to_homogeneous(points3d)
    points2d = np.dot(proj_mat, points3d.T)
    points2d /= points2d[[2]]
    return points2d[:2].T


def compute_reprojection_errors(points3d: np.ndarray, points2d: np.ndarray,
                                proj_mat: np.ndarray) -> np.ndarray:
    projected_points = project_points(points3d, proj_mat)
    points2d_diff = points2d - projected_points
    return np.linalg.norm(points2d_diff, axis=1)


def calc_inlier_indices(points3d: np.ndarray, points2d: np.ndarray,
                        proj_mat: np.ndarray, max_error: float) -> np.ndarray:
    errors = compute_reprojection_errors(points3d, points2d, proj_mat)
    mask = (errors <= max_error).flatten()
    indices = np.nonzero(mask)
    return indices[0]


def _to_camera_center(view_mat):
    return view_mat[:, :3].T @ -view_mat[:, 3]


def _calc_triangulation_angle_mask(view_mat_1: np.ndarray,
                                   view_mat_2: np.ndarray,
                                   points3d: np.ndarray,
                                   min_angle_deg: float) -> np.ndarray:
    camera_center_1 = _to_camera_center(view_mat_1)
    camera_center_2 = _to_camera_center(view_mat_2)
    vecs_1 = normalize(camera_center_1 - points3d)
    vecs_2 = normalize(camera_center_2 - points3d)
    coss = np.einsum('ij,ij->i', vecs_1, vecs_2)
    angles_mask = coss <= np.cos(np.deg2rad(min_angle_deg))
    return angles_mask


Correspondences = namedtuple(
    'Correspondences',
    ('ids', 'points_1', 'points_2')
)


TriangulationParameters = namedtuple(
    'TriangulationParameters',
    ('max_reprojection_error', 'min_triangulation_angle_deg', 'min_depth')
)


def remove_correspondences_with_ids(correspondences: Correspondences,
                                    ids_to_remove: np.ndarray) \
        -> Correspondences:
    ids = correspondences.ids.flatten()
    ids_to_remove = ids_to_remove.flatten()
    _, (indices_1, _) = snp.intersect(ids, ids_to_remove, indices=True)
    mask = np.full(ids.shape, True)
    mask[indices_1] = False
    return Correspondences(
        ids[mask],
        correspondences.points_1[mask],
        correspondences.points_2[mask]
    )


def build_correspondences(corners_1: FrameCorners, corners_2: FrameCorners,
                          ids_to_remove=None) -> Correspondences:
    ids_1 = corners_1.ids.flatten()
    ids_2 = corners_2.ids.flatten()
    _, (indices_1, indices_2) = snp.intersect(ids_1, ids_2, indices=True)
    corrs = Correspondences(
        ids_1[indices_1],
        corners_1.points[indices_1],
        corners_2.points[indices_2]
    )
    if ids_to_remove is not None:
        corrs = remove_correspondences_with_ids(corrs, ids_to_remove)
    return corrs


def _calc_z_mask(points3d, view_mat, min_depth):
    points3d = _to_homogeneous(points3d)
    points3d_in_camera_space = np.dot(view_mat, points3d.T)
    return points3d_in_camera_space[2].flatten() >= min_depth


def _calc_reprojection_error_mask(points3d, points2d_1, points2d_2,
                                  view_mat_1, view_mat_2, intrinsic_mat,
                                  max_reprojection_error):
    reproj_errs_1 = compute_reprojection_errors(points3d, points2d_1,
                                                intrinsic_mat @ view_mat_1)
    reproj_errs2 = compute_reprojection_errors(points3d, points2d_2,
                                               intrinsic_mat @ view_mat_2)
    reproj_err_mask = np.logical_and(
        reproj_errs_1.flatten() < max_reprojection_error,
        reproj_errs2.flatten() < max_reprojection_error
    )
    return reproj_err_mask


def triangulate_correspondences(correspondences: Correspondences,
                                view_mat_1: np.ndarray, view_mat_2: np.ndarray,
                                intrinsic_mat: np.ndarray,
                                parameters: TriangulationParameters) \
        -> Tuple[np.ndarray, np.ndarray]:
    points2d_1 = correspondences.points_1
    points2d_2 = correspondences.points_2

    normalized_points2d_1 = cv2.undistortPoints(
        points2d_1.reshape(-1, 1, 2),
        intrinsic_mat,
        np.array([])
    ).reshape(-1, 2)
    normalized_points2d_2 = cv2.undistortPoints(
        points2d_2.reshape(-1, 1, 2),
        intrinsic_mat,
        np.array([])
    ).reshape(-1, 2)

    points3d = cv2.triangulatePoints(view_mat_1, view_mat_2,
                                     normalized_points2d_1.T,
                                     normalized_points2d_2.T)
    points3d = cv2.convertPointsFromHomogeneous(points3d.T).reshape(-1, 3)

    reprojection_error_mask = _calc_reprojection_error_mask(
        points3d,
        points2d_1,
        points2d_2,
        view_mat_1,
        view_mat_2,
        intrinsic_mat,
        parameters.max_reprojection_error
    )
    z_mask_1 = _calc_z_mask(points3d, view_mat_1, parameters.min_depth)
    z_mask_2 = _calc_z_mask(points3d, view_mat_2, parameters.min_depth)
    angle_mask = _calc_triangulation_angle_mask(
        view_mat_1,
        view_mat_2,
        points3d,
        parameters.min_triangulation_angle_deg
    )
    common_mask = reprojection_error_mask & z_mask_1 & z_mask_2 & angle_mask

    return points3d[common_mask], correspondences.ids[common_mask]


def check_inliers_mask(inliers_mask: np.ndarray,
                       min_inlier_count: int,
                       min_inlier_ratio: float) -> bool:
    inlier_count = np.count_nonzero(inliers_mask)
    inlier_ratio = inlier_count / float(inliers_mask.size)
    return (inlier_count >= min_inlier_count and
            inlier_ratio >= min_inlier_ratio)


def check_baseline(view_mat_1: np.ndarray, view_mat_2: np.ndarray,
                   min_distance: float) -> bool:
    camera_center_1 = _to_camera_center(view_mat_1)
    camera_center_2 = _to_camera_center(view_mat_2)
    distance = np.linalg.norm(camera_center_2 - camera_center_1)
    return distance >= min_distance


def view_mat3x4_to_rodrigues_and_translation(view_mat: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    r_mat = view_mat[:, :3]
    t_vec = view_mat[:, 3]
    r_vec, _ = cv2.Rodrigues(r_mat)
    return r_vec, t_vec


def rodrigues_and_translation_to_view_mat3x4(r_vec: np.ndarray,
                                             t_vec: np.ndarray) -> np.ndarray:
    rot_mat, _ = cv2.Rodrigues(r_vec)
    view_mat = np.hstack((rot_mat, t_vec))
    return view_mat


class PointCloudBuilder:

    __slots__ = ('_ids', '_points', '_colors')

    def __init__(self, ids: np.ndarray = None, points: np.ndarray = None,
                 colors: np.ndarray = None) -> None:
        super().__init__()
        self._ids = ids if ids is not None else np.array([], dtype=np.int)
        self._points = points if points is not None else np.array([])
        self._colors = colors
        self._sort_data()

    @property
    def ids(self) -> np.ndarray:
        return self._ids

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def colors(self) -> np.ndarray:
        return self._colors

    def __iter__(self):
        yield self.ids
        yield self.points
        yield self.colors

    def add_points(self, ids: np.ndarray, points: np.ndarray) -> None:
        ids = ids.reshape(-1, 1)
        points = points.reshape(-1, 3)
        _, (idx_1, idx_2) = snp.intersect(self.ids.flatten(), ids.flatten(),
                                          indices=True)
        self.points[idx_1] = points[idx_2]
        self._ids = np.vstack((self.ids, np.delete(ids, idx_2, axis=0)))
        self._points = np.vstack((self.points, np.delete(points, idx_2, axis=0)))
        self._sort_data()

    def set_colors(self, colors: np.ndarray) -> None:
        assert self._ids.size == colors.shape[0]
        self._colors = colors

    def update_points(self, ids: np.ndarray, points: np.ndarray) -> None:
        _, (idx_1, idx_2) = snp.intersect(self.ids.flatten(), ids.flatten(),
                                          indices=True)
        self._points[idx_1] = points[idx_2]

    def build_point_cloud(self) -> PointCloud:
        return PointCloud(self.ids, self.points, self.colors)

    def _sort_data(self):
        sorting_idx = np.argsort(self.ids.flatten())
        self._ids = self.ids[sorting_idx].reshape(-1, 1)
        self._points = self.points[sorting_idx].reshape(-1, 3)
        if self.colors is not None:
            self._colors = self.colors[sorting_idx].reshape(-1, 3)


def _to_int_tuple(point):
    return tuple(map(int, np.round(np.squeeze(point))))


def _draw_cross(bgr, point, size, color):
    x, y = point
    radius = int(np.round(size / 2))
    cv2.line(bgr, (x + radius, y + radius), (x - radius, y - radius), color)
    cv2.line(bgr, (x + radius, y - radius), (x - radius, y + radius), color)


def draw_residuals(grayscale_image: np.ndarray, corners: FrameCorners,
                   point_cloud: PointCloud, camera_params: CameraParameters,
                   pose: Pose) -> np.ndarray:
    bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    intrinsic_mat = to_opencv_camera_mat3x3(camera_params,
                                            grayscale_image.shape[0])
    proj_mat = intrinsic_mat @ pose_to_view_mat3x4(pose)
    _, (point_cloud_idx, corners_idx) = snp.intersect(
        point_cloud.ids.flatten(),
        corners.ids.flatten(),
        indices=True
    )
    corner_points = corners.points[corners_idx]
    projected_points = project_points(point_cloud.points[point_cloud_idx],
                                      proj_mat)
    corner_sizes = corners.sizes[corners_idx]

    zipped_arrays = zip(projected_points, corner_points, corner_sizes)
    for projected_point, corner_point, corner_size in zipped_arrays:
        corner_point = _to_int_tuple(corner_point)
        projected_point = _to_int_tuple(projected_point)
        corner_radius = int(corner_size.item() / 2)
        cv2.line(bgr, corner_point, projected_point, (0.7, 0.7, 0))
        cv2.circle(bgr, corner_point, corner_radius, (0, 1, 0))
        _draw_cross(bgr, projected_point, 5, (0.5, 0, 1))

    return bgr


def calc_point_cloud_colors(pc_builder: PointCloudBuilder,
                            rgb_sequence: pims.FramesSequence,
                            view_mats: List[np.ndarray],
                            intrinsic_mat: np.ndarray,
                            corner_storage: CornerStorage,
                            max_reproj_error: float) -> None:
    point_cloud_points = np.zeros((corner_storage.max_corner_id() + 1, 3))
    point_cloud_points[pc_builder.ids.flatten()] = pc_builder.points

    color_sums = np.zeros_like(point_cloud_points)
    color_counts = np.zeros_like(color_sums)

    with click.progressbar(zip(rgb_sequence, view_mats, corner_storage),
                           label='Calculating colors',
                           length=len(view_mats)) as bar:
        for image, view, corners in bar:
            proj_mat = intrinsic_mat @ view
            points3d = point_cloud_points[corners.ids.flatten()]
            with np.errstate(invalid='ignore'):
                errors = compute_reprojection_errors(points3d, corners.points,
                                                     proj_mat)
                errors = np.nan_to_num(errors)

            consistency_mask = (
                (errors <= max_reproj_error) &
                (corners.points[:, 0] >= 0) &
                (corners.points[:, 1] >= 0) &
                (corners.points[:, 0] < image.shape[1] - 0.5) &
                (corners.points[:, 1] < image.shape[0] - 0.5)).flatten()
            ids_to_process = corners.ids[consistency_mask].flatten()
            corner_points = np.round(
                corners.points[consistency_mask]
            ).astype(np.int32)

            rows = corner_points[:, 1].flatten()
            cols = corner_points[:, 0].flatten()
            color_sums[ids_to_process] += image[rows, cols]
            color_counts[ids_to_process] += 1

    nonzero_mask = (color_counts[:, 0] != 0).flatten()
    color_sums[nonzero_mask] /= color_counts[nonzero_mask]
    colors = color_sums[pc_builder.ids.flatten()]

    pc_builder.set_colors(colors)


def _resize_image_for_display_resolution(img):
    display_width, display_height = 1280, 720
    img_width, img_height = img.shape[1], img.shape[0]
    scale_width = min(1.0, display_width / img_width)
    scale_height = min(1.0, display_height / img_height)
    scale = min(scale_width, scale_height)
    dim = (int(scale * img_width), int(scale * img_height))
    return cv2.resize(img, dim)


def create_cli(track_and_calc_colors):
    @click.command()
    @click.argument('frame_sequence')
    @click.argument('camera', type=click.File('r'))
    @click.argument('track_destination', type=click.File('w'))
    @click.argument('point_cloud_destination', type=click.File('w'))
    @click.option('file_to_load_corners', '--load-corners',
                  type=click.File('rb'))
    @click.option('--show', is_flag=True)
    def cli(frame_sequence, camera, track_destination,
            point_cloud_destination,
            file_to_load_corners, show):
        """
        FRAME_SEQUENCE path to a video file or shell-like wildcard describing
        multiple images\n
        CAMERA intrinsic parameters of camera\n
        TRACK_DESTINATION path to file for dumping result camera track\n
        POINT_CLOUD_DESTINATION path to file for dumping result point cloud
        """
        sequence = frameseq.read_grayscale_f32(frame_sequence)
        if file_to_load_corners is not None:
            corner_storage = load(file_to_load_corners)
        else:
            corner_storage = build(sequence)

        camera_parameters = read_camera_parameters(camera)
        poses, point_cloud = track_and_calc_colors(camera_parameters,
                                                   corner_storage,
                                                   frame_sequence)
        write_poses(poses, track_destination)
        write_point_cloud(point_cloud, point_cloud_destination)

        if show:
            click.echo(
                "Press 'q' to stop, 'd' to go forward, 'a' to go backward, "
                "'r' to restart"
            )
            frame = 0
            while True:
                grayscale = sequence[frame]
                bgra = draw_residuals(grayscale, corner_storage[frame],
                                      point_cloud, camera_parameters,
                                      poses[frame])
                bgra = _resize_image_for_display_resolution(bgra)
                cv2.imshow('Frame', bgra)
                key = chr(cv2.waitKey(20) & 0xFF)
                if key == 'r':
                    frame = 0
                if key == 'a' and frame > 0:
                    frame -= 1
                if key == 'd' and frame + 1 < len(corner_storage):
                    frame += 1
                if key == 'q':
                    break
    return cli
