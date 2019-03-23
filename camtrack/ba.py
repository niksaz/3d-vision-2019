from typing import List

import numpy as np

from corners import FrameCorners
from _camtrack import *


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    N = len(view_mats)
    M = pc_builder.ids.size
    p = np.zeros(6*N+3*M)
    for frame_num in range(N):
        p_id = 6*frame_num
        r_vec, t_vec = view_mat3x4_to_rodrigues_and_translation(view_mats[frame_num])
        p[p_id:p_id+3] = r_vec.reshape(-1)
        p[p_id+3:p_id+6] = t_vec.reshape(-1)
    p[6*N:] = pc_builder.points.reshape(-1)

    _optimize_parameters(p, intrinsic_mat, list_of_corners)

    view_mats = []
    for frame_num in range(N):
        p_id = 6*frame_num
        r_vec = p[p_id:p_id+3].reshape(3, 1)
        t_vec = p[p_id+3:p_id+6].reshape(3, 1)
        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        view_mats.append(view_mat)
    pc_builder.update_points(pc_builder.ids, p[6*N:].reshape(-1, 3))
    return view_mats


def _optimize_parameters(p: np.ndarray,
                         intrinsic_mat: np.ndarray,
                         list_of_corners: List[FrameCorners]) -> None:
    pass
