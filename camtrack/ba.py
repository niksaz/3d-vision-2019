__all__ = [
    'run_bundle_adjustment',
]

from typing import List
from recordclass import recordclass
import sortednp as snp

import numpy as np
import autograd.numpy as anp
from autograd import jacobian
import scipy

from corners import FrameCorners
from _camtrack import *

CornerInlier = recordclass('CornerInlier', ('point3d_id', 'frame_num', 'point'))


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    print('Running bundle adjustment...')
    point3d_ids = set()
    inlier_corners = []
    non_empty_view_mats = []
    was_frame_unmatched = [True]
    for frame_corners, view_mat in zip(list_of_corners[1:], view_mats[1:]):
        _, (indices_3d, indices_2d) = snp.intersect(
            pc_builder.ids.flatten(), frame_corners.ids.flatten(), indices=True)
        points3d = pc_builder.points[indices_3d]
        points2d = frame_corners.points[indices_2d]
        proj_mat = intrinsic_mat @ view_mat
        indices = calc_inlier_indices(points3d, points2d, proj_mat, max_inlier_reprojection_error)
        was_frame_unmatched.append(len(indices) == 0)
        if was_frame_unmatched[-1]:
            continue
        frame_num = len(non_empty_view_mats)
        non_empty_view_mats.append(view_mat)
        for ind in indices:
            point3d_id = pc_builder.ids[indices_3d[ind], 0]
            point3d_ids.add(point3d_id)
            corner_inlier = CornerInlier(point3d_id, frame_num, anp.array(points2d[ind]))
            inlier_corners.append(corner_inlier)
    point3d_ids = sorted(list(point3d_ids))
    for i, (point3d_id, _, _) in enumerate(inlier_corners):
        inlier_corners[i].point3d_id = point3d_ids.index(point3d_id)

    N = len(non_empty_view_mats)
    M = len(point3d_ids)
    if N == 0 or M == 0:
        return view_mats
    p = np.zeros(6*N+3*M)
    for frame_num in range(N):
        p_id = 6*frame_num
        r_vec, t_vec = view_mat3x4_to_rodrigues_and_translation(non_empty_view_mats[frame_num])
        p[p_id:p_id+3] = r_vec.reshape(-1)
        p[p_id+3:p_id+6] = t_vec.reshape(-1)
    _, (indices, _) = snp.intersect(pc_builder.ids.flatten(), np.array(point3d_ids), indices=True)
    p[6*N:] = pc_builder.points[indices].reshape(-1)

    _optimize_parameters(p, N, M, intrinsic_mat, inlier_corners)

    final_view_mats = []
    optimized_view_mat_num = 0
    for was_unmatched, view_mat in zip(was_frame_unmatched, view_mats):
        if was_unmatched:
            final_view_mats.append(view_mat)
        else:
            p_id = 6*optimized_view_mat_num
            optimized_view_mat_num += 1
            r_vec = p[p_id:p_id+3].reshape(3, 1)
            t_vec = p[p_id+3:p_id+6].reshape(3, 1)
            view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
            final_view_mats.append(view_mat)
    pc_builder.update_points(np.array(point3d_ids), p[6*N:].reshape(-1, 3))
    return final_view_mats


def is_pos_def(X):
    return np.all(np.linalg.eigvals(X) > 0)


def _rodrigues_vec_to_r_mat(r_vec):
    theta = anp.linalg.norm(r_vec)
    if theta == 0:
        return anp.eye(3)
    r_vec = r_vec / theta
    T = anp.array([[0, -r_vec[2], r_vec[1]],
                   [r_vec[2], 0, -r_vec[0]],
                   [-r_vec[1], r_vec[0], 0]])
    R = anp.cos(theta)*anp.identity(3) + (1 - anp.cos(theta))*anp.dot(r_vec, r_vec.T) + anp.sin(theta)*T
    return R


def _project_point(point3d, proj_mat):
    point3d_hom = anp.hstack((point3d, 1))
    point2d = anp.dot(proj_mat, point3d_hom)
    point2d = point2d / point2d[2]
    return point2d.T[:2]


def _optimize_parameters(p: np.ndarray,
                         N: int,
                         M: int,
                         intrinsic_mat: np.ndarray,
                         inlier_corners: List[CornerInlier]):
    K = 6*N + 3*M
    T = len(inlier_corners)
    assert p.size == K
    intrinsic_mat = anp.array(intrinsic_mat)

    def reproj_errors(params):
        proj_mats = []
        for frame_num in range(N):
            p_id = 6*frame_num
            r_mat = _rodrigues_vec_to_r_mat(params[p_id:p_id + 3])
            t_vec = params[p_id + 3:p_id + 6].reshape(3, 1)
            view_mat = anp.hstack((r_mat, t_vec))
            proj_mat = anp.dot(intrinsic_mat, view_mat)
            proj_mats.append(proj_mat)
        errors = []
        for point3d_id, frame_num, point in inlier_corners:
            p_id = 6*N + 3*point3d_id
            point3d = params[p_id:p_id + 3]
            proj_point2d = _project_point(point3d, proj_mats[frame_num])
            proj_error = (point - proj_point2d).reshape(-1)
            norm = anp.linalg.norm(proj_error)
            errors.append(norm)
        return anp.array(errors)

    def cum_reproj_error(params):
        errors = reproj_errors(params)
        return anp.dot(errors, errors)

    params = anp.array(p)
    reproj_error = cum_reproj_error(params)
    print('Initial reprojection error:', reproj_error)
    print('Computing Jacobian matrix... The size will be {}x{}'.format(T, K))

    jacobian_reproj_errors = jacobian(reproj_errors)
    J = jacobian_reproj_errors(params)
    A = J.T.dot(J)
    best_reporj_error = reproj_error
    best_p = p
    for lambd in [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
        B = A + lambd * np.diag(np.diagonal(A))
        U = B[:6*N, :6*N]
        W = B[:6*N, 6*N:]
        V = B[6*N:, 6*N:]
        V_inv = np.linalg.inv(V)

        u = reproj_errors(params)
        g = J.T.dot(u)
        g_c, g_x = g[:6*N], g[6*N:]
        L = U - W.dot(V_inv).dot(W.T)
        R = W.dot(V_inv).dot(g_x) - g_c
        if not is_pos_def(L):
            print('Non positive-definite. Returning...')
            return
        L = scipy.linalg.cholesky(L, lower=True)
        delta_c = scipy.linalg.cho_solve((L, True), R)
        delta_x = V_inv.dot(-g_x - W.T.dot(delta_c))
        new_p = p.copy()
        new_p += np.hstack((delta_c, delta_x))
        reproj_error = cum_reproj_error(anp.array(new_p))
        print('Reprojection error={} for lambda={}'.format(reproj_error, lambd))
        if reproj_error < best_reporj_error:
            best_reporj_error = reproj_error
            best_p = new_p
    print('Final reprojection error:', best_reporj_error)
    p[:] = best_p
