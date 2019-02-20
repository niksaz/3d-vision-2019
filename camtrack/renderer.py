#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo

import data3d


def _build_program():
    vertex_transform_shader = shaders.compileShader(
        """
        #version 140
        uniform mat4 mvp;

        in vec3 position;
        in vec3 in_color;
        
        out vec3 color;

        void main() {
            color = in_color;
            gl_Position = mvp * vec4(position, 1.0);
        }""",
        GL.GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(
        """
        #version 140
        
        in vec3 color;
        
        out vec3 out_color;

        void main() {
            out_color = color;
        }""",
        GL.GL_FRAGMENT_SHADER)
    return shaders.compileProgram(vertex_transform_shader, fragment_shader)


def _build_projection_matrix(fov_y, aspect_ratio, z_near, z_far):
    y_max = z_near * np.tan(fov_y / 2.0)
    x_max = y_max * aspect_ratio
    z_diff = z_near - z_far
    return np.array([
        [z_near / x_max, 0, 0, 0],
        [0, z_near / y_max, 0, 0],
        [0, 0, (z_far + z_near) / z_diff, 2 * z_far * z_near / z_diff],
        [0, 0, -1, 0]
    ], dtype=np.float32)


def _build_rotation_matrix(rotation):
    result_mat = np.eye(4, dtype=np.float32)
    result_mat[:3, :3] = rotation
    return result_mat


def _build_translation_matrix(translation):
    result_mat = np.eye(4, dtype=np.float32)
    result_mat[0, 3] = translation[0]
    result_mat[1, 3] = translation[1]
    result_mat[2, 3] = translation[2]
    return result_mat


# Inverse y and z axes
_opencvgl_matrix = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


class CameraTrackRenderer:
    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        self._program = _build_program()

        self._tracked_cam_parameters = tracked_cam_parameters

        self._tracked_cam_track = tracked_cam_track
        self._cam_count = len(tracked_cam_track)
        cam_points = np.array([cam_pose.t_vec for cam_pose in tracked_cam_track]).reshape(-1).astype(np.float32)
        self._cam_points_vbo = vbo.VBO(cam_points)
        self._cam_colors_vbo = vbo.VBO(np.ones_like(cam_points))

        self._count = len(point_cloud.ids)
        points = point_cloud.points.reshape(-1).astype(np.float32)
        colors = point_cloud.colors.reshape(-1).astype(np.float32)
        self._points_vbo = vbo.VBO(points)
        self._colors_vbo = vbo.VBO(colors)

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        aspect_ratio = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)

        proj_mat = _build_projection_matrix(camera_fov_y, aspect_ratio, 0.1, 100)
        unrotate_mat = _build_rotation_matrix(np.linalg.inv(camera_rot_mat))
        untranslate_mat = _build_translation_matrix(-camera_tr_vec)
        mvp = proj_mat.dot(unrotate_mat.dot(untranslate_mat.dot(_opencvgl_matrix)))

        self._render_in_mode(mvp, self._points_vbo, self._colors_vbo, self._count, GL.GL_POINTS)
        self._render_in_mode(mvp, self._cam_points_vbo, self._cam_colors_vbo, self._cam_count, GL.GL_LINE_STRIP)

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)
        cam_track_pos = self._tracked_cam_track[tracked_cam_track_pos]
        frustrum_points = self._get_frustrum_points(cam_track_pos,
                                                    self._tracked_cam_parameters.fov_y,
                                                    self._tracked_cam_parameters.aspect_ratio,
                                                    20.0)
        frustrum_border_points = np.array([
            frustrum_points[0], frustrum_points[1],
            frustrum_points[1], frustrum_points[2],
            frustrum_points[2], frustrum_points[3],
            frustrum_points[3], frustrum_points[0],
            cam_track_pos.t_vec, frustrum_points[0],
            cam_track_pos.t_vec, frustrum_points[1],
            cam_track_pos.t_vec, frustrum_points[2],
            cam_track_pos.t_vec, frustrum_points[3]], dtype=np.float32)

        points_vbo = vbo.VBO(frustrum_border_points.reshape(-1))
        yellow_color = [1, 1, 0]
        colors_vbo = vbo.VBO(np.array(yellow_color * len(frustrum_border_points), dtype=np.float32))
        self._render_in_mode(mvp, points_vbo, colors_vbo, len(frustrum_border_points), GL.GL_LINES)

        GLUT.glutSwapBuffers()

    @staticmethod
    def _get_frustrum_points(pose, fov_y, aspect_ratio, z_far):
        y_max = z_far * np.tan(fov_y)
        x_max = y_max * aspect_ratio
        signs = [(-1, -1), (-1, +1), (+1, +1), (+1, -1)]
        points = np.zeros((len(signs), 3), dtype=np.float32)
        for index, sign in enumerate(signs):
            viewed_point = np.array([x_max * sign[0], y_max * sign[1], z_far], dtype=np.float32)
            points[index] = pose.r_mat.dot(viewed_point) + pose.t_vec
        return points

    def _render_in_mode(self, mvp, points_vbo, colors_vbo, count, mode):
        shaders.glUseProgram(self._program)
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._program, 'mvp'),
            1, True, mvp)

        points_vbo.bind()
        arg_position = GL.glGetAttribLocation(self._program, 'position')
        GL.glEnableVertexAttribArray(arg_position)
        GL.glVertexAttribPointer(arg_position, 3, GL.GL_FLOAT, False, 0, points_vbo)

        colors_vbo.bind()
        arg_in_color = GL.glGetAttribLocation(self._program, 'in_color')
        GL.glEnableVertexAttribArray(arg_in_color)
        GL.glVertexAttribPointer(arg_in_color, 3, GL.GL_FLOAT, False, 0, colors_vbo)

        GL.glDrawArrays(mode, 0, count)

        points_vbo.unbind()
        colors_vbo.unbind()

        GL.glDisableVertexAttribArray(arg_position)
        GL.glDisableVertexAttribArray(arg_in_color)
        shaders.glUseProgram(0)
