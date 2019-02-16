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


def _build_example_program():
    example_vertex_shader = shaders.compileShader(
        """
        #version 140
        uniform mat4 mvp;

        in vec3 position;

        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
        }""",
        GL.GL_VERTEX_SHADER
    )
    example_fragment_shader = shaders.compileShader(
        """
        #version 140
        out vec3 out_color;

        void main() {
            out_color = vec3(1, 0, 0);
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        example_vertex_shader, example_fragment_shader
    )


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

        self._example_buffer_object = vbo.VBO(np.array([0, 0, 0], dtype=np.float32))

        self._example_program = _build_example_program()

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

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self._render_example_point(np.eye(4))

        GLUT.glutSwapBuffers()

    def _render_example_point(self, mvp):
        shaders.glUseProgram(self._example_program)
        self._example_buffer_object.bind()
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._example_program, 'mvp'),
            1, True, mvp)
        position_loc = GL.glGetAttribLocation(self._example_program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._example_buffer_object)

        GL.glDrawArrays(GL.GL_POINTS, 0, 1)

        GL.glDisableVertexAttribArray(position_loc)
        self._example_buffer_object.unbind()
        shaders.glUseProgram(0)
