#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    radius = 10
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.03,
                          minDistance=7,
                          blockSize=7)
    # params for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(
                     cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    image_0_8bit = frame_sequence[0]
    centers = cv2.goodFeaturesToTrack(image_0_8bit, mask=None, **feature_params)
    ids = np.array(range(len(centers)))
    next_id = len(centers)
    radiuses = np.array(np.full(len(centers), radius))
    corners = FrameCorners(ids, centers.reshape((-1, 2)), radiuses)

    builder.set_corners_at_frame(0, corners)
    image_0_8bit = np.uint8(image_0_8bit * 255.0)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1_8bit = np.uint8(image_1 * 255.0)
        p1, st, err = cv2.calcOpticalFlowPyrLK(image_0_8bit, image_1_8bit, centers, None, **lk_params)
        same_ids = []
        for i in range(len(ids)):
            if st[i] == 1:
                same_ids.append(i)

        if p1 is not None:
            ids = ids[same_ids]
            centers = centers[same_ids]
            radiuses = radiuses[same_ids]
            centers = p1[same_ids]

        if len(centers) < feature_params['maxCorners']:
            p0 = cv2.goodFeaturesToTrack(image_1, mask=None, **feature_params)
            new_ids = []
            new_centers = []
            new_radiuses = []
            for arr in p0:
                if len(ids) + len(new_ids) >= feature_params['maxCorners']:
                    break
                closest_dst = None
                for center in centers:
                    dst = np.linalg.norm(center[0] - arr[0])
                    if closest_dst is None or dst < closest_dst:
                        closest_dst = dst
                if closest_dst > feature_params['minDistance']:
                    new_ids.append(next_id)
                    new_centers.append(arr)
                    new_radiuses.append(radius)
                    next_id += 1
            if len(new_ids) > 0:
                ids = np.concatenate([ids, new_ids])
                centers = np.concatenate([centers, new_centers])
                radiuses = np.concatenate([radiuses, new_radiuses])

        corners = FrameCorners(ids, centers.reshape((-1, 2)), radiuses)
        builder.set_corners_at_frame(frame, corners)
        image_0_8bit = image_1_8bit


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
