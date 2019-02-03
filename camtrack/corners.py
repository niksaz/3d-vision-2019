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
    radius = 15
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1,
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
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(image_0_8bit, image_1_8bit, centers, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(image_1_8bit, image_0_8bit, p1, None, **lk_params)
        d = abs(centers - p0r).reshape(-1, 2).max(-1)
        present_fs = d < 1

        ids = ids[present_fs]
        centers = p1[present_fs]
        radiuses = radiuses[present_fs]

        if len(centers) < feature_params['maxCorners']:
            mask = np.full(image_1.shape, 255, dtype=np.uint8)
            for arr in centers:
                x, y = arr[0]
                cv2.circle(mask, (x, y), feature_params['minDistance'], 0, -1)
            p0 = cv2.goodFeaturesToTrack(image_1, mask=mask, **feature_params)
            if p0 is not None:
                new_ids = []
                new_centers = []
                new_radiuses = []
                for arr in p0:
                    if len(ids) + len(new_ids) >= feature_params['maxCorners']:
                        break
                    new_ids.append(next_id)
                    new_centers.append(arr)
                    new_radiuses.append(radius)
                    next_id += 1
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
