"""
SORT (Simple Online and Realtime Tracking) tracker.

Assigns stable IDs to detected people across video frames so each
queuing person keeps their own identity. Every track runs a
constant-velocity Kalman filter and is matched to new detections using
IoU + the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

No extra dependencies: relies on numpy + scipy (already required by
the YOLOv5 environment).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - intersection
    return intersection / union if union > 0 else 0.0


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / h if h > 0 else 1.0
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w > 0 else 0.0
    return np.array([
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ])


class _KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf_x = np.zeros((7, 1))
        self.kf_x[:4] = convert_bbox_to_z(bbox)
        self.kf_P = 1000.0 * np.eye(7)
        self.kf_P[4:, 4:] *= 1000.0
        self.kf_P *= 10.0
        self.kf_F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.kf_H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)
        self.kf_R = np.eye(4)
        self.kf_R[2:, 2:] *= 10.0
        self.kf_Q = np.eye(7)
        self.kf_Q[4:, 4:] *= 0.01
        self.kf_Q[-1, -1] *= 0.01

        self.id = _KalmanBoxTracker.count
        _KalmanBoxTracker.count += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        if (self.kf_x[6] + self.kf_x[2]) <= 0:
            self.kf_x[6] *= 0.0
        self.kf_x = self.kf_F @ self.kf_x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        z = convert_bbox_to_z(bbox)
        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        y = z - self.kf_H @ self.kf_x
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(7) - K @ self.kf_H) @ self.kf_P

    def get_state(self):
        return convert_x_to_bbox(self.kf_x).reshape(1, -1)


def _associate_detections_to_trackers(detections, trackers, iou_threshold):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = bbox_iou(det, trk)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = np.asarray(linear_sum_assignment(-iou_matrix)).T
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = [d for d in range(len(detections))
                            if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers))
                          if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class SortTracker:
    """
    Frame-to-frame person tracker.

    Parameters
    ----------
    max_age : int
        Frames a track survives without a matching detection.
    min_hits : int
        Consecutive matches required before a new track is reported.
    iou_threshold : float
        Minimum IoU for a detection/track association.
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        _KalmanBoxTracker.count = 0

    def update(self, detections):
        """
        Feed this frame's detections and get tracked boxes with IDs.

        Parameters
        ----------
        detections : np.ndarray
            Shape (N, 5): (x1, y1, x2, y2, confidence).

        Returns
        -------
        np.ndarray
            Shape (M, 5): (x1, y1, x2, y2, track_id).
        """
        self.frame_count += 1
        detections = np.asarray(detections, dtype=float).reshape(-1, 5)
        if len(detections) == 0:
            detections = np.empty((0, 5))

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0.0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = _associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )

        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])

        for i in unmatched_dets:
            self.trackers.append(_KalmanBoxTracker(detections[i]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            box = trk.get_state()[0]
            if (trk.time_since_update < 1 and
                    (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((box, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))