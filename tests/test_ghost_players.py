"""Regression: one person close to the camera (head + chest only) became
Players 1-3. RTMPose guesses the legs far outside the image at low
confidence, each crop guessed differently, and every variant was accepted
as a new player."""

import numpy as np

from conftest import make_cfg
from motionplay.skeleton import J, NUM_KEYPOINTS
from motionplay.tracking.tracker import PlayerTracker
from motionplay.types import PoseCandidate

W, H = 1280, 720
LOWER = [J.LEFT_HIP, J.RIGHT_HIP, J.HIP, J.LEFT_KNEE, J.RIGHT_KNEE, J.LEFT_ANKLE, J.RIGHT_ANKLE,
         J.LEFT_BIG_TOE, J.RIGHT_BIG_TOE, J.LEFT_SMALL_TOE, J.RIGHT_SMALL_TOE, J.LEFT_HEEL, J.RIGHT_HEEL]


def closeup_pose(rng, leg_spread: float, from_detection: bool) -> PoseCandidate:
    k = np.zeros((NUM_KEYPOINTS, 2), np.float32)
    s = np.full(NUM_KEYPOINTS, 0.9, np.float32)
    k[J.HEAD] = (640, 230); k[J.NOSE] = (640, 520)
    k[J.LEFT_EYE] = (700, 470); k[J.RIGHT_EYE] = (580, 470)
    k[J.LEFT_EAR] = (760, 480); k[J.RIGHT_EAR] = (520, 480)
    k[J.NECK] = (640, 640); k[J.LEFT_SHOULDER] = (900, 690); k[J.RIGHT_SHOULDER] = (380, 690)
    k[J.LEFT_ELBOW] = (1000, 900); k[J.RIGHT_ELBOW] = (280, 900)
    k[J.LEFT_WRIST] = (1050, 1100); k[J.RIGHT_WRIST] = (230, 1100)
    # Legs: guessed below the image, at low confidence, different each time.
    for j in LOWER:
        side = 1 if "LEFT" in J(j).name else (-1 if "RIGHT" in J(j).name else 0)
        k[j] = (640 + side * leg_spread * (1 + rng.random()), 1300 + 600 * rng.random())
        s[j] = 0.15 + 0.2 * rng.random()
    s[[J.LEFT_ELBOW, J.RIGHT_ELBOW, J.LEFT_WRIST, J.RIGHT_WRIST]] = 0.3
    k += rng.normal(0, 2, k.shape).astype(np.float32)
    return PoseCandidate(box=np.array([300, 200, 980, 720], np.float32), kpts=k, scores=s,
                         det_score=0.8, from_detection=from_detection)


def test_one_close_person_stays_one_player():
    rng = np.random.default_rng(0)
    tracker = PlayerTracker(make_cfg(), W, H)
    image = np.full((H, W, 3), 120, np.uint8)
    for i in range(120):
        cands = [closeup_pose(rng, 150, from_detection=(i % 10 == 0)),
                 closeup_pose(rng, 400, from_detection=False),
                 closeup_pose(rng, 800, from_detection=(i % 10 == 0))]
        tracker.update(cands, image, i / 60)
    assert list(tracker.tracks) == [1]


def test_boxes_stay_inside_the_image():
    rng = np.random.default_rng(1)
    tracker = PlayerTracker(make_cfg(), W, H)
    image = np.full((H, W, 3), 120, np.uint8)
    for i in range(10):
        tracker.update([closeup_pose(rng, 600, True)], image, i / 60)
    box = tracker.tracks[1].box
    assert box[0] >= 0 and box[1] >= 0 and box[2] <= W and box[3] <= H
