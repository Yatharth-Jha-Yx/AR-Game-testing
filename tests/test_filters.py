import numpy as np

from motionplay.config import TrackingConfig
from motionplay.tracking.filters import BoxKalman, OneEuroFilter
from motionplay.tracking.smoothing import JointSmoother


def test_one_euro_removes_jitter_when_still(rng):
    f = OneEuroFilter((1,), min_cutoff=1.0, beta=0.01)
    noisy = 500 + rng.normal(0, 3, 300)
    out = np.array([f(np.array([x]), i / 60)[0] for i, x in enumerate(noisy)])
    assert np.std(out[60:]) < np.std(noisy[60:]) / 3


def test_one_euro_keeps_up_with_fast_motion():
    """beta makes the cutoff rise with speed, so a fast move has little lag."""
    slow = OneEuroFilter((1,), min_cutoff=1.0, beta=0.0)
    fast = OneEuroFilter((1,), min_cutoff=1.0, beta=0.01)
    lag_slow = lag_fast = 0.0
    for i in range(60):
        t = i / 60
        x = np.array([2000.0 * t])            # 2000 px/s, like a swinging wrist
        lag_slow = x[0] - slow(x, t)[0]
        lag_fast = x[0] - fast(x, t)[0]
    assert lag_fast < lag_slow / 3


def test_one_euro_mask_holds_state():
    f = OneEuroFilter((2,), min_cutoff=1.0, beta=0.0)
    f(np.array([0.0, 0.0]), 0.0)
    out = f(np.array([100.0, 100.0]), 1 / 60, mask=np.array([True, False]))
    assert out[0] > 0 and out[1] == 0.0


def test_smoother_gates_holds_and_hides():
    cfg = TrackingConfig()
    sm = JointSmoother(cfg, frame_height=1080)
    k = np.full((26, 2), 100.0, np.float32)
    s = np.full(26, 0.9, np.float32)
    sm.update(k, s, 0.0)
    s_bad = s.copy()
    s_bad[9] = 0.1                                   # wrist drops out
    moved = k + 50
    for i in range(cfg.hold_frames):
        _, conf = sm.update(moved, s_bad, (i + 1) / 60)
        assert conf[9] > 0                            # still held
    assert np.allclose(sm.smoothed[9], 100.0)         # held where it was
    _, conf = sm.update(moved, s_bad, (cfg.hold_frames + 1) / 60)
    assert conf[9] == 0                               # then hidden


def test_prediction_only_for_confident_joints():
    cfg = TrackingConfig()
    sm = JointSmoother(cfg, frame_height=1080)
    s = np.full(26, 0.9, np.float32)
    s[0] = 0.4                                         # below predict_min_conf
    for i in range(30):
        sm.update(np.full((26, 2), 10.0 * i, np.float32), s, i / 60)
    ahead = sm.predicted(1 / 60)
    assert ahead[5, 0] > sm.smoothed[5, 0]             # moving joint pushed ahead
    assert ahead[0, 0] == sm.smoothed[0, 0]            # shaky joint not extrapolated


def test_box_kalman_predicts_motion():
    kf = BoxKalman(np.array([0, 0, 100, 200.0]), 0.0)
    for i in range(1, 20):
        kf.update(np.array([10 * i, 0, 100 + 10 * i, 200.0]), i / 60)
    pred = kf.predict_box(20 / 60)
    assert abs(pred[0] - 200) < 15
