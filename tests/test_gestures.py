from conftest import make_engine, run_frames, scenario_period

from motionplay.gestures.engine import GestureEngine


def _events(scenario, loops=1.0, one_stage=False):
    engine, src = make_engine(scenario, one_stage=one_stage)
    out = []
    run_frames(engine, src, scenario_period(src) * loops,
               lambda r: out.extend((r.frame.meta["t_scene"], g) for g in r.gestures))
    return out


def test_demo_script_produces_exactly_the_scripted_gestures():
    """Each scripted move fires once, with the right arm/direction, and
    nothing else fires (no false positives from T-pose, walking, etc.)."""
    got = [(g.gesture, g.phase, g.side, g.direction) for _, g in _events("demo")]
    assert got == [
        ("jump", "trigger", None, None),
        ("squat", "start", None, None),
        ("squat", "end", None, None),
        ("punch", "trigger", "left", None),
        ("punch", "trigger", "right", None),
        ("swipe", "trigger", "right", "right"),
        ("swipe", "trigger", "left", "left"),
        ("arm_raise", "start", "both", None),
        ("arm_raise", "end", "both", None),
        ("lean", "start", None, "left"),
        ("lean", "end", None, "left"),
        ("lean", "start", None, "right"),
        ("lean", "end", None, "right"),
    ]


def test_gestures_fire_soon_after_the_move_starts():
    events = {(g.gesture, g.side): t for t, g in _events("demo")}
    # Scripted start times in the demo timeline (see capture/synthetic.py).
    assert 4.30 < events[("jump", None)] < 4.30 + 0.15
    assert 7.90 < events[("punch", "left")] < 7.90 + 0.35        # guard 0.15 s, extend 0.12 s


def test_two_players_get_their_own_events():
    events = _events("duo")
    players = {g.player_id for _, g in events}
    assert players == {1, 2}
    kinds = {(g.player_id, g.gesture) for _, g in events}
    assert any(k[1] == "jump" for k in kinds) and any(k[1] == "squat" for k in kinds)


def test_jump_has_refractory_and_needs_quick_rise():
    import numpy as np
    from motionplay.skeleton import J

    ge = GestureEngine(1)
    base = np.zeros((26, 2), np.float32)
    base[J.NECK] = (500, 300)
    base[J.HIP] = (500, 500)              # torso = 200 px
    base[J.LEFT_ANKLE] = (480, 900)
    base[J.RIGHT_ANKLE] = (520, 900)
    conf = np.ones(26, np.float32)
    for i in range(30):
        ge.update(base, conf, i / 60)
    # Slow rise over 1 s (like stepping back from the camera): no jump.
    fired = []
    for i in range(60):
        k = base.copy()
        k[:, 1] -= 60 * i / 60
        fired += ge.update(k, conf, 0.5 + i / 60)
    assert not [e for e in fired if e.gesture == "jump"]
