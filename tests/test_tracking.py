import numpy as np
import pytest
from conftest import make_cfg, make_engine, run_frames, scenario_period

from motionplay.skeleton import J
from motionplay.tracking.tracker import oks


def _actor_for(track_kpts, track_scores, gt):
    """Which synthetic actor does this track's latest measurement belong to?
    Compared on confident joints only (hidden joints are held, not measured)."""
    ok = track_scores >= 0.5
    if ok.sum() < 3:
        return None
    d = [float(np.mean(np.linalg.norm(g["kpts"][ok] - track_kpts[ok], axis=1))) for g in gt]
    return gt[int(np.argmin(d))]["pid"] if d else None


@pytest.mark.parametrize("scenario", ["cross", "duo", "quad"])
@pytest.mark.parametrize("one_stage", [False, True])
def test_ids_never_swap(scenario, one_stage):
    engine, src = make_engine(scenario, one_stage=one_stage)
    mapping, events = {}, []

    def check(r):
        events.extend(e["event"] for e in r.player_events)
        for tr in engine.tracker.tracks.values():
            if not tr.matched_now:
                continue
            actor = _actor_for(tr.kpts, tr.scores, r.frame.meta["gt"])
            if actor is None:
                continue
            assert mapping.setdefault(tr.id, actor) == actor, \
                f"player {tr.id} switched from actor {mapping[tr.id]} to {actor} at {r.frame.meta['t_scene']:.2f}s"

    run_frames(engine, src, scenario_period(src) * 2, check)
    n_actors = len(src.actors)
    assert len(mapping) == n_actors
    assert events.count("joined") == n_actors
    assert "left" not in events


def test_leave_and_return_reattaches_within_window_then_frees_slot():
    engine, src = make_engine("leave_return")
    log = []
    run_frames(engine, src, scenario_period(src),
               lambda r: log.extend((round(r.frame.meta["t_scene"], 2), e["event"], e["player_id"])
                                    for e in r.player_events))
    kinds = [e[1] for e in log]
    # Hidden 1.2 s (< 2 s): same ID comes back. Hidden 3 s: slot freed, rejoins as new.
    assert kinds[:3] == ["joined", "lost", "rejoined"]
    assert kinds[3:6] == ["lost", "left", "joined"]
    assert all(e[2] == 1 for e in log)


def test_extra_people_beyond_max_players_are_ignored():
    cfg = make_cfg()
    cfg.vision.max_players = 2
    engine, src = make_engine("quad", cfg=cfg)
    run_frames(engine, src, 1.0)
    assert sorted(engine.tracker.tracks) == [1, 2]


def test_oks_identical_is_one_and_far_is_zero():
    k = np.random.default_rng(1).uniform(0, 200, (26, 2)).astype(np.float32)
    s = np.ones(26, np.float32)
    assert oks(k, s, k, s, area=200 * 200) == pytest.approx(1.0)
    assert oks(k, s, k + 500, s, area=200 * 200) < 0.01


def test_detector_runs_every_n_frames_when_stable():
    engine, src = make_engine("demo")
    calls = []
    original = engine.backend.detect
    engine.backend.detect = lambda f: calls.append(1) or original(f)
    run_frames(engine, src, 2.0)
    # 120 frames, detect_interval 10 -> about 12 detector runs (+ a few at start-up).
    assert 10 <= len(calls) <= 20
    assert engine.tracker.tracks[1].kpts[J.HIP][0] > 0
