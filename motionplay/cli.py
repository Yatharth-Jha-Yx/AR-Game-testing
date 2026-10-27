"""Command line: ``motionplay run | models | bench | cameras | config``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Config, data_dir


def _config(args: argparse.Namespace) -> Config:
    cfg = Config.load()
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        cfg.set(key, value)
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    cam, vis = cfg.camera, cfg.vision
    if args.source:
        cam.source = args.source
    if args.width:
        cam.width = args.width
    if args.height:
        cam.height = args.height
    if args.fps:
        cam.fps = args.fps
    if args.no_mirror:
        cam.mirror = False
    if args.backend:
        vis.backend = args.backend
    if args.tier:
        vis.tier = args.tier
    if args.segmentation:
        vis.segmentation = True
    if args.port:
        cfg.server.port = args.port
    if args.host:
        cfg.server.host = args.host
    if args.no_video:
        cfg.privacy.video_enabled = False
    if cam.source.startswith("synthetic") and not args.width:
        cam.width, cam.height = 1280, 720      # cheaper to render; same maths

    from .capture.sources import open_source
    from .engine import Engine
    from .server import run_server
    from .vision.factory import build_backend, build_segmenter, tier_file

    if vis.backend in ("auto", "rtmpose") and vis.tier == "auto" and not cam.source.startswith("synthetic") \
            and not tier_file().exists() and not args.skip_bench:
        print("First launch: benchmarking models to pick a quality tier (once)...")
        from . import bench
        try:
            bench.run(cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"benchmark failed ({exc}); continuing with defaults")
    source = open_source(cam)
    backend = build_backend(cfg, source.name)
    engine = Engine(cfg, source, backend, build_segmenter(cfg))
    if args.record_keypoints:
        from .recording import KeypointRecorder
        engine.recorder = KeypointRecorder(args.record_keypoints, source.width, source.height, cam.mirror)
        print(f"recording joint positions (no video) to {args.record_keypoints}")
    run_server(engine, cfg, open_browser=args.open)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .vision import models
    from .vision.factory import TIERS

    if args.action == "list":
        for name, spec in models.REGISTRY.items():
            mark = "x" if models.is_downloaded(name) else " "
            print(f"[{mark}] {name:22s} {spec.kind:15s} {spec.license:10s} {spec.url}")
        print(f"\nstored in {models.models_dir()}")
        return 0
    names = list(args.names)
    if args.tier:
        names += [n for k, n in TIERS[args.tier].items() if k in ("detector", "pose")]
    if args.all:
        names = list(models.REGISTRY)
    if not names:
        names = [TIERS["full"]["detector"], TIERS["full"]["pose"], TIERS["lite"]["pose"],
                 "mediapipe-pose-lite", "selfie-landscape"]
    failed = 0
    for name in names:
        try:
            path, _ = models.ensure(name)
            print(f"ok  {name} -> {path}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERR {name}: {exc}")
    return 1 if failed else 0


def cmd_bench(args: argparse.Namespace) -> int:
    from . import bench

    return 0 if bench.run(_config(args), save=not args.no_save) else 1


def cmd_replay(args: argparse.Namespace) -> int:
    from .recording import parse_thresholds, replay

    out = replay(args.file, parse_thresholds(args.threshold))
    for e in out["events"]:
        extra = " ".join(f"{k}={e[k]}" for k in ("side", "direction") if e[k])
        print(f"{e['t']:9.3f}  P{e['player_id']}  {e['gesture']:9s} {e['phase']:7s} {extra}")
    print(f"\n{len(out['events'])} events with these thresholds "
          f"({len(out['recorded'])} were detected live when recording)")
    return 0


def cmd_cameras(args: argparse.Namespace) -> int:
    from .capture.sources import probe_cameras

    cams = probe_cameras()
    if not cams:
        print("no cameras found")
        return 1
    for cam in cams:
        if "error" in cam:
            print(f"camera:{cam['index']}  {cam['error']}")
        else:
            print(f"camera:{cam['index']}  best mode: {cam['best']}  {'OK' if cam['ok'] else 'SLOW'}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = Config.load()
    if args.action == "set":
        cfg.set(args.key, args.value)
        print(f"saved {cfg.save()}")
    elif args.action == "reset":
        print(f"saved {Config().save()}")
    else:
        print(json.dumps(cfg.to_dict(), indent=2))
        print(f"\n(file: {data_dir() / 'config.json'})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="motionplay", description="Webcam motion console engine")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="start the tracker + web server")
    r.add_argument("--source", help="camera:0 | video:clip.mp4 | image:photo.jpg | synthetic[:demo|fruit|duo|cross|leave_return|quad]")
    r.add_argument("--backend", choices=["auto", "rtmpose", "rtmo", "mediapipe", "synthetic"])
    r.add_argument("--tier", choices=["auto", "full", "balanced", "lite"])
    r.add_argument("--width", type=int)
    r.add_argument("--height", type=int)
    r.add_argument("--fps", type=int)
    r.add_argument("--no-mirror", action="store_true")
    r.add_argument("--segmentation", action="store_true", help="person mask for AR occlusion (MediaPipe)")
    r.add_argument("--no-video", action="store_true", help="parental mode: never send camera video to games")
    r.add_argument("--host")
    r.add_argument("--port", type=int)
    r.add_argument("--open", action="store_true", help="open the launcher in a browser")
    r.add_argument("--skip-bench", action="store_true")
    r.add_argument("--record-keypoints", metavar="FILE.jsonl",
                   help="save joint positions (never video) for tuning gestures with `motionplay replay`")
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any config value, e.g. vision.detect_interval=8")
    r.set_defaults(fn=cmd_run)

    m = sub.add_parser("models", help="list / download model weights")
    m.add_argument("action", choices=["list", "download"])
    m.add_argument("names", nargs="*")
    m.add_argument("--tier", choices=["full", "balanced", "lite"])
    m.add_argument("--all", action="store_true")
    m.set_defaults(fn=cmd_models)

    b = sub.add_parser("bench", help="time the models on this PC and choose a tier")
    b.add_argument("--no-save", action="store_true")
    b.add_argument("--set", action="append", metavar="KEY=VALUE")
    b.set_defaults(fn=cmd_bench)

    rp = sub.add_parser("replay", help="re-run gesture detection on a keypoint recording")
    rp.add_argument("file")
    rp.add_argument("--threshold", action="append", metavar="NAME=VALUE",
                    help="override a gesture threshold, e.g. swipe_speed=5 (see gestures/engine.py)")
    rp.set_defaults(fn=cmd_replay)

    c = sub.add_parser("cameras", help="list cameras and which modes they accept")
    c.set_defaults(fn=cmd_cameras)

    cf = sub.add_parser("config", help="show or change saved settings")
    cf.add_argument("action", choices=["show", "set", "reset"], nargs="?", default="show")
    cf.add_argument("key", nargs="?")
    cf.add_argument("value", nargs="?")
    cf.set_defaults(fn=cmd_config)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
