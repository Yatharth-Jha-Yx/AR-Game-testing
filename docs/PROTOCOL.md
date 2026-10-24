# MotionPlay WebSocket protocol (version 1)

Endpoint: `ws://127.0.0.1:8765/ws`. Games never open the camera. They only
see what the engine sends here. The JavaScript SDK (`/sdk/motionplay.js`)
wraps all of this, so most games never read these messages directly.

## Coordinates

* **Screen space** (`joints`, `bbox`): 0..1 of the camera image, already
  **mirrored** when `camera.mirrored` is true, so a game can draw it straight
  onto a canvas that shows the mirrored preview. Joint confidence `c` is
  0..1; `c = 0` means hidden (out of view for more than 5 frames).
* **Play space** (`play`): origin at the mid-hip, **1 unit = torso length**
  (neck to mid-hip), y up, x towards screen-right. A child and an adult
  doing the same move give the same numbers.
* Joint order is Halpe26 (sent in `hello.skeleton.keypoints`):
  nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, then
  head (top), neck, hip (mid), big toes, small toes, heels.

## Server to game (JSON text)

`hello`, sent once on connect:
```json
{"type": "hello", "protocol": 1, "engine": "motionplay", "version": "0.1.0",
 "skeleton": {"name": "halpe26", "keypoints": ["nose", "..."], "edges": [[15, 13], "..."]},
 "camera": {"width": 1920, "height": 1080, "fps": 60, "mirrored": true},
 "backend": "rtmpose-m-halpe26 + rtmdet-nano (full)",
 "settings": {"video_enabled": true, "skeleton_only": false, "cooldown_minutes": 20, "segmentation": false}}
```

`frame`, every processed camera frame (dropped for slow clients, never queued):
```json
{"type": "frame", "seq": 1234, "t": 81.2345, "latency_ms": 9.8,
 "players": [{
   "id": 1, "confidence": 0.91, "bbox": [0.41, 0.08, 0.6, 0.97],
   "joints": [0.51, 0.13, 0.98, "... 26 x (x, y, c)"],
   "play": [0.0, 2.6, "... 26 x (x, y)"],
   "depth_m": 2.43, "depth_calibrated": true, "position_m": [-0.12, 2.43],
   "in_bounds": true, "bounds_hint": null, "reach_torso": 1.55, "predicted": false}]}
```
* `position_m`: [sideways from the camera axis (screen-right +), distance].
* `bounds_hint`: `step_back`, `step_forward`, `move_left`, `move_right` or null.
* `reach_torso`: half arm span in torso lengths (from calibration; 1.6 before).
* `predicted`: true when pose was skipped this frame (3+ players over budget)
  and joints were extrapolated.

`gesture`, never dropped:
```json
{"type": "gesture", "player_id": 1, "gesture": "swipe", "phase": "trigger",
 "strength": 0.8, "side": "right", "direction": "left", "t": 81.2345}
```

| gesture | phases | side | direction |
| --- | --- | --- | --- |
| `jump` | trigger | - | - |
| `squat` | start, end | - | - |
| `punch` | trigger | arm: left/right | - |
| `swipe` | trigger | hand: left/right | screen: left/right |
| `arm_raise` | start, end | left/right/both | - |
| `lean` | start, end | - | screen: left/right |

`player`: `{"type": "player", "event": "joined" | "lost" | "rejoined" | "left", "player_id": 2}`.
"lost" means show *Player 2, step back in*; after 2 s without them, "left" frees the slot.

`calibration`: progress while calibrating:
```json
{"type": "calibration", "player_id": 1, "state": "placement | tpose | done | failed | cancelled",
 "progress": 0.4, "message": "Step back until your feet are visible",
 "checks": {"full_body": false, "issues": ["feet"], "lighting": "ok"},
 "result": {"height_m": 1.69, "wingspan_m": 1.88, "distance_m": 2.0, "...": "..."}}
```

`camera_lock`: `{"type": "camera_lock", "report": {"supported": true, "exposure_seconds": 0.0078, ...}}`

`status`: once per second: fps, latency (mean/p95), per-stage timings, players.

`settings`: after a settings change: `{"type": "settings", "settings": {...}}`.

`pong`: reply to `ping`: `{"type": "pong", "t": <your t>, "server_t": ...}`.

## Server to game (binary)

16-byte little-endian header, then the payload:

| bytes | field |
| --- | --- |
| 0 | kind: 1 = JPEG preview, 2 = PNG person mask |
| 1-3 | padding |
| 4-7 | uint32 frame `seq` |
| 8-15 | float64 capture time (s) |

Video is only sent to clients that subscribed, and never when
`privacy.video_enabled` is false (parental switch) or `skeleton_only` is on.

## Game to server

```json
{"type": "subscribe", "frames": true, "video": true, "mask": false}
{"type": "calibrate", "action": "start", "player_id": null, "distance_m": 2.0}
{"type": "calibrate", "action": "cancel"}
{"type": "camera", "action": "lock"}
{"type": "camera", "action": "unlock"}
{"type": "settings", "skeleton_only": true}
{"type": "ping", "t": 1234.5}
```
A game can only change display preferences. The parental video switch lives
in `config.json` / the `--no-video` flag, so a child can't turn video back on
from the TV.
