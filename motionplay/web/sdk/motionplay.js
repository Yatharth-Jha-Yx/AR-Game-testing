// MotionPlay browser SDK.
//
//   import { MotionPlay } from "/sdk/motionplay.js";
//   const mp = new MotionPlay({ video: true });
//   mp.on("gesture", (g) => { if (g.gesture === "jump") ... });
//   mp.on("frame", () => { for (const body of mp.bodies()) body.joint("right_wrist") });
//
// Games never touch the camera: they receive normalised, mirrored body data
// and discrete gesture events from the local engine over a WebSocket.
// Coordinates are 0..1 of the camera image; use mp.view(w, h) to map them to
// a canvas drawn with the same "cover" fit as the video.

const BIN_VIDEO_JPEG = 1;
const BIN_MASK_PNG = 2;
const BIN_HEADER = 16;

export class Body {
  constructor(raw, index) {
    this.raw = raw;
    this._index = index;
    this.id = raw.id;
    this.confidence = raw.confidence;
    this.depth = raw.depth_m;
    this.depthCalibrated = raw.depth_calibrated;
    this.position = { x: raw.position_m[0], z: raw.position_m[1] };
    this.inBounds = raw.in_bounds;
    this.boundsHint = raw.bounds_hint;
    this.reach = raw.reach_torso;
    this.bbox = raw.bbox;
    this.predicted = raw.predicted;
  }
  /** {x, y, c} in 0..1 screen coordinates (mirrored), c = confidence. */
  joint(name) {
    const i = typeof name === "number" ? name : this._index[name];
    const j = this.raw.joints;
    return { x: j[i * 3], y: j[i * 3 + 1], c: j[i * 3 + 2] };
  }
  visible(name, thr = 0.3) {
    return this.joint(name).c >= thr;
  }
  /** Play-space {x, y}: origin mid-hip, 1 unit = torso length, y up, x screen-right. */
  play(name) {
    const i = typeof name === "number" ? name : this._index[name];
    return { x: this.raw.play[i * 2], y: this.raw.play[i * 2 + 1] };
  }
  /** Hand centre: wrist pushed ~30% of the forearm further out. */
  hand(side) {
    const w = this.joint(`${side}_wrist`);
    const e = this.joint(`${side}_elbow`);
    return { x: w.x + (w.x - e.x) * 0.3, y: w.y + (w.y - e.y) * 0.3, c: Math.min(w.c, e.c) };
  }
}

export class MotionPlay extends EventTarget {
  constructor(opts = {}) {
    super();
    const params = new URLSearchParams(location.search);
    this.url = opts.url || params.get("engine") ||
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host || "127.0.0.1:8765"}/ws`;
    this.wantVideo = opts.video ?? false;
    this.wantMask = opts.mask ?? false;
    this.mouseFallback = opts.mouseFallback ?? true;
    this.forceMouse = params.has("mouse");
    this.connected = false;
    this.mode = "connecting"; // connecting | engine | mouse | offline
    this.hello = null;
    this.settings = {};
    this.camera = { width: 16, height: 9, mirrored: true };
    this.index = {};
    this.edges = [];
    this.frame = null;
    this.status = null;
    this.videoBitmap = null;
    this.maskBitmap = null;
    this._bodies = [];
    this._retry = 500;
    this._mouse = null;
    if (this.forceMouse) this._startMouse();
    else this.connect();
  }

  on(type, fn) {
    const handler = (e) => fn(e.detail);
    this.addEventListener(type, handler);
    return () => this.removeEventListener(type, handler);
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  connect() {
    let opened = false;
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      opened = true;
      this.connected = true;
      this.mode = "engine";
      this._retry = 500;
      if (this._mouse) this._stopMouse();
      this.subscribe({ video: this.wantVideo, mask: this.wantMask });
      this._emit("connection", { connected: true, mode: this.mode });
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") this._onJson(JSON.parse(ev.data));
      else this._onBinary(ev.data);
    };
    ws.onclose = () => {
      this.connected = false;
      if (!opened && this.mouseFallback && !this._mouse) this._startMouse();
      else if (!this._mouse) this.mode = "offline";
      this._emit("connection", { connected: false, mode: this.mode });
      setTimeout(() => this.connect(), this._retry);
      this._retry = Math.min(this._retry * 2, 5000);
    };
  }

  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  subscribe({ video, mask, frames = true } = {}) {
    if (video !== undefined) this.wantVideo = video;
    if (mask !== undefined) this.wantMask = mask;
    this.send({ type: "subscribe", frames, video: this.wantVideo, mask: this.wantMask });
  }

  calibrate(playerId = null, distanceM = null) {
    this.send({ type: "calibrate", action: "start", player_id: playerId, distance_m: distanceM });
  }
  cancelCalibration() { this.send({ type: "calibrate", action: "cancel" }); }
  lockCamera() { this.send({ type: "camera", action: "lock" }); }
  unlockCamera() { this.send({ type: "camera", action: "unlock" }); }
  setSkeletonOnly(on) { this.send({ type: "settings", skeleton_only: !!on }); }

  /** Should games draw the camera video? (privacy + parental switches) */
  get videoAllowed() {
    return this.mode === "engine" && this.settings.video_enabled !== false && !this.settings.skeleton_only;
  }

  bodies() { return this._bodies; }
  body(id) { return this._bodies.find((b) => b.id === id) || null; }

  _onJson(msg) {
    switch (msg.type) {
      case "hello":
        this.hello = msg;
        this.camera = msg.camera;
        this.settings = msg.settings || {};
        this.edges = msg.skeleton.edges;
        this.index = Object.fromEntries(msg.skeleton.keypoints.map((n, i) => [n, i]));
        this._emit("hello", msg);
        this._emit("settings", this.settings);
        break;
      case "frame":
        this.frame = msg;
        this._bodies = msg.players.map((p) => new Body(p, this.index));
        this._emit("frame", msg);
        break;
      case "settings":
        this.settings = msg.settings;
        this._emit("settings", this.settings);
        break;
      case "status":
        this.status = msg;
        this._emit("status", msg);
        break;
      default:
        this._emit(msg.type, msg); // gesture, player, calibration, camera_lock, pong, error
    }
  }

  async _onBinary(buf) {
    const kind = new DataView(buf).getUint8(0);
    const blob = new Blob([buf.slice(BIN_HEADER)], { type: kind === BIN_MASK_PNG ? "image/png" : "image/jpeg" });
    try {
      const bmp = await createImageBitmap(blob);
      if (kind === BIN_VIDEO_JPEG) {
        if (this.videoBitmap) this.videoBitmap.close();
        this.videoBitmap = bmp;
        this._emit("video", bmp);
      } else if (kind === BIN_MASK_PNG) {
        if (this.maskBitmap) this.maskBitmap.close();
        this.maskBitmap = bmp;
        this._emit("mask", bmp);
      }
    } catch (e) { /* a dropped/corrupt frame is fine to skip */ }
  }

  /** Mapping between camera-normalised coordinates and a w x h canvas
   *  using "cover" fit (fills the screen, crops the overflow). */
  view(w, h) {
    const cw = this.camera.width, ch = this.camera.height;
    const s = Math.max(w / cw, h / ch);
    const dw = cw * s, dh = ch * s;
    const ox = (w - dw) / 2, oy = (h - dh) / 2;
    return {
      scale: s, width: dw, height: dh, x: ox, y: oy,
      toScreen: (p) => ({ x: ox + p.x * dw, y: oy + p.y * dh, c: p.c }),
      drawVideo: (ctx, bmp = this.videoBitmap) => { if (bmp) ctx.drawImage(bmp, ox, oy, dw, dh); },
    };
  }

  // --- mouse / keyboard fallback (no engine running, or ?mouse in the URL) ---
  _startMouse() {
    this.mode = "mouse";
    const names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
      "left_ankle", "right_ankle", "head", "neck", "hip", "left_big_toe", "right_big_toe", "left_small_toe",
      "right_small_toe", "left_heel", "right_heel"];
    this.index = Object.fromEntries(names.map((n, i) => [n, i]));
    this.camera = { width: 16, height: 9, mirrored: true };
    this.settings = { video_enabled: false, skeleton_only: true, cooldown_minutes: 20 };
    this.edges = [[15, 13], [13, 11], [11, 19], [16, 14], [14, 12], [12, 19], [17, 18], [18, 19], [18, 5], [5, 7],
      [7, 9], [18, 6], [6, 8], [8, 10]];
    const state = { x: 0.65, y: 0.45, lx: 0.35, ly: 0.55, t: 0 };
    const onMove = (e) => {
      const r = document.documentElement.getBoundingClientRect();
      state.x = e.clientX / r.width;
      state.y = e.clientY / r.height;
    };
    const keyMap = { " ": ["jump", "trigger"], j: ["jump", "trigger"], p: ["punch", "trigger", "right"],
      ArrowLeft: ["swipe", "trigger", "right", "left"], ArrowRight: ["swipe", "trigger", "right", "right"],
      r: ["arm_raise", "start", "both"], Enter: ["arm_raise", "start", "both"], s: ["squat", "start"] };
    const onKey = (e) => {
      const k = keyMap[e.key];
      if (!k) return;
      this._emit("gesture", { type: "gesture", player_id: 1, gesture: k[0], phase: k[1], side: k[2] || null,
        direction: k[3] || null, strength: 1, t: performance.now() / 1000 });
    };
    addEventListener("pointermove", onMove);
    addEventListener("keydown", onKey);
    const tick = () => {
      state.t += 1 / 60;
      const sway = Math.sin(state.t * 1.3) * 0.01;
      const j = new Array(26 * 3).fill(0);
      const set = (n, x, y) => { const i = this.index[n]; j[i * 3] = x; j[i * 3 + 1] = y; j[i * 3 + 2] = 0.95; };
      const cx = 0.5 + sway;
      set("head", cx, 0.12); set("nose", cx, 0.18); set("left_eye", cx - 0.01, 0.17); set("right_eye", cx + 0.01, 0.17);
      set("left_ear", cx - 0.02, 0.18); set("right_ear", cx + 0.02, 0.18); set("neck", cx, 0.27);
      set("left_shoulder", cx - 0.06, 0.29); set("right_shoulder", cx + 0.06, 0.29);
      set("hip", cx, 0.55); set("left_hip", cx - 0.035, 0.55); set("right_hip", cx + 0.035, 0.55);
      set("left_knee", cx - 0.04, 0.72); set("right_knee", cx + 0.04, 0.72);
      set("left_ankle", cx - 0.04, 0.9); set("right_ankle", cx + 0.04, 0.9);
      // Mouse = right hand (screen right, mirrored view). Left hand trails the opposite side.
      set("right_wrist", state.x, state.y);
      set("right_elbow", (cx + 0.06 + state.x) / 2 + 0.01, (0.29 + state.y) / 2 + 0.03);
      state.lx += ((1 - state.x) - state.lx) * 0.05;
      state.ly += (Math.max(0.35, state.y) - state.ly) * 0.05;
      set("left_wrist", state.lx, state.ly);
      set("left_elbow", (cx - 0.06 + state.lx) / 2 - 0.01, (0.29 + state.ly) / 2 + 0.03);
      const play = new Array(52).fill(0);
      const raw = { id: 1, confidence: 0.95, bbox: [cx - 0.1, 0.1, cx + 0.1, 0.92], joints: j, play,
        depth_m: 2.5, depth_calibrated: false, position_m: [0, 2.5], in_bounds: true, bounds_hint: null,
        reach_torso: 1.6, predicted: false };
      this.frame = { type: "frame", seq: 0, t: state.t, latency_ms: 0, players: [raw] };
      this._bodies = [new Body(raw, this.index)];
      this._emit("frame", this.frame);
    };
    this._mouse = { timer: setInterval(tick, 1000 / 60), onMove, onKey };
    this._emit("hello", { type: "hello", mouse: true, camera: this.camera, settings: this.settings });
    this._emit("settings", this.settings);
    this._emit("connection", { connected: false, mode: "mouse" });
  }

  _stopMouse() {
    clearInterval(this._mouse.timer);
    removeEventListener("pointermove", this._mouse.onMove);
    removeEventListener("keydown", this._mouse.onKey);
    this._mouse = null;
  }
}

export const PLAYER_COLORS = ["#ffd166", "#4cc9f0", "#f72585", "#80ed99"];
export function playerColor(id) { return PLAYER_COLORS[(id - 1) % PLAYER_COLORS.length]; }

/** Draw a body as a friendly avatar (used in skeleton-only mode). */
export function drawAvatar(ctx, mp, body, view, color = playerColor(body.id)) {
  const P = (n) => view.toScreen(body.joint(n));
  const ok = (n) => body.joint(n).c >= 0.3;
  const torsoPx = Math.hypot(P("neck").x - P("hip").x, P("neck").y - P("hip").y) || 50;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.85;
  const limb = (a, b, wScale) => {
    if (!ok(a) || !ok(b)) return;
    const pa = P(a), pb = P(b);
    ctx.lineWidth = torsoPx * wScale;
    ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
  };
  limb("neck", "hip", 0.55);
  limb("left_shoulder", "right_shoulder", 0.3);
  limb("left_hip", "right_hip", 0.3);
  for (const s of ["left", "right"]) {
    limb(`${s}_shoulder`, `${s}_elbow`, 0.2); limb(`${s}_elbow`, `${s}_wrist`, 0.17);
    limb(`${s}_hip`, `${s}_knee`, 0.24); limb(`${s}_knee`, `${s}_ankle`, 0.2);
  }
  if (ok("head") && ok("neck")) {
    const h = P("head"), n = P("neck");
    const r = Math.hypot(h.x - n.x, h.y - n.y) * 0.55;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc((h.x + n.x) / 2, (h.y + n.y) / 2, r, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}

/** Thin debug skeleton. */
export function drawSkeleton(ctx, mp, body, view, color = playerColor(body.id)) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 3;
  for (const [a, b] of mp.edges) {
    const ja = body.joint(a), jb = body.joint(b);
    if (ja.c < 0.3 || jb.c < 0.3) continue;
    const pa = view.toScreen(ja), pb = view.toScreen(jb);
    ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
  }
  for (let i = 0; i < 26; i++) {
    const j = body.joint(i);
    if (j.c < 0.3) continue;
    const p = view.toScreen(j);
    ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}
