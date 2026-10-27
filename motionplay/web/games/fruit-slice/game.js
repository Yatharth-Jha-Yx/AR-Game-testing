// Fruit Slice: swing your hands fast to slice fruit, avoid the bombs.
// Input primitive exercised: hand position + hand speed (from wrist/elbow).
// Start/restart: raise both hands (arm_raise gesture), or Enter / click.

import { MotionPlay, drawAvatar, playerColor } from "/sdk/motionplay.js";
import { installOverlay } from "/sdk/overlay.js";

const mp = new MotionPlay({ video: true });
const overlay = installOverlay(mp);
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const ROUND_S = 60;
const LIVES = 3;
const GRAVITY = 1.25;        // screen heights / s^2
const BLADE_SPEED = 1.1;     // screen heights / s needed to cut
const TRAIL_S = 0.14;

const FRUITS = [
  { name: "apple", r: 0.048, skin: "#e63946", skin2: "#9d0208", flesh: "#fff3d6", juice: "#ff8fa3" },
  { name: "orange", r: 0.05, skin: "#fb8500", skin2: "#c45f00", flesh: "#ffb703", juice: "#ffb703" },
  { name: "lime", r: 0.042, skin: "#70e000", skin2: "#38b000", flesh: "#ccff33", juice: "#ccff33" },
  { name: "watermelon", r: 0.068, skin: "#2d6a4f", skin2: "#1b4332", flesh: "#ff4d6d", juice: "#ff4d6d", stripes: true },
  { name: "plum", r: 0.04, skin: "#7b2cbf", skin2: "#3c096c", flesh: "#e0aaff", juice: "#c77dff" },
];

let W = 0, H = 0, DPR = 1;
function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = innerWidth; H = innerHeight;
  canvas.width = W * DPR; canvas.height = H * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
addEventListener("resize", resize);
resize();

const game = {
  phase: "title", // title | countdown | play | over
  phaseT: 0,
  fruits: [], halves: [], particles: [], splats: [], popups: [],
  scores: new Map(), lives: LIVES, timeLeft: ROUND_S, spawnIn: 1, elapsed: 0,
  shake: 0, paused: false, combos: new Map(), debug: false,
};
const hands = new Map(); // "id-side" -> { trail: [{x, y, t}], speed, color, id }

// ---- audio (tiny synthesised effects, no files) -------------------------
let audio = null;
function initAudio() { if (!audio) try { audio = new AudioContext(); } catch (e) { audio = null; } }
addEventListener("pointerdown", initAudio);
addEventListener("keydown", initAudio);
function noise(dur, freq, gain) {
  if (!audio) return;
  const len = Math.floor(audio.sampleRate * dur);
  const buf = audio.createBuffer(1, len, audio.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len);
  const src = audio.createBufferSource(); src.buffer = buf;
  const f = audio.createBiquadFilter(); f.type = "bandpass"; f.frequency.value = freq;
  const g = audio.createGain(); g.gain.value = gain;
  src.connect(f).connect(g).connect(audio.destination); src.start();
}
const sfx = {
  slice: () => noise(0.12, 2400, 0.5),
  splat: () => noise(0.2, 500, 0.6),
  boom: () => noise(0.6, 120, 1.2),
};

// ---- game flow --------------------------------------------------------------
function setPhase(p) { game.phase = p; game.phaseT = 0; }
function startCountdown() {
  if (game.phase !== "title" && !(game.phase === "over" && game.phaseT > 2)) return;
  game.fruits = []; game.halves = []; game.particles = []; game.splats = []; game.popups = [];
  game.scores = new Map(); game.lives = LIVES; game.timeLeft = ROUND_S; game.spawnIn = 0.6; game.elapsed = 0;
  setPhase("countdown");
}
mp.on("gesture", (g) => {
  if (g.gesture === "arm_raise" && g.phase === "start" && g.side === "both" && !overlay.paused) startCountdown();
});
addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !overlay.paused) startCountdown();
  if (e.key === "d") game.debug = !game.debug;
});
canvas.addEventListener("click", () => { if (!overlay.paused) startCountdown(); });
mp.on("pause", () => { game.paused = true; });
mp.on("resume", () => { game.paused = false; });

// ---- spawning ---------------------------------------------------------------
function launch(bomb) {
  const kind = bomb ? null : FRUITS[Math.floor(Math.random() * FRUITS.length)];
  const r = (bomb ? 0.05 : kind.r) * H;
  const x = W * (0.18 + Math.random() * 0.64);
  const peak = H * (0.12 + Math.random() * 0.25);
  const vy = -Math.sqrt(2 * GRAVITY * H * (H + r - peak));
  const vx = ((W / 2 - x) / W) * H * 0.5 + (Math.random() - 0.5) * H * 0.25;
  game.fruits.push({ x, y: H + r, vx, vy, r, kind, bomb, rot: Math.random() * 6, spin: (Math.random() - 0.5) * 4 });
}
function spawnWave() {
  const n = 1 + Math.floor(Math.random() * Math.min(3, 1 + game.elapsed / 15));
  for (let i = 0; i < n; i++) launch(game.elapsed > 8 && Math.random() < 0.13);
  game.spawnIn = Math.max(0.55, 1.3 - game.elapsed * 0.012) + Math.random() * 0.4;
}

// ---- hands & slicing --------------------------------------------------------
function updateHands(now) {
  const view = mp.view(W, H);
  const seen = new Set();
  for (const body of mp.bodies()) {
    for (const side of ["left", "right"]) {
      const h = body.hand(side);
      if (h.c < 0.3) continue;
      const key = `${body.id}-${side}`;
      seen.add(key);
      const p = view.toScreen(h);
      let hand = hands.get(key);
      if (!hand) { hand = { trail: [], speed: 0, id: body.id }; hands.set(key, hand); }
      const last = hand.trail[hand.trail.length - 1];
      if (!last || last.x !== p.x || last.y !== p.y) hand.trail.push({ x: p.x, y: p.y, t: now });
      while (hand.trail.length > 2 && now - hand.trail[0].t > TRAIL_S) hand.trail.shift();
      if (hand.trail.length >= 2) {
        const a = hand.trail[hand.trail.length - 2], b = hand.trail[hand.trail.length - 1];
        const dt = Math.max(b.t - a.t, 1 / 240);
        const inst = Math.hypot(b.x - a.x, b.y - a.y) / dt / H;
        hand.speed = hand.speed * 0.4 + inst * 0.6;
      }
    }
  }
  for (const key of hands.keys()) if (!seen.has(key)) hands.delete(key);
}

function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function checkSlices(now) {
  if (game.phase !== "play") return;
  for (const hand of hands.values()) {
    if (hand.speed < BLADE_SPEED || hand.trail.length < 2) continue;
    const a = hand.trail[hand.trail.length - 2], b = hand.trail[hand.trail.length - 1];
    for (const f of game.fruits) {
      if (f.cut) continue;
      if (segDist(f.x, f.y, a.x, a.y, b.x, b.y) < f.r * 1.15) cut(f, hand, Math.atan2(b.y - a.y, b.x - a.x), now);
    }
  }
  game.fruits = game.fruits.filter((f) => !f.cut);
}

function cut(f, hand, angle, now) {
  f.cut = true;
  if (f.bomb) {
    game.lives -= 1; game.shake = 0.5; sfx.boom();
    burst(f.x, f.y, 40, ["#ffd166", "#ff6b35", "#ffffff"], 1.3);
    popup(f.x, f.y, "BOOM!", "#ff6b35");
    if (game.lives <= 0) setPhase("over");
    return;
  }
  sfx.slice(); sfx.splat();
  const nx = Math.cos(angle + Math.PI / 2), ny = Math.sin(angle + Math.PI / 2);
  for (const s of [-1, 1]) {
    game.halves.push({ x: f.x, y: f.y, vx: f.vx + nx * s * H * 0.25, vy: f.vy * 0.5 + ny * s * H * 0.25,
      r: f.r, kind: f.kind, angle, side: s, rot: 0, spin: s * 3, life: 2.5 });
  }
  burst(f.x, f.y, 16, [f.kind.juice, f.kind.flesh], 0.7);
  game.splats.push({ x: f.x, y: f.y, r: f.r * 1.8, color: f.kind.juice, life: 1.6, seed: Math.random() * 10 });
  const pts = (game.scores.get(hand.id) || 0) + 1;
  game.scores.set(hand.id, pts);
  // Combo: 3+ fruit by the same player within 0.35 s.
  const c = game.combos.get(hand.id) || { n: 0, t: 0 };
  c.n = now - c.t < 0.35 ? c.n + 1 : 1; c.t = now;
  game.combos.set(hand.id, c);
  if (c.n >= 3) {
    game.scores.set(hand.id, game.scores.get(hand.id) + 1);
    popup(f.x, f.y - f.r * 1.5, `Combo x${c.n}!`, playerColor(hand.id));
  } else popup(f.x, f.y, "+1", playerColor(hand.id));
}

function burst(x, y, n, colors, power) {
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2, s = (0.2 + Math.random() * 0.6) * H * power;
    game.particles.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s - H * 0.2,
      r: H * (0.004 + Math.random() * 0.008), color: colors[i % colors.length], life: 0.6 + Math.random() * 0.6 });
  }
}
function popup(x, y, text, color) { game.popups.push({ x, y, text, color, life: 0.9 }); }

// ---- simulation ---------------------------------------------------------------
function step(dt, now) {
  game.phaseT += dt;
  if (game.phase === "countdown" && game.phaseT >= 3) setPhase("play");
  if (game.phase === "play") {
    game.elapsed += dt;
    game.timeLeft -= dt;
    game.spawnIn -= dt;
    if (game.spawnIn <= 0) spawnWave();
    if (game.timeLeft <= 0) { game.timeLeft = 0; setPhase("over"); }
  }
  const g = GRAVITY * H;
  for (const f of game.fruits) { f.vy += g * dt; f.x += f.vx * dt; f.y += f.vy * dt; f.rot += f.spin * dt; }
  // Missed fruit costs a life (bombs don't).
  game.fruits = game.fruits.filter((f) => {
    if (f.y - f.r > H && f.vy > 0) {
      if (!f.bomb && game.phase === "play") {
        game.lives -= 1;
        popup(f.x, H - H * 0.06, "miss", "#ff6b6b");
        if (game.lives <= 0) setPhase("over");
      }
      return false;
    }
    return true;
  });
  for (const h of game.halves) { h.vy += g * dt; h.x += h.vx * dt; h.y += h.vy * dt; h.rot += h.spin * dt; h.life -= dt; }
  game.halves = game.halves.filter((h) => h.life > 0 && h.y - h.r < H * 1.2);
  for (const p of game.particles) { p.vy += g * dt; p.x += p.vx * dt; p.y += p.vy * dt; p.life -= dt; }
  game.particles = game.particles.filter((p) => p.life > 0);
  for (const s of game.splats) s.life -= dt;
  game.splats = game.splats.filter((s) => s.life > 0);
  for (const p of game.popups) { p.y -= H * 0.08 * dt; p.life -= dt; }
  game.popups = game.popups.filter((p) => p.life > 0);
  game.shake = Math.max(0, game.shake - dt);
}

// ---- drawing ------------------------------------------------------------------
function drawBackground() {
  const view = mp.view(W, H);
  if (mp.videoAllowed && mp.videoBitmap) {
    view.drawVideo(ctx);
    ctx.fillStyle = "rgba(10, 10, 30, 0.28)";
    ctx.fillRect(0, 0, W, H);
  } else {
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, "#1d2b53"); grad.addColorStop(1, "#7e2553");
    ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);
    for (const body of mp.bodies()) drawAvatar(ctx, mp, body, view);
  }
}

function drawFruit(f) {
  ctx.save();
  ctx.translate(f.x, f.y); ctx.rotate(f.rot);
  if (f.bomb) {
    ctx.fillStyle = "#222"; ctx.beginPath(); ctx.arc(0, 0, f.r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#ff3b3b"; ctx.lineWidth = f.r * 0.12;
    ctx.beginPath(); ctx.arc(0, 0, f.r * 0.7, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = "#c9a227"; ctx.lineWidth = f.r * 0.1;
    ctx.beginPath(); ctx.moveTo(0, -f.r); ctx.quadraticCurveTo(f.r * 0.4, -f.r * 1.5, f.r * 0.2, -f.r * 1.6); ctx.stroke();
    ctx.fillStyle = Math.random() < 0.5 ? "#ffd166" : "#ff6b35";
    ctx.beginPath(); ctx.arc(f.r * 0.2, -f.r * 1.6, f.r * (0.15 + Math.random() * 0.12), 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    return;
  }
  const k = f.kind;
  const grad = ctx.createRadialGradient(-f.r * 0.35, -f.r * 0.35, f.r * 0.1, 0, 0, f.r);
  grad.addColorStop(0, k.skin); grad.addColorStop(1, k.skin2);
  ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(0, 0, f.r, 0, Math.PI * 2); ctx.fill();
  if (k.stripes) {
    ctx.strokeStyle = "rgba(10, 40, 20, 0.6)"; ctx.lineWidth = f.r * 0.12;
    for (let i = -2; i <= 2; i++) {
      ctx.beginPath(); ctx.ellipse(i * f.r * 0.35, 0, f.r * 0.12, f.r * 0.95, 0, 0, Math.PI * 2); ctx.stroke();
    }
  }
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.beginPath(); ctx.ellipse(-f.r * 0.35, -f.r * 0.4, f.r * 0.25, f.r * 0.15, -0.6, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#4a7c23";
  ctx.beginPath(); ctx.ellipse(f.r * 0.25, -f.r * 0.95, f.r * 0.28, f.r * 0.12, -0.5, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawHalf(h) {
  ctx.save();
  ctx.globalAlpha = Math.min(1, h.life);
  ctx.translate(h.x, h.y); ctx.rotate(h.angle + h.rot);
  const start = h.side > 0 ? 0 : Math.PI;
  ctx.fillStyle = h.kind.skin2;
  ctx.beginPath(); ctx.arc(0, 0, h.r, start, start + Math.PI); ctx.closePath(); ctx.fill();
  ctx.fillStyle = h.kind.flesh;
  ctx.beginPath(); ctx.arc(0, 0, h.r * 0.86, start, start + Math.PI); ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawTrails(now) {
  for (const hand of hands.values()) {
    const tr = hand.trail;
    if (tr.length < 2) continue;
    const color = playerColor(hand.id);
    const active = hand.speed >= BLADE_SPEED;
    ctx.save();
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.shadowColor = color; ctx.shadowBlur = active ? 24 : 6;
    for (let i = 1; i < tr.length; i++) {
      const age = (now - tr[i].t) / TRAIL_S;
      ctx.globalAlpha = Math.max(0, 1 - age) * (active ? 1 : 0.35);
      ctx.strokeStyle = active ? "#ffffff" : color;
      ctx.lineWidth = (active ? 12 : 6) * (1 - age * 0.6);
      ctx.beginPath(); ctx.moveTo(tr[i - 1].x, tr[i - 1].y); ctx.lineTo(tr[i].x, tr[i].y); ctx.stroke();
    }
    const tip = tr[tr.length - 1];
    ctx.globalAlpha = 1; ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(tip.x, tip.y, H * 0.014, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }
}

function text(str, x, y, size, color = "#fff", align = "center") {
  ctx.save();
  ctx.font = `800 ${size}px "Segoe UI", system-ui, sans-serif`;
  ctx.textAlign = align; ctx.textBaseline = "middle";
  ctx.lineWidth = size * 0.14; ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.strokeText(str, x, y);
  ctx.fillStyle = color; ctx.fillText(str, x, y);
  ctx.restore();
}

function drawHud() {
  const s = H * 0.045;
  let y = s;
  const ids = [...new Set([...game.scores.keys(), ...mp.bodies().map((b) => b.id)])].sort();
  for (const id of ids) {
    text(`P${id}  ${game.scores.get(id) || 0}`, s * 0.6, y, s, playerColor(id), "left");
    y += s * 1.2;
  }
  if (game.phase === "play" || game.phase === "over") {
    text(`${Math.ceil(game.timeLeft)}`, W / 2, s, s * 1.1);
    text("♥".repeat(Math.max(0, game.lives)) + "♡".repeat(Math.max(0, LIVES - game.lives)),
      W / 2, s * 2.2, s * 0.8, "#ff6b6b");
  }
  if (game.phase === "title") {
    text("FRUIT SLICE", W / 2, H * 0.32, H * 0.12, "#ffd166");
    text("Swing your hands fast to slice the fruit. Don't touch the bombs!", W / 2, H * 0.45, H * 0.035);
    text(mp.bodies().length ? "Raise both hands to start" : "Step in front of the camera",
      W / 2, H * 0.56, H * 0.05, "#80ed99");
  } else if (game.phase === "countdown") {
    text(`${Math.max(1, 3 - Math.floor(game.phaseT))}`, W / 2, H * 0.45, H * 0.25, "#ffd166");
  } else if (game.phase === "over") {
    text(game.lives <= 0 ? "GAME OVER" : "TIME!", W / 2, H * 0.3, H * 0.1, "#ffd166");
    const best = [...game.scores.entries()].sort((a, b) => b[1] - a[1]);
    best.forEach(([id, pts], i) => text(`Player ${id}: ${pts}`, W / 2, H * (0.44 + i * 0.07), H * 0.05, playerColor(id)));
    if (!best.length) text("No fruit sliced - try swinging faster!", W / 2, H * 0.45, H * 0.04);
    if (game.phaseT > 2) text("Raise both hands to play again", W / 2, H * 0.78, H * 0.045, "#80ed99");
  }
  for (const p of game.popups) {
    ctx.globalAlpha = Math.min(1, p.life * 2);
    text(p.text, p.x, p.y, H * 0.045, p.color);
    ctx.globalAlpha = 1;
  }
  if (game.debug) {
    const st = mp.status;
    const lines = [`mode ${mp.mode}`, `players ${mp.bodies().length}`,
      st ? `engine ${st.processed_fps} fps, latency ${st.latency_ms.mean} ms (p95 ${st.latency_ms.p95})` : "",
      `hands ${[...hands.values()].map((h) => h.speed.toFixed(1)).join(" ")}`];
    lines.forEach((l, i) => text(l, 12, H - 20 - i * 22, 16, "#fff", "left"));
  }
}

let last = performance.now() / 1000;
function frame() {
  const now = performance.now() / 1000;
  const dt = Math.min(0.05, now - last);
  last = now;
  updateHands(now);
  if (!game.paused) {
    step(dt, now);
    checkSlices(now);
  }
  ctx.save();
  if (game.shake > 0) ctx.translate((Math.random() - 0.5) * 20 * game.shake, (Math.random() - 0.5) * 20 * game.shake);
  drawBackground();
  for (const s of game.splats) {
    ctx.globalAlpha = Math.min(0.5, s.life * 0.4);
    ctx.fillStyle = s.color;
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2, rr = s.r * (0.6 + 0.4 * Math.abs(Math.sin(s.seed + i * 1.7)));
      ctx.lineTo(s.x + Math.cos(a) * rr, s.y + Math.sin(a) * rr);
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  for (const h of game.halves) drawHalf(h);
  for (const f of game.fruits) drawFruit(f);
  for (const p of game.particles) {
    ctx.globalAlpha = Math.min(1, p.life * 2);
    ctx.fillStyle = p.color; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;
  drawTrails(now);
  ctx.restore();
  drawHud();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Exposed for automated smoke tests.
window.__fruitSlice = { game, hands, mp };
