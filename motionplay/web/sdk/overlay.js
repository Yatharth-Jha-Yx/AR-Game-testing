// Safety + privacy overlay every game should install:
//  * camera-on indicator (and "skeleton only" badge),
//  * clear-space warning before play (dismiss: raise both hands / Enter / click),
//  * "Player N, step back in" when a player is lost, and bounds hints,
//  * a break reminder every N minutes (default 20) that pauses the game,
//  * a banner when the engine isn't running.
// Games listen for mp "pause"/"resume" events.

const HINTS = {
  step_back: "step back a little",
  step_forward: "come a bit closer",
  move_left: "move a little to your left",
  move_right: "move a little to your right",
};

export function installOverlay(mp, opts = {}) {
  const root = document.createElement("div");
  root.className = "mp-overlay";
  root.innerHTML = `
    <div class="mp-cam" hidden><span class="mp-dot"></span><span class="mp-cam-text">Camera on</span></div>
    <div class="mp-toasts"></div>
    <div class="mp-banner" hidden></div>
    <div class="mp-modal" hidden><div class="mp-card"></div></div>`;
  document.body.appendChild(root);
  const cam = root.querySelector(".mp-cam");
  const camText = root.querySelector(".mp-cam-text");
  const toasts = root.querySelector(".mp-toasts");
  const banner = root.querySelector(".mp-banner");
  const modal = root.querySelector(".mp-modal");
  const card = root.querySelector(".mp-card");

  const state = { modal: null, lost: new Map(), hintToast: new Map(), playStart: performance.now() };

  const showModal = (kind, html, onDone) => {
    state.modal = { kind, onDone };
    card.innerHTML = html;
    modal.hidden = false;
    mp._emit("pause", { reason: kind });
  };
  const closeModal = () => {
    if (!state.modal) return;
    const done = state.modal.onDone;
    state.modal = null;
    modal.hidden = true;
    mp._emit("resume", {});
    state.playStart = performance.now();
    if (done) done();
  };

  // Camera indicator + engine banner.
  const refresh = () => {
    const engine = mp.mode === "engine";
    cam.hidden = !engine;
    const skeleton = !mp.videoAllowed;
    camText.textContent = skeleton ? "Camera on · skeleton only" : "Camera on";
    banner.hidden = mp.mode === "engine";
    banner.textContent = mp.mode === "mouse"
      ? "Engine not running - mouse mode (move = right hand, arrows = swipe, space = jump). Start it with: motionplay run"
      : mp.mode === "offline" ? "Lost connection to the engine - reconnecting..." : "Connecting to engine...";
  };
  mp.on("connection", refresh);
  mp.on("settings", refresh);
  refresh();

  // Clear-space warning (once per browser session).
  if (opts.clearSpace !== false && !sessionStorage.getItem("mp-clear-space")) {
    showModal("clear-space", `
      <h2>Clear some space</h2>
      <p>Make sure you have about <b>2 metres</b> of free space around you, away from furniture,
         pets and little siblings.</p>
      <p class="mp-small">Raise both hands, press Enter, or click to continue.</p>`,
      () => sessionStorage.setItem("mp-clear-space", "1"));
  }
  const tryDismiss = () => { if (state.modal && state.modal.kind !== "cooldown") closeModal(); };
  addEventListener("keydown", (e) => { if (e.key === "Enter") tryDismiss(); });
  modal.addEventListener("click", tryDismiss);
  mp.on("gesture", (g) => {
    if (g.gesture === "arm_raise" && g.phase === "start" && g.side === "both") tryDismiss();
  });

  // Lost players + bounds hints.
  const toast = (key, text) => {
    let el = toasts.querySelector(`[data-key="${key}"]`);
    if (!el) {
      el = document.createElement("div");
      el.className = "mp-toast";
      el.dataset.key = key;
      toasts.appendChild(el);
    }
    el.textContent = text;
    return el;
  };
  const untoast = (key) => toasts.querySelector(`[data-key="${key}"]`)?.remove();
  mp.on("player", (ev) => {
    if (ev.event === "lost") toast(`lost-${ev.player_id}`, `Player ${ev.player_id}, step back in!`);
    else untoast(`lost-${ev.player_id}`);
  });
  mp.on("frame", () => {
    const seen = new Set();
    for (const b of mp.bodies()) {
      if (b.boundsHint && HINTS[b.boundsHint]) {
        toast(`hint-${b.id}`, `Player ${b.id}: ${HINTS[b.boundsHint]}`);
        seen.add(b.id);
      }
    }
    toasts.querySelectorAll('[data-key^="hint-"]').forEach((el) => {
      if (!seen.has(Number(el.dataset.key.slice(5)))) el.remove();
    });
  });

  // Break reminder.
  const minutes = () => opts.cooldownMinutes ?? mp.settings.cooldown_minutes ?? 20;
  setInterval(() => {
    if (state.modal || !minutes()) return;
    if (performance.now() - state.playStart < minutes() * 60000) return;
    let left = 30;
    showModal("cooldown", `<h2>Time for a short break</h2>
      <p>You've been moving for ${minutes()} minutes. Drink some water and shake out your arms.</p>
      <p class="mp-small">Back in <b class="mp-count">${left}</b> s</p>`);
    const timer = setInterval(() => {
      left -= 1;
      const el = card.querySelector(".mp-count");
      if (el) el.textContent = left;
      if (left <= 0) { clearInterval(timer); closeModal(); }
    }, 1000);
  }, 5000);

  return { showModal, closeModal, toast, untoast, get paused() { return !!state.modal; } };
}
