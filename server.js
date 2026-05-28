/**
 * MediaMTX Multi-Camera WHEP Server
 * -----------------------------------
 * Cameras saved to cameras.json · hot-restarts MediaMTX on change
 * Face-detection events received from face_detector.py · broadcast via WebSocket
 */

const express    = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const { spawn }  = require("child_process");
const { WebSocketServer } = require("ws");
const http       = require("http");
const path       = require("path");
const fs         = require("fs");

const app    = express();
const PORT   = 8129;
const MTX_WEBRTC = "http://localhost:8889";
const MTX_API    = "http://localhost:9997";

const MTX_BIN    = path.join(__dirname, "mediamtx");
const MTX_YML    = path.join(__dirname, "mediamtx.yml");
const CAMERAS_DB  = path.join(__dirname, "cameras.json");
const CLUSTERS_DB = path.join(__dirname, "clusters.json");
const EVENTS_DB   = path.join(__dirname, "face_events.json");

// face_detector.py management API (enroll / known endpoints)
const MGMT_BASE   = "http://localhost:5051";

// Keep last N face events in memory so the UI can fetch history on load
const MAX_EVENTS = 2000;
const faceEvents = (() => {
  // Restore from disk on startup so clusters survive server restarts
  if (fs.existsSync(EVENTS_DB)) {
    try { return JSON.parse(fs.readFileSync(EVENTS_DB, "utf8")); } catch {}
  }
  return [];
})();

let _saveEventsTimer = null;
function scheduleEventsSave() {
  clearTimeout(_saveEventsTimer);
  _saveEventsTimer = setTimeout(() => {
    try { fs.writeFileSync(EVENTS_DB, JSON.stringify(faceEvents)); } catch {}
  }, 2000); // debounce — write at most every 2s
}

app.use(express.json({ limit: "16mb" }));  // HD face crops / cluster payloads

// ── Camera persistence ────────────────────────────────────────────────────────

function loadCameras() {
  if (!fs.existsSync(CAMERAS_DB)) return [];
  try { return JSON.parse(fs.readFileSync(CAMERAS_DB, "utf8")); } catch { return []; }
}

function saveCameras(cams) {
  fs.writeFileSync(CAMERAS_DB, JSON.stringify(cams, null, 2));
}

// ── Cluster persistence ────────────────────────────────────────────────────────

function loadClusters() {
  if (!fs.existsSync(CLUSTERS_DB)) return { clusters: [], excludedFaceTs: [] };
  try {
    const data = JSON.parse(fs.readFileSync(CLUSTERS_DB, "utf8"));
    if (Array.isArray(data)) return { clusters: data, excludedFaceTs: [] };
    return {
      clusters: Array.isArray(data.clusters) ? data.clusters : [],
      excludedFaceTs: Array.isArray(data.excludedFaceTs) ? data.excludedFaceTs : [],
    };
  } catch { return { clusters: [], excludedFaceTs: [] }; }
}

function saveClustersPayload(payload) {
  fs.writeFileSync(CLUSTERS_DB, JSON.stringify(payload, null, 2));
}

// ── MediaMTX config generation ────────────────────────────────────────────────

function writeMtxConfig(cameras) {
  const pathsBlock = cameras.map(cam => {
    return `  ${cam.id}:\n    source: ${cam.url}\n    sourceOnDemand: no\n    sourceProtocol: tcp\n    readUser:\n    readPass:`;
  }).join("\n\n");

  const yml = `logLevel: info

rtsp: yes
rtspAddress: :8554

rtmp: no
hls: no

webrtc: yes
webrtcAddress: :8889

api: yes
apiAddress: :9997

paths:
${pathsBlock || "  ~.*:\n    source: publisher"}
`;
  fs.writeFileSync(MTX_YML, yml);
}

// ── MediaMTX process management ───────────────────────────────────────────────

let mtxProc = null;
let mtxRestartTimer = null;

function startMtx() {
  if (!fs.existsSync(MTX_BIN)) {
    console.error("❌  mediamtx binary not found");
    process.exit(1);
  }

  if (mtxRestartTimer) { clearTimeout(mtxRestartTimer); }
  mtxRestartTimer = setTimeout(() => {
    if (mtxProc) {
      mtxProc.removeAllListeners();
      mtxProc.kill("SIGTERM");
      mtxProc = null;
    }
    mtxProc = spawn(MTX_BIN, [MTX_YML], { stdio: "inherit" });
    mtxProc.on("error", err => console.error("❌  mediamtx error:", err.message));
    mtxProc.on("exit",  code => { console.log(`ℹ️   mediamtx exited (${code})`); mtxProc = null; });
    console.log("🚀 MediaMTX (re)started");
  }, 300);
}

// Init
const initCams = loadCameras();
writeMtxConfig(initCams);
startMtx();

process.on("exit",   () => mtxProc && mtxProc.kill());
process.on("SIGINT", () => { mtxProc && mtxProc.kill(); process.exit(); });
process.on("SIGTERM",() => { mtxProc && mtxProc.kill(); process.exit(); });

// ── REST API ──────────────────────────────────────────────────────────────────

app.get("/api/cameras", (_req, res) => res.json(loadCameras()));

app.post("/api/cameras", (req, res) => {
  const { name, url } = req.body;
  if (!name || !url) return res.status(400).json({ error: "name and url required" });

  const cams = loadCameras();
  const id   = name.toLowerCase().replace(/[^a-z0-9]/g,"_").replace(/_+/g,"_").replace(/^_|_$/g,"")
             + "_" + Date.now().toString(36);

  const cam  = { id, name, url, addedAt: new Date().toISOString() };
  cams.push(cam);
  saveCameras(cams);
  writeMtxConfig(cams);
  startMtx();
  res.json(cam);
});

app.delete("/api/cameras/:id", (req, res) => {
  let cams = loadCameras();
  const before = cams.length;
  cams = cams.filter(c => c.id !== req.params.id);
  if (cams.length === before) return res.status(404).json({ error: "not found" });
  saveCameras(cams);
  writeMtxConfig(cams);
  startMtx();
  res.json({ ok: true });
});

app.get("/api/status", async (_req, res) => {
  try {
    const r = await fetch(`${MTX_API}/v3/paths/list`);
    if (!r.ok) throw new Error(`API ${r.status}`);
    const data = await r.json();
    res.json({ ok: true, paths: data.items || [] });
  } catch (e) {
    res.json({ ok: false, error: e.message, paths: [] });
  }
});

// ── Face events API ───────────────────────────────────────────────────────────

// GET /api/face-events  → last MAX_EVENTS events (newest first)
app.get("/api/face-events", (_req, res) => {
  res.json([...faceEvents].reverse());
});

// POST /api/face-events  → called by face_detector.py
app.post("/api/face-events", (req, res) => {
  const ev = req.body;
  if (!ev || !ev.camId) return res.status(400).json({ error: "camId required" });

  ev.serverTs = Date.now();
  faceEvents.push(ev);
  if (faceEvents.length > MAX_EVENTS) faceEvents.shift();
  scheduleEventsSave();

  // Broadcast to all connected WebSocket clients
  const msg = JSON.stringify({ type: "face_event", event: ev });
  wss.clients.forEach(client => {
    if (client.readyState === 1 /* OPEN */) client.send(msg);
  });

  res.json({ ok: true });
});

// DELETE /api/face-events  → clear log
app.delete("/api/face-events", (_req, res) => {
  faceEvents.length = 0;
  try { fs.writeFileSync(EVENTS_DB, "[]"); } catch {}
  wss.clients.forEach(c => { if (c.readyState === 1) c.send(JSON.stringify({ type: "face_clear" })); });
  res.json({ ok: true });
});

// ── Cluster API ───────────────────────────────────────────────────────────────

// GET /api/clusters → load persisted cluster data
app.get("/api/clusters", (_req, res) => {
  res.json(loadClusters());
});

// POST /api/clusters → save cluster data (full replace)
app.post("/api/clusters", (req, res) => {
  const body = req.body;
  let payload;
  if (Array.isArray(body)) {
    payload = { clusters: body, excludedFaceTs: loadClusters().excludedFaceTs };
  } else if (body && Array.isArray(body.clusters)) {
    payload = {
      clusters: body.clusters,
      excludedFaceTs: Array.isArray(body.excludedFaceTs) ? body.excludedFaceTs : [],
    };
  } else {
    return res.status(400).json({ error: "expected clusters array or { clusters, excludedFaceTs }" });
  }
  saveClustersPayload(payload);
  res.json({ ok: true, count: payload.clusters.length });
});

// ── Worker config API ─────────────────────────────────────────────────────────

// Each camera gets a config entry pointing at MediaMTX's local RTSP relay.
// worker.py polls this to know which cameras are active.
app.get("/api/internal/worker-config", (_req, res) => {
  const cams = loadCameras();
  res.json({
    version: cams.map(c => c.id).join(",") || "empty",
    cameras: cams.map(cam => ({
      id:       cam.id,
      name:     cam.name,
      // Pull from MediaMTX's local RTSP relay — NOT the original RTSP source.
      // This way worker.py always reads the stream MediaMTX is already ingesting.
      source:   `rtsp://127.0.0.1:8554/${cam.id}`,
      forceTcp: true,
      enabled:  true,
    })),
  });
});

// Camera live-status updates from worker.py
const cameraStatus = {};   // id → { connected, updatedAt }
const cameraPreviews = {}; // id → base64 jpeg

app.post("/api/cameras/:id/status", (req, res) => {
  const { connected } = req.body;
  cameraStatus[req.params.id] = { connected: !!connected, updatedAt: Date.now() };
  wss.clients.forEach(c => {
    if (c.readyState === 1) c.send(JSON.stringify({
      type: "cam_status",
      id: req.params.id,
      connected: !!connected,
    }));
  });
  res.json({ ok: true });
});

app.post("/api/cameras/:id/preview", (req, res) => {
  const { previewB64 } = req.body;
  if (!previewB64) return res.status(400).json({ error: "previewB64 required" });
  cameraPreviews[req.params.id] = previewB64;
  wss.clients.forEach(c => {
    if (c.readyState === 1) c.send(JSON.stringify({
      type: "cam_preview",
      id: req.params.id,
      previewB64,
    }));
  });
  res.json({ ok: true });
});

app.get("/api/cameras/:id/preview", (req, res) => {
  const b64 = cameraPreviews[req.params.id];
  if (!b64) return res.status(404).json({ error: "no preview yet" });
  res.json({ previewB64: b64 });
});

// ── Detection API (called by worker.py after recognition) ────────────────────
// worker.py posts { cameraId, cameraName, embedding, snapshotB64, pose }
// We proxy to face_detector.py for recognition, then broadcast the result.

app.post("/api/detections", async (req, res) => {
  const { cameraId, cameraName, embedding, snapshotB64, pose } = req.body;
  if (!cameraId || !embedding) return res.status(400).json({ error: "cameraId and embedding required" });

  let name = null, score = null, known = false;

  // Ask face_detector.py to do recognition (if running)
  try {
    const r = await fetch(`${MGMT_BASE}/api/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embedding }),
      signal: AbortSignal.timeout(4000),
    });
    if (r.ok) {
      const d = await r.json();
      name  = d.name  ?? null;
      score = d.score ?? null;
      known = d.matched === true;
    }
  } catch (_) {
    // face_detector.py not running — still record the event as unknown
  }

  const ev = {
    camId:   cameraId,
    camName: cameraName,
    name:    name || "Unknown",
    known,
    score,
    pose,
    crop:    snapshotB64 || null,
    ts:      Date.now(),
    serverTs: Date.now(),
  };

  faceEvents.push(ev);
  if (faceEvents.length > MAX_EVENTS) faceEvents.shift();
  scheduleEventsSave();

  const msg = JSON.stringify({ type: "face_event", event: ev });
  wss.clients.forEach(client => {
    if (client.readyState === 1) client.send(msg);
  });

  res.json({ ok: true, known, name, score });
});

// ── Face DB management proxy → face_detector.py :5051 ────────────────────────

// GET /api/known → list enrolled names
app.get("/api/known", async (_req, res) => {
  try {
    const r = await fetch(`${MGMT_BASE}/api/known`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    res.json(d);
  } catch (e) {
    // Return empty gracefully if face_detector.py not running
    res.json({ names: [] });
  }
});

// DELETE /api/known/:name → delete person from face DB
app.delete("/api/known/:name", async (req, res) => {
  try {
    const name = req.params.name;
    const r = await fetch(`${MGMT_BASE}/api/known/${encodeURIComponent(name)}`,
      { method: "DELETE", signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    res.status(r.ok ? 200 : 404).json(d);
  } catch (e) {
    res.status(502).json({ error: "face_detector.py not reachable", detail: e.message });
  }
});

// POST /api/enroll → enroll a single face crop (proxy to face_detector.py)
app.post("/api/enroll", async (req, res) => {
  const { name, crop_b64 } = req.body;
  if (!name || !crop_b64) return res.status(400).json({ error: "name and crop_b64 required" });
  try {
    const r = await fetch(`${MGMT_BASE}/api/enroll`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, crop_b64 }),
      signal: AbortSignal.timeout(8000),
    });
    const d = await r.json();
    res.status(r.ok ? 200 : 400).json(d);
  } catch (e) {
    res.status(502).json({ error: "face_detector.py not reachable", detail: e.message });
  }
});

// ── Static (place index.html AND clusters.html in ./public/) ─────────────────
app.use(express.static(path.join(__dirname, "public")));

// ── WHEP proxy ────────────────────────────────────────────────────────────────
app.use("/whep", createProxyMiddleware({
  target: MTX_WEBRTC,
  changeOrigin: true,
  pathRewrite: p => {
    const id = p.replace(/^\//,"").split("/")[0];
    return `/${id}/whep`;
  },
  on: {
    proxyReq: (_, req) => console.log(`🔀 WHEP ${req.method} ${req.path}`),
    error: (err, _req, res) => {
      console.error("Proxy:", err.message);
      res.status(502).json({ error: "MediaMTX not ready, retry shortly" });
    }
  }
}));

// ── HTTP server + WebSocket ───────────────────────────────────────────────────
const server = http.createServer(app);
const wss    = new WebSocketServer({ server, path: "/ws" });

wss.on("connection", ws => {
  console.log("🔌 WS client connected");
  // Send face event history on connect
  ws.send(JSON.stringify({ type: "face_history", events: [...faceEvents].reverse() }));
  // Send current camera statuses so UI reflects worker state immediately
  ws.send(JSON.stringify({ type: "cam_status_all", statuses: cameraStatus }));
  // Send any cached preview frames
  for (const [id, previewB64] of Object.entries(cameraPreviews)) {
    ws.send(JSON.stringify({ type: "cam_preview", id, previewB64 }));
  }
  ws.on("close", () => console.log("🔌 WS client disconnected"));
});

server.listen(PORT, () => {
  console.log(`\n✅  Server → http://localhost:${PORT}`);
  console.log(`    Cameras loaded: ${initCams.length}`);
  console.log(`    WebSocket: ws://localhost:${PORT}/ws`);
});
