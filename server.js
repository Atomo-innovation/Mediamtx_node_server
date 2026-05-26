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
const PORT   = 3000;
const MTX_WEBRTC = "http://localhost:8889";
const MTX_API    = "http://localhost:9997";

const MTX_BIN    = path.join(__dirname, "mediamtx");
const MTX_YML    = path.join(__dirname, "mediamtx.yml");
const CAMERAS_DB = path.join(__dirname, "cameras.json");

// Keep last N face events in memory so the UI can fetch history on load
const MAX_EVENTS = 200;
const faceEvents = [];

app.use(express.json({ limit: "4mb" }));   // face crops are base64 JPEG

// ── Camera persistence ────────────────────────────────────────────────────────

function loadCameras() {
  if (!fs.existsSync(CAMERAS_DB)) return [];
  try { return JSON.parse(fs.readFileSync(CAMERAS_DB, "utf8")); } catch { return []; }
}

function saveCameras(cams) {
  fs.writeFileSync(CAMERAS_DB, JSON.stringify(cams, null, 2));
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
  wss.clients.forEach(c => { if (c.readyState === 1) c.send(JSON.stringify({ type: "face_clear" })); });
  res.json({ ok: true });
});

// ── Static ────────────────────────────────────────────────────────────────────
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
  // Send history on connect
  ws.send(JSON.stringify({ type: "face_history", events: [...faceEvents].reverse() }));
  ws.on("close", () => console.log("🔌 WS client disconnected"));
});

server.listen(PORT, () => {
  console.log(`\n✅  Server → http://localhost:${PORT}`);
  console.log(`    Cameras loaded: ${initCams.length}`);
  console.log(`    WebSocket: ws://localhost:${PORT}/ws`);
});
