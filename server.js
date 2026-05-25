/**
 * MediaMTX Multi-Camera WHEP Server
 * -----------------------------------
 * - Cameras persisted in cameras.json
 * - Dynamically rewrites mediamtx.yml on add/remove
 * - Hot-restarts MediaMTX when config changes
 * - Proxies WHEP signaling to MediaMTX
 */

const express    = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const { spawn }  = require("child_process");
const path       = require("path");
const fs         = require("fs");

const app  = express();
const PORT = 8050;
const MTX_WEBRTC = "http://localhost:8889";
const MTX_API    = "http://localhost:9997";

const MTX_BIN    = path.join(__dirname, "mediamtx");
const MTX_YML    = path.join(__dirname, "mediamtx.yml");
const CAMERAS_DB = path.join(__dirname, "cameras.json");

app.use(express.json());

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
    const id = cam.id;
    return `  ${id}:\n    source: ${cam.url}\n    sourceOnDemand: no\n    sourceProtocol: tcp\n    readUser:\n    readPass:`;
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

function startMtx() {
  if (!fs.existsSync(MTX_BIN)) {
    console.error("❌  mediamtx binary not found");
    process.exit(1);
  }

  if (mtxProc) {
    mtxProc.removeAllListeners();
    mtxProc.kill("SIGTERM");
    mtxProc = null;
  }

  mtxProc = spawn(MTX_BIN, [MTX_YML], { stdio: "inherit" });

  mtxProc.on("error", err => console.error("❌  mediamtx error:", err.message));
  mtxProc.on("exit",  code => {
    console.log(`ℹ️   mediamtx exited (${code})`);
    mtxProc = null;
  });

  console.log("🚀 MediaMTX (re)started");
}

// Init
const cameras = loadCameras();
writeMtxConfig(cameras);
startMtx();

process.on("exit",   () => mtxProc && mtxProc.kill());
process.on("SIGINT", () => { mtxProc && mtxProc.kill(); process.exit(); });
process.on("SIGTERM",() => { mtxProc && mtxProc.kill(); process.exit(); });

// ── REST API ──────────────────────────────────────────────────────────────────

// GET /api/cameras  — list all saved cameras
app.get("/api/cameras", (req, res) => {
  res.json(loadCameras());
});

// POST /api/cameras  — add a camera  { name, url }
app.post("/api/cameras", (req, res) => {
  const { name, url } = req.body;
  if (!name || !url) return res.status(400).json({ error: "name and url required" });

  const cams = loadCameras();

  // Build a safe path ID from name
  const id = name.toLowerCase().replace(/[^a-z0-9]/g, "_").replace(/_+/g, "_").replace(/^_|_$/g, "") + "_" + Date.now().toString(36);

  const cam = { id, name, url, addedAt: new Date().toISOString() };
  cams.push(cam);
  saveCameras(cams);
  writeMtxConfig(cams);
  startMtx();   // hot-restart with new config

  res.json(cam);
});

// DELETE /api/cameras/:id
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

// GET /api/status  — MediaMTX path status for all cameras
app.get("/api/status", async (req, res) => {
  try {
    const r = await fetch(`${MTX_API}/v3/paths/list`);
    if (!r.ok) throw new Error(`API ${r.status}`);
    const data = await r.json();
    res.json({ ok: true, paths: data.items || [] });
  } catch (e) {
    res.json({ ok: false, error: e.message, paths: [] });
  }
});

// ── Static frontend ───────────────────────────────────────────────────────────

app.use(express.static(path.join(__dirname, "public")));

// ── WHEP proxy  /whep/:camId  →  http://localhost:8889/:camId/whep ────────────

app.use("/whep", createProxyMiddleware({
  target: MTX_WEBRTC,
  changeOrigin: true,
  pathRewrite: reqPath => {
    const camId = reqPath.replace(/^\//, "").split("/")[0];
    return `/${camId}/whep`;
  },
  on: {
    proxyReq: (_, req) => console.log(`🔀 WHEP ${req.method} ${req.path}`),
    error: (err, _req, res) => {
      console.error("Proxy error:", err.message);
      res.status(502).json({ error: "MediaMTX not ready, retry shortly" });
    }
  }
}));

// ── Start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`\n✅  Server → http://localhost:${PORT}`);
  console.log(`    Cameras  : ${cameras.length} loaded from cameras.json`);
});
