/**
 * MediaMTX + WHEP Demo Server
 * ----------------------------
 * - Spawns mediamtx as a child process
 * - Serves the WHEP viewer frontend
 * - Proxies WHEP signaling requests to MediaMTX
 * - Exposes /api/status to check stream health
 */

const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 3000;
const MEDIAMTX_WEBRTC = "http://localhost:8889";
const MEDIAMTX_API = "http://localhost:9997";

// ── 1. Spawn MediaMTX ────────────────────────────────────────────────────────

const mtxBin = path.join(__dirname, "mediamtx");
const mtxCfg = path.join(__dirname, "mediamtx.yml");

if (!fs.existsSync(mtxBin)) {
  console.error("❌  mediamtx binary not found. Download it and place it next to server.js");
  process.exit(1);
}

const mtx = spawn(mtxBin, [mtxCfg], { stdio: "inherit" });

mtx.on("error", (err) => {
  console.error("❌  Failed to start mediamtx:", err.message);
  process.exit(1);
});

mtx.on("exit", (code) => {
  console.log(`ℹ️  mediamtx exited with code ${code}`);
});

process.on("exit", () => mtx.kill());
process.on("SIGINT", () => { mtx.kill(); process.exit(); });
process.on("SIGTERM", () => { mtx.kill(); process.exit(); });

console.log("🚀 MediaMTX started");

// ── 2. Serve static frontend ─────────────────────────────────────────────────

app.use(express.static(path.join(__dirname, "public")));

// ── 3. Stream status API ──────────────────────────────────────────────────────
// GET /api/status  →  fetches path list from MediaMTX API

app.get("/api/status", async (req, res) => {
  try {
    const response = await fetch(`${MEDIAMTX_API}/v3/paths/list`);
    if (!response.ok) throw new Error(`MediaMTX API error: ${response.status}`);
    const data = await response.json();

    // Find our camera path
    const camera = (data.items || []).find((p) => p.name === "camera");
    res.json({
      ok: true,
      stream: camera
        ? {
            name: camera.name,
            ready: camera.ready,
            readyTime: camera.readyTime,
            tracks: camera.tracks,
            bytesReceived: camera.bytesReceived,
          }
        : null,
    });
  } catch (err) {
    // MediaMTX might still be starting up
    res.json({ ok: false, error: err.message });
  }
});

// ── 4. Proxy WHEP requests to MediaMTX WebRTC port ───────────────────────────
// Browser sends:  POST /whep/camera
// We forward to:  POST http://localhost:8889/camera/whep

app.use(
  "/whep",
  createProxyMiddleware({
    target: MEDIAMTX_WEBRTC,
    changeOrigin: true,
    // /whep/camera  →  /camera/whep
    pathRewrite: (path) => {
      const streamName = path.replace(/^\//, "").replace(/\/.*$/, "");
      return `/${streamName}/whep`;
    },
    on: {
      proxyReq: (proxyReq, req) => {
        console.log(`🔀 WHEP ${req.method} ${req.path} → MediaMTX`);
      },
      error: (err, req, res) => {
        console.error("Proxy error:", err.message);
        res.status(502).json({ error: "MediaMTX not ready yet, retry in a moment" });
      },
    },
  })
);

// ── 5. Start ──────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`\n✅  Demo server running at http://localhost:${PORT}`);
  console.log(`    WHEP endpoint : http://localhost:${PORT}/whep/camera`);
  console.log(`    Stream status : http://localhost:${PORT}/api/status`);
  console.log(`    MediaMTX API  : ${MEDIAMTX_API}\n`);
});
