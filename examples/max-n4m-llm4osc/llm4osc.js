/**
 * LLM4OSC — Node for Max thin client (example)
 *
 * Resolves natural language via local `llm4osc serve` (POST /v1/resolve).
 * Does **not** send UDP. Wire outlet messages to [udpsend] only when you intend live send.
 *
 * Default backend: b0 (rules). Prefer:
 *   llm4osc serve --no-preload
 *
 * Max messages (to [node.script]):
 *   resolve set gain to 50%
 *   serve http://127.0.0.1:8765
 *   device max-msp
 *   backend b0
 *   gate 1
 *   health
 *
 * Outlet 0 (leftmost) — selector + payload:
 *   osc <address> <args...>
 *   refuse <reason> <message...>
 *   error <text...>
 *   health <ok-or-fail> <detail...>
 *   status <text...>
 */

"use strict";

const http = require("http");
const https = require("https");
const { URL } = require("url");

let Max = null;
try {
  Max = require("max-api");
} catch (_err) {
  // Allow plain-node smoke tests without Max.
  Max = {
    post: (...args) => console.error("[llm4osc]", ...args),
    outlet: (...args) => console.log(JSON.stringify({ outlet: args })),
    addHandler: () => {},
  };
}

const state = {
  serveUrl: process.env.LLM4OSC_SERVE_URL || "http://127.0.0.1:8765",
  deviceId: "max-msp",
  backend: "b0",
  retrievalGate: true,
};

function postJson(urlString, payload) {
  const url = new URL(urlString);
  const body = JSON.stringify(payload);
  const lib = url.protocol === "https:" ? https : http;
  const options = {
    hostname: url.hostname,
    port: url.port || (url.protocol === "https:" ? 443 : 80),
    path: url.pathname + url.search,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(body),
    },
    timeout: 120000,
  };

  return new Promise((resolve, reject) => {
    const req = lib.request(options, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        let data;
        try {
          data = JSON.parse(text);
        } catch (err) {
          reject(new Error(`invalid JSON from serve (${res.statusCode}): ${text}`));
          return;
        }
        if (res.statusCode && res.statusCode >= 400) {
          reject(new Error(data.error || `HTTP ${res.statusCode}`));
          return;
        }
        resolve(data);
      });
    });
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("serve request timed out"));
    });
    req.write(body);
    req.end();
  });
}

function getJson(urlString) {
  const url = new URL(urlString);
  const lib = url.protocol === "https:" ? https : http;
  const options = {
    hostname: url.hostname,
    port: url.port || (url.protocol === "https:" ? 443 : 80),
    path: url.pathname + url.search,
    method: "GET",
    timeout: 10000,
  };

  return new Promise((resolve, reject) => {
    const req = lib.request(options, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        try {
          resolve(JSON.parse(text));
        } catch (err) {
          reject(new Error(`invalid JSON from health: ${text}`));
        }
      });
    });
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("health request timed out"));
    });
    req.end();
  });
}

async function resolveNl(nl) {
  const trimmed = String(nl || "").trim();
  if (!trimmed) {
    Max.outlet("error", "empty_nl");
    return;
  }

  Max.post(
    `resolve [${state.backend}] device=${state.deviceId} ← ${trimmed}`
  );
  Max.outlet("status", "resolving");

  try {
    const data = await postJson(`${state.serveUrl.replace(/\/$/, "")}/v1/resolve`, {
      nl: trimmed,
      device_id: state.deviceId,
      backend: state.backend,
      retrieval_gate: state.retrievalGate,
    });

    if (!data.ok) {
      Max.outlet("error", data.error || "resolve_failed");
      return;
    }

    const result = data.result;
    if (!result || !result.kind) {
      Max.outlet("error", "malformed_result");
      return;
    }

    if (result.kind === "refusal") {
      Max.outlet("refuse", result.reason || "unknown", result.message || "");
      Max.post(`refuse ${result.reason}: ${result.message || ""}`);
      return;
    }

    if (result.kind === "intent") {
      const args = Array.isArray(result.args) ? result.args : [];
      Max.outlet("osc", result.address, ...args);
      Max.post(
        `preview ${result.address} ${JSON.stringify(args)} ` +
          `(pattern=${result.pattern_id}) — not sent; wire [udpsend] yourself`
      );
      return;
    }

    Max.outlet("error", "unknown_kind", String(result.kind));
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    Max.post(`error: ${msg}`);
    Max.outlet("error", msg);
  }
}

function registerHandlers() {
  if (!Max.addHandler) return;

  Max.addHandler("resolve", (...words) => {
    void resolveNl(words.join(" "));
  });

  Max.addHandler("serve", (url) => {
    if (url) {
      state.serveUrl = String(url).replace(/\/$/, "");
      Max.post(`serveUrl = ${state.serveUrl}`);
      Max.outlet("status", "serve", state.serveUrl);
    }
  });

  Max.addHandler("device", (id) => {
    if (id) {
      state.deviceId = String(id);
      Max.post(`deviceId = ${state.deviceId}`);
      Max.outlet("status", "device", state.deviceId);
    }
  });

  Max.addHandler("backend", (b) => {
    const key = String(b || "").toLowerCase();
    if (!["b0", "b1", "b2", "b3"].includes(key)) {
      Max.outlet("error", "backend must be b0|b1|b2|b3");
      return;
    }
    state.backend = key;
    Max.post(`backend = ${state.backend}`);
    Max.outlet("status", "backend", state.backend);
  });

  Max.addHandler("gate", (v) => {
    const on = !(v === 0 || v === "0" || v === false || v === "off");
    state.retrievalGate = on;
    Max.post(`retrieval_gate = ${on}`);
    Max.outlet("status", "gate", on ? 1 : 0);
  });

  Max.addHandler("health", async () => {
    try {
      const data = await getJson(`${state.serveUrl.replace(/\/$/, "")}/health`);
      Max.outlet("health", data.ok ? "ok" : "fail", JSON.stringify(data));
      Max.post(`health: ${JSON.stringify(data)}`);
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      Max.outlet("health", "fail", msg);
      Max.post(`health fail: ${msg}`);
    }
  });

  Max.addHandler("help", () => {
    Max.post(
      "llm4osc n4m: resolve <nl> | serve <url> | device <id> | backend b0|b1|b2|b3 | gate 0|1 | health"
    );
  });
}

registerHandlers();
Max.post(
  `llm4osc n4m ready · ${state.serveUrl} · device=${state.deviceId} · backend=${state.backend}`
);
Max.outlet("status", "ready");

module.exports = { resolveNl, state, postJson };
