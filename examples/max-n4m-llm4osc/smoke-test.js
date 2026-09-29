#!/usr/bin/env node
/**
 * Smoke-test the N4M client logic against llm4osc serve (no Max required).
 *
 *   llm4osc serve --no-preload
 *   node examples/max-n4m-llm4osc/smoke-test.js
 */

"use strict";

const http = require("http");
const { resolveNl, state, postJson } = require("./llm4osc.js");

async function getHealth(base) {
  return new Promise((resolve, reject) => {
    http
      .get(`${base}/health`, (res) => {
        let raw = "";
        res.on("data", (c) => (raw += c));
        res.on("end", () => {
          try {
            resolve(JSON.parse(raw));
          } catch (err) {
            reject(err);
          }
        });
      })
      .on("error", reject);
  });
}

async function main() {
  const base = state.serveUrl.replace(/\/$/, "");
  console.error(`serve: ${base}  device=${state.deviceId}  backend=${state.backend}`);

  const health = await getHealth(base);
  console.error("health:", health);

  const cases = [
    "set gain to 50%",
    "make the level half",
    "boost the bass band by 3db",
    "start",
  ];

  for (const nl of cases) {
    console.error("\n---", nl);
    await resolveNl(nl);
  }

  const direct = await postJson(`${base}/v1/resolve`, {
    nl: "set gain to 50%",
    device_id: "max-msp",
    backend: "b0",
    retrieval_gate: true,
  });
  if (
    !direct.ok ||
    direct.result.kind !== "intent" ||
    direct.result.address !== "/gain"
  ) {
    console.error("FAIL direct resolve", direct);
    process.exit(1);
  }
  console.error("\nOK direct resolve →", direct.result.address, direct.result.args);
}

main().catch((err) => {
  console.error(err.message || err);
  console.error("\nStart serve first:\n  llm4osc serve --no-preload");
  process.exit(1);
});
