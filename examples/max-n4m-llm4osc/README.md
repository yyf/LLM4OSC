# LLM4OSC · Node for Max example

Thin Max client for the existing local stack. Max **resolves** NL via `llm4osc serve`; it does **not** embed the model or Tier 3. UDP send is optional and gated in the patch (`live` toggle).

```
Max [node.script]
      │  POST /v1/resolve  { nl, device_id, backend: b0 }
      ▼
llm4osc serve (localhost:8765)
      │  SuccessIntent | RefusalIntent
      ▼
outlet: osc /gain 0.5  |  refuse unknown_pattern …
      │
      ├─ print (always)
      └─ [gate] → [udpsend 127.0.0.1 7400]   only if live=1
```

**Invariant:** models/rules propose; you choose whether the patch sends. Default is preview-only.

---

## Requirements

- Max 8+ with **Node for Max**
- LLM4OSC installed (`pip install -e .`)
- Committed `max-msp` profile (ships with the repo)

---

## Quick start

**Terminal 1 — serve (B0; no GPU preload):**

```bash
cd /path/to/LLM4OSC
llm4osc serve --no-preload
```

**Max**

1. Open `examples/max-n4m-llm4osc/llm4osc.maxpat`
2. Click `script start` (or enable `@autostart 1` on `node.script`)
3. Click `health` — Max window should show serve OK
4. Click an example (`resolve set gain to 50%`) or type NL in the textedit
5. Keep **live** off until a second patch is listening: `[udpreceive 7400]`

**Optional second Max patch (target):**

```
[udpreceive 7400] → [route /gain /volume /freq /pan …] → number / toggle UI
```

Same host/port as CLI: `llm4osc send --device max-msp --nl "…" -y`

---

## Messages to `[node.script]`

```
Message                         | Effect
--------------------------------+----------------------------------
resolve <natural language…>     | POST /v1/resolve
serve http://127.0.0.1:8765     | Set serve base URL
device max-msp                   | Committed profile device_id
backend b0|b1|b2|b3             | Default b0 (recommended)
gate 0|1                        | Retrieval gate for B1–B3
health                          | GET /health
help                            | Console usage line
script start / script stop      | N4M process control
```

---

## Outlet format (leftmost)

```
Selector   | Payload
-----------+----------------------------------
osc        | <address> <args…>     e.g. /gain 0.5
refuse     | <reason> <message…>
error      | <text…>
health     | ok|fail <detail…>
status     | ready|resolving|…
```

The patch `[route osc refuse error health status]` splits these. Only the `osc` branch feeds `[udpsend]`, and only when `live` is on.

---

## Smoke test without Max

With serve running:

```bash
node examples/max-n4m-llm4osc/smoke-test.js
```

---

## Design notes

- **Not** a Max SDK C external — Node for Max keeps the client thin.
- Prefer **b0** for show-like demos; B1–B3 need `pip install -e ".[llm]"` and a warm serve (drop `--no-preload`, optional adapter).
- Profile Acceptance / commit gate stay in the Python CLI or Gradio demo — this example is the Max-side resolve + optional send surface.
- Package Manager distribution would wrap this folder later; for now it lives under `examples/`.
