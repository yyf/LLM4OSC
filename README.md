# LLM4OSC

Turn natural language into **validated Open Sound Control (OSC)** — locally, deterministically, and reliably.

```
device profile  +  natural language
        ↓
   intent JSON  (proposed)
        ↓
   validate → encode → OSC
```

Rules or a small local LLM **propose** structured intents; deterministic code **validates and sends**. Same validated intent → same OSC bytes. Ambiguous requests are refused, not guessed.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional **Qwen2-0.5B** (`--backend b1` / `b2`):

```bash
python -m pip install -e ".[dev,llm]"
```

## Quick start

```bash
llm4osc profile list-patterns --device max-msp
llm4osc send --device max-msp --nl "set gain to 50%" --dry-run -y   # preview only
llm4osc send --device max-msp --nl "set gain to 50%" -y              # live UDP → 127.0.0.1:7400
python scripts/osc_listen.py                                        # loopback listener
```

Max/MSP: add `udpreceive 7400` in a patch. Set `LLM4OSC_HOST` / `LLM4OSC_PORT` to override defaults.

NL backends: `--backend b0` (rules, default), `b1` (Qwen zero-shot), `b2` (Qwen few-shot).  
`LLM4OSC_DEBUG=1` prints raw model output. `LLM4OSC_MODEL` overrides the Hugging Face model id.

### LLM serve (B1/B2)

Load Qwen **once** and reuse across CLI calls:

```bash
# Terminal 1 — loads model at startup (~few seconds once cached)
llm4osc serve

# Terminal 2 — talks to server instead of reloading weights
export LLM4OSC_SERVE_URL=http://127.0.0.1:8765
llm4osc send --device max-msp --nl "set gain to 50%" --backend b1 --dry-run -y
llm4osc score-compare --backends b0,b1,b2
```

Or pass `--serve-url http://127.0.0.1:8765` per command. B0 ignores the server (rules only).

## Benchmark results (Max/MSP hero profile)

### Literal suite (CI gates)

Frozen suite: 8 NL + 3 refusal cases — `benchmarks/golden_nl/`, `benchmarks/golden_refusal/`.

| Metric | B0 (rules) | B1 (Qwen2-0.5B) |
|--------|------------|-----------------|
| Semantic accuracy | **100%** | 37.5% |
| Wrong-send rate | **0%** | 9.1% |
| Refusal recall | **100%** | 0% |
| Latency p50 | **0.04 ms** | ~3.2 s |

B0 passes all gates on the literal suite. Use **B0 for demos and live control**.

### Track C — paraphrase suite

8 paraphrase NL cases (`benchmarks/golden_nl_paraphrase/`) — same intents as the literal suite, different wording.

| Backend | Literal accuracy | Paraphrase accuracy | Gap |
|---------|------------------|---------------------|-----|
| **B0** | 100% | 0% | 100 pts |
| B1 | 37.5% | 12.5% | 25 pts |
| **B2** | 62.5% | **62.5%** | 0 pts |

B2 (few-shot) beats B1 on paraphrase but still below the 90% gate. **LoRA go/no-go: yes** — large B0 paraphrase gap and no backend ≥90% on paraphrase.

```bash
pytest
llm4osc score                         # full literal + refusal (B0 gates)
llm4osc score --suite paraphrase      # paraphrase only
llm4osc score-compare --backends b0,b1,b2   # Track C (requires [llm] for B1/B2)
python benchmarks/score_track_c.py
```

Full reports: `benchmarks/results/baseline.json`, `benchmarks/results/track_c.json`.

## Layout

| Path | Purpose |
|------|---------|
| `llm4osc/` | CLI, NL resolver, optional Qwen path |
| `tier3/` | Validate → clamp → encode → send |
| `profiles/committed/` | Versioned device patterns |
| `benchmarks/` | Golden tests, paraphrase suite, scorecards |
| `schemas/` | Intent and profile JSON Schema |

## License

MIT
