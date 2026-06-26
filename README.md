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

## Benchmark results (Max/MSP hero profile)

Frozen suite: 8 NL + 3 refusal cases — see `benchmarks/golden_nl/` and `benchmarks/golden_refusal/`.

| Metric | B0 (rules) | B1 (Qwen2-0.5B) |
|--------|------------|-----------------|
| Semantic accuracy | **100%** | 25% |
| Wrong-send rate | **0%** | 9.1% |
| Refusal recall | **100%** | 0% |
| Latency p50 | **0.04 ms** | ~3.2 s |

B0 passes all gates on this suite. B1 is experimental — use **B0 for demos and live control** until fine-tuning or more eval.

```bash
pytest
llm4osc score                    # B0
llm4osc score --backend b1       # re-run B1 (requires [llm], model download)
```

Full scorecards: `benchmarks/results/baseline.json`, `benchmarks/results/b1.json`.

## Layout

| Path | Purpose |
|------|---------|
| `llm4osc/` | CLI, NL resolver, optional Qwen path |
| `tier3/` | Validate → clamp → encode → send |
| `profiles/committed/` | Versioned device patterns |
| `benchmarks/` | Golden tests + scorecards |
| `schemas/` | Intent and profile JSON Schema |

## License

MIT
