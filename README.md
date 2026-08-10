# LLM4OSC

<img width="960" height="526" alt="demo" src="https://github.com/user-attachments/assets/bfc4f260-8ab8-4a30-8347-0c0b8b24b7a3" />

<br>

[![arXiv](https://img.shields.io/badge/arXiv-2607.26024-b31b1b.svg)](https://arxiv.org/abs/2607.26024)

<br>

**Don't let the LLM touch the wire.**

Natural language → structured intent → validated OSC. Models *propose* JSON over a versioned device profile; deterministic code *decides* what hits UDP. Same validated intent → same OSC bytes. Out-of-profile input is refused, not guessed.

Follow-on to [MCP2OSC](https://github.com/yyf/MCP2OSC) (NeurIPS 2025 Creative AI Track).

```
device profile  +  natural language
        ↓
   intent JSON  (proposed)
        ↓
   validate → clamp → encode → OSC
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

LLM backends (`b1`–`b3`): `pip install -e ".[dev,llm]"`

## Quick start

```bash
llm4osc send --device max-msp --nl "set gain to 50%" --dry-run -y      # literal
llm4osc send --device max-msp --nl "make the level half" --dry-run -y   # paraphrase → /gain 0.5
llm4osc send --device max-msp --nl "boost the bass band by 3db" --dry-run -y  # refuses
llm4osc send --device max-msp --nl "set gain to 50%" -y                 # live → 127.0.0.1:7400
```

Max/MSP: `[udpreceive 7400]`. `LLM4OSC_HOST` / `LLM4OSC_PORT` to override.

```bash
python scripts/osc_listen.py
llm4osc profile list-patterns --device max-msp
```

Default backend: **B0** (rules + slot fill, no GPU).

## Backends

| Backend | What                             | When                          |
|---------|----------------------------------|-------------------------------|
| **b0**  | Profile retrieval + slot parsing | Live control, demos (default)   |
| b1      | Qwen2-0.5B zero-shot             | Baseline only                 |
| b2      | Qwen + few-shot                  | Baseline only                 |
| b3      | Qwen + LoRA + NL refine          | Paraphrase experiments        |

`--backend b0|b1|b2|b3` · `LLM4OSC_DEBUG=1` · `LLM4OSC_MODEL` · `LLM4OSC_ADAPTER`

```bash
llm4osc serve                                    # load Qwen once
export LLM4OSC_SERVE_URL=http://127.0.0.1:8765   # reuse in other shells
```

## Evaluation

Frozen Max/MSP holdout suite (excluded from LoRA training): **8 literal + 8 paraphrase + 4 refusal** on a 12-pattern hero profile.

Primary safety metric: **wrong-send rate** — mismatched intents that would pass Tier 3 dry-run and transmit OSC. Missed commands (safe refusals) do not count.

Release gates: semantic accuracy ≥ 90%, **wrong-send rate 0%**. CI enforces B0 on literal + refusal (`pytest`, `llm4osc score`).

### Track C (2026-07-07) — with retrieval gate

B1–B3 re-apply B0’s refuse policy after the model proposes JSON (zero retrieval score / ambiguous tie / unfillable slot → refuse). Without that gate, raw B1/B2 wrong-sent ~12–38% on paraphrase.

| Backend | Literal | Paraphrase | Wrong-send (para) | Refusal recall (lit) | p50 (para) |
|---------|---------|------------|-------------------|----------------------|------------|
| **B0**  | 100%    | 100%       | 0%                | 100%                 | ~0.06 ms   |
| B1      | 100%    | 100%       | 0%                | 100%                 | ~3.9 s     |
| B2      | 100%    | 100%       | 0%                | 100%                 | ~3.8 s     |
| B3      | 100%    | 100%       | 0%                | 100%                 | ~3.8 s     |

Use **B0** for live control (sub-ms, no GPU). B1–B3 gating clears the suite but does not prove the model understood the phrase — the gate carries refusal. Scorecards: [`benchmarks/results/track_c.json`](benchmarks/results/track_c.json). B3 recipe: [`models/qwen2-0.5b-osc/model_card.md`](models/qwen2-0.5b-osc/model_card.md).

```bash
pytest
llm4osc score
llm4osc score --suite paraphrase
llm4osc score-compare --backends b0,b1,b2,b3 --adapter models/qwen2-0.5b-osc/adapter
```

### Train B3 (optional)

Weights gitignored — reproduce locally:

```bash
llm4osc train-data --device max-msp
pip install -e ".[train]" && python training/train_lora.py
```

## Limitations

- One device profile (Max/MSP, 12 patterns) — other rigs need new profiles
- Small benchmark (20 NL-facing cases); passing gates ≠ show-day coverage
- B1–B3 unsafe without retrieval gate + Tier 3; LLM path ~3–4 s p50
- Text NL only — no speech, MCP server, or automated manual ingest

## Layout

`llm4osc/` · `tier3/` · `profiles/committed/` · `benchmarks/` · `schemas/` · `training/`

## License

MIT
