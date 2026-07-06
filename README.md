# LLM4OSC

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

Frozen Max/MSP suite: 8 literal + 8 paraphrase + 3 refusal.  
Gates: semantic accuracy ≥ 90%, **wrong-send rate 0%**.

| Backend | Literal | Paraphrase | Wrong-send (para) | p50      |
|---------|---------|------------|-------------------|----------|
| **B0**  | 100%    | 100%       | 0%                | ~0.05 ms |
| B1      | 37.5%   | 12.5%      | 37.5%             | ~3 s     |
| B2      | 62.5%   | 62.5%      | 12.5%             | ~3 s     |
| B3      | 100%†   | 100%       | 0%                | ~4 s     |

†B3 literal+refusal: refusal recall gaps on frozen cases — use B0 in production.

```bash
pytest
llm4osc score
llm4osc score --suite paraphrase
llm4osc score-compare --backends b0,b1,b2,b3 --adapter models/qwen2-0.5b-osc/adapter
```

[`benchmarks/results/track_c.json`](benchmarks/results/track_c.json) · [`models/qwen2-0.5b-osc/model_card.md`](models/qwen2-0.5b-osc/model_card.md)

### Train B3 (optional)

Weights gitignored — reproduce locally:

```bash
llm4osc train-data --device max-msp
pip install -e ".[train]" && python training/train_lora.py
```

## Limitations

- One device profile (Max/MSP, 12 patterns) — other rigs need new profiles
- Small benchmark; passing gates ≠ show-day coverage
- B1/B2 unsafe without Tier 3; B3 needs local training, ~3–4 s inference
- Text NL only — no speech, MCP server, or automated manual ingest

## Layout

`llm4osc/` · `tier3/` · `profiles/committed/` · `benchmarks/` · `schemas/` · `training/`

## License

MIT
