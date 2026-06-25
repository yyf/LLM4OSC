# LLM4OSC

Turn natural language into **validated Open Sound Control (OSC)** — locally, deterministically, and reliably.

Inspired by [MCP2OSC](https://arxiv.org/html/2508.10414v1) (NeurIPS 2025 Creative AI Track).

## How it works

```
device profile  +  natural language
        ↓
   intent JSON  (proposed)
        ↓
   validate → encode → OSC
```

Models and rules **propose** structured intents; deterministic code **validates and sends**. Same validated intent → same OSC bytes. Ambiguous requests are refused, not guessed.

## Install

```bash
cd LLM4OSC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Without activating the venv:

```bash
python3 -m pip install -e ".[dev]"
# or: pip3 install -e ".[dev]"
```

## Quick start

```bash
# List patterns in the Max/MSP hero profile
llm4osc profile list-patterns --device max-msp

# Natural language → preview → send (dry run, no UDP)
llm4osc send --device max-msp --nl "set gain to 50%" --dry-run -y

# Live send (Max udpreceive on port 7400, or scripts/osc_listen.py)
llm4osc send --device max-msp --nl "set gain to 50%" -y

# B0 benchmark scorecard
llm4osc score --write benchmarks/results/baseline.json
```

Set `LLM4OSC_HOST` and `LLM4OSC_PORT` (default `127.0.0.1:7400`) for live UDP send.

Listen for packets:

```bash
python scripts/osc_listen.py
```

## Tests

```bash
pytest
```

B0 gates (0 wrong sends, ≥90% semantic accuracy on NL goldens):

```bash
llm4osc score
```

## License

MIT
