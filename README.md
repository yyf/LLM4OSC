# LLM4OSC

Turn natural language into **validated Open Sound Control (OSC)** — locally, deterministically, and reliably.

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

Optional **Qwen2-0.5B** local inference (~1 GB download on first run):

```bash
python -m pip install -e ".[dev,llm]"
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

# Qwen zero-shot (B1) or few-shot (B2) — still validated by Tier 3
llm4osc send --device max-msp --nl "set gain to 50%" --backend b1 --dry-run -y
llm4osc score --backend b1
```

Set `LLM4OSC_HOST` and `LLM4OSC_PORT` (default `127.0.0.1:7400`) for live UDP send.  
Set `LLM4OSC_MODEL` to override the default `Qwen/Qwen2-0.5B-Instruct`.  
Set `LLM4OSC_DEBUG=1` to print raw Qwen output on stderr when using `--backend b1` or `b2`.

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
