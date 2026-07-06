# Qwen2-0.5B OSC LoRA adapter

Fine-tuned **Qwen/Qwen2-0.5B-Instruct** for NL → validated intent JSON on the Max/MSP hero profile.

## Train

```bash
llm4osc train-data --device max-msp
python -m pip install -e ".[train]"
python training/train_lora.py
```

Artifacts land in `models/qwen2-0.5b-osc/adapter/` (weights gitignored; commit `model_card.json` after training).

## Infer (B3)

```bash
export LLM4OSC_ADAPTER=models/qwen2-0.5b-osc/adapter
llm4osc serve --adapter "$LLM4OSC_ADAPTER"
llm4osc send --device max-msp --nl "bring gain to fifty percent" --backend b3 --dry-run -y
llm4osc score-compare --backends b0,b3 --adapter "$LLM4OSC_ADAPTER"
```

## Gates

Same as Track C: paraphrase semantic accuracy ≥ 90%, wrong-send rate 0%.

Training data excludes frozen goldens in `benchmarks/golden_nl*`, `benchmarks/golden_refusal/`. Every label passes Tier 3 dry-run.
