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

## Principles

- **Deterministic** — reproducible control messages  
- **Local** — no cloud required at runtime  
- **Profile-bound** — only verified patterns from device manuals  

## License

MIT 
