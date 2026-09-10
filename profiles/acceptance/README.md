# Profile Acceptance suites

Per-device acceptance goldens (must-work / must-refuse) tied to a `profile_version`.

```
profiles/acceptance/<device_id>/
  meta.json
  golden_nl/
  golden_refusal/
  last_report.json   # gitignored
```

Seed Max from benchmarks:

```bash
llm4osc acceptance ensure --device max-msp
llm4osc acceptance run --device max-msp
```

See `docs/internal/profile-acceptance-eval-dev-plan.md`.
