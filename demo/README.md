# LLM4OSC demo UI

```bash
pip install -e ".[demo]"          # add ,[llm] for B1–B3
python demo/app.py
# or: llm4osc demo
```

Open http://127.0.0.1:7860

Optional warm LLM:

```bash
llm4osc serve --adapter models/qwen2-0.5b-osc/adapter
export LLM4OSC_SERVE_URL=http://127.0.0.1:8765
```

## Approved UI v2 (current)

Accepted as of 2026-08-14. Snapshot: `demo/app.v2.py`. Restore with:

```bash
cp demo/app.v2.py demo/app.py
```

Layout (top → bottom):

- Header: title + dry-run note
- Row: Profile (wider) · Backend (What/When in menu labels) · Retrieval gate
- Query row: Natural language panel (label + example chips on one line; inner text field
  matching OSC-preview block chrome) · Resolve (same height as panel)
- Status HTML
- Row: OSC preview · Retrieval scores
- Intent JSON

Behavior: always dry-run (no UDP); B0 ignores gate; B1–B3 honor gate.

## Default UI v1 (earlier revert baseline)

Earlier accepted default. Snapshot: `demo/app.default.py`. Restore with:

```bash
cp demo/app.default.py demo/app.py
```

Layout: Backend What/When HTML table; separate Examples block; Intent JSON below outputs.
