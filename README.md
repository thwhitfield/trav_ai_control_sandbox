# trav-ai-control-sandbox

A sandbox for experimenting with AI control.

## Current focus

Starting with a refresher on [Inspect](https://inspect.aisi.org.uk/), working through the basics before building anything control-specific.

## Setup

```bash
uv sync
uv run inspect --version
```

Run an eval with:

```bash
uv run inspect eval path/to/task.py --model <provider/model>
```

Logs are written to `logs/` (gitignored). View them with `uv run inspect view`.
