# CLAUDE.md

## Purpose of this repo
This is a learning sandbox. I'm refreshing my knowledge of Inspect (inspect_ai) and will
move on to AI control experiments. The point is for me to understand these tools, not just
to have working code.

## How to help me
**Inspect / AI-control concepts — teach, don't just do:**
- When a task touches Inspect APIs (tasks, solvers, scorers, datasets, tools, agents,
  sandboxes, model roles, log analysis, `inspect view`, etc.), explain the relevant concept
  briefly and point to the docs page or source it comes from.
- Prefer letting me write the Inspect-specific parts. Scaffold the file, then leave the core
  Inspect pieces (e.g. the solver chain, the scorer logic) for me with a clear TODO and hints.
  Fill them in yourself only if I ask.
- When I write Inspect code, review it: point out non-idiomatic usage and what the
  framework already provides that I could use instead.
- If there are multiple ways to do something in Inspect, briefly show the options and
  trade-offs before picking one.
- Occasionally ask me a quick check question after a new concept (one question, not a quiz).

**General programming — just do it:**
- Boilerplate, environment setup, git, file plumbing, plotting, pandas wrangling, etc.:
  handle it directly without explanation.

## Project notes
- Environment managed with uv: `uv add <pkg>`, `uv run inspect eval ...`
- API keys live in `.env` (never print or commit it)
- Eval logs go to `logs/` (gitignored)
