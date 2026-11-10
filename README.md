# trav_ai_control_sandbox

My sandbox for learning [Inspect](https://inspect.aisi.org.uk/) and running AI control
experiments (eventually with [ControlArena](https://github.com/UKGovernmentBEIS/control-arena)).

## Setup

```bash
make install          # uv sync, including dev tools
cp .env.example .env  # then fill in API keys
```

Inspect reads `.env` automatically. Useful keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
and optionally `INSPECT_EVAL_MODEL` to set a default model.

## Running

```bash
uv run inspect eval tasks/hello_world.py --model anthropic/claude-haiku-4-5
make view             # browse results in ./logs
```

## Layout

| Path       | What                                                           |
|------------|----------------------------------------------------------------|
| `tasks/`   | `@task` definitions, one per file                              |
| `notes/`   | Learning notes                                                 |
| `logs/`    | Eval logs (gitignored)                                         |
| `archive/` | First round of work (Jan 2026); also tagged `archive/v1-jan2026` |

Shared solvers/scorers can move into their own module once something is reused across tasks.

## Learning path

1. Tasks, samples, `generate()`, basic scorers; `inspect eval` and `inspect view`
2. Datasets (`hf_dataset`, `csv_dataset`, `FieldSpec`/`record_to_sample`)
3. Solvers: prompt templates, chain-of-thought, multiple choice, custom `@solver`
4. Scorers: `match`, `includes`, `model_graded_qa`, custom `@scorer`, metrics
5. Tools and agents (`use_tools`, `react()`), then sandboxes (Docker)
6. `eval_set`, log analysis (`inspect_ai.analysis`), `inspect-evals`
7. ControlArena: settings, policies, monitors, micro-protocols
