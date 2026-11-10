"""The smallest possible Inspect task: one sample, plain generation, exact-match scoring.

Run it:
    uv run inspect eval tasks/hello_world.py --model anthropic/claude-haiku-4-5
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task
def hello_world():
    return Task(
        dataset=[Sample(input="Just reply with Hello World", target="Hello World")],
        solver=[generate()],
        scorer=exact(),
    )
