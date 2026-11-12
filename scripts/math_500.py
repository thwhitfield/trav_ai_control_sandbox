import re

from inspect_ai.model import get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    AnswerPattern,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

# Grader prompt
EQUIVALENCE_TEMPLATE = """
Are these two expressions equivalent? Answer Yes or No without any additional symbols.

Expression 1: %(expression1)s
Expression 2: %(expression2)s
"""


@scorer(metrics=[accuracy(), stderr()])
def expression_equivalence():
    async def score(state: TaskState, target: Target):

        match = re.search(AnswerPattern.LINE)
