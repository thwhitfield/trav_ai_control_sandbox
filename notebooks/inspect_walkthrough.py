"""Inspect walkthrough.

A tour of inspect_ai from the bottom up: models -> datasets -> tasks -> solvers
-> scorers -> tools -> agents -> state -> running many evals -> log analysis.

Run from the repo root:  uv run python notebooks/inspect_walkthrough.py
Comment out sections you don't want to re-run.

Docs home: https://inspect.aisi.org.uk/
After running, open the logs with `uv run inspect view --log-dir logs/walkthrough`
and match what you see there to the Python objects.
"""

# ==============================================================================
# 0. Setup
# ==============================================================================

import asyncio

from dotenv import (
    load_dotenv,  # Inspect also reads .env on its own; this just makes it explicit
)

load_dotenv()

MODEL = "anthropic/claude-haiku-5-5"  # cheap and fast; swap freely
LOG_DIR = "logs/walkthrough"  # keep this notebook's logs separate from the rest
LIMIT = 5  # samples per eval while learning

# ==============================================================================
# 1. Models on their own
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/models.html
#
# Before any eval machinery: `get_model()` returns a `Model`; `generate()` returns a
# `ModelOutput`. Every solver/agent ultimately calls this same API.

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)


async def model_basics():
    model = get_model(MODEL)
    out = await model.generate("What is the capital of Australia? One word.")

    print(out.completion)  # the text, convenience accessor
    print(out.choices[0].message)  # the full ChatMessageAssistant
    print(out.stop_reason)  # "stop", "max_tokens", "tool_calls", ...
    print(out.usage)  # input/output tokens

    # A list of messages instead of a string. This is what a conversation looks like
    # internally: state.messages in a TaskState is exactly this kind of list.
    messages = [
        ChatMessageSystem(content="You are a pirate. Answer in one short sentence."),
        ChatMessageUser(content="What is 2 + 2?"),
        ChatMessageAssistant(content="Arr, that be 4, matey."),
        ChatMessageUser(content="And 3 + 3?"),
    ]
    out = await model.generate(messages)
    print(out.completion)

    # GenerateConfig: per-call, or baked into the model.
    out = await model.generate(
        "Write a haiku about evals.",
        config=GenerateConfig(temperature=1.0, max_tokens=60),
    )
    print(out.completion)

    hot_model = get_model(MODEL, config=GenerateConfig(temperature=1.0))  # noqa: F841
    # EXERCISE: call hot_model.generate on the same prompt 3 times. Do the outputs vary?


# Model calls are async. In a plain script, drive them with asyncio.run().
# (In Jupyter you'd write `await model_basics()` instead.)
asyncio.run(model_basics())

# ==============================================================================
# 2. Datasets and Sample
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/datasets.html
#
# A `Sample` is one test case: `input` (str or list of messages), `target` (str or
# list of str), optional `choices`, `id`, `metadata`. A `Dataset` is a sequence of them.

from inspect_ai.dataset import (
    FieldSpec,
    MemoryDataset,
    Sample,
    example_dataset,
    hf_dataset,
)

# A small hand-made dataset that is reused from here on, so each section changes
# only one thing.
qa_dataset = MemoryDataset(
    [
        Sample(
            id="cap-au", input="What is the capital of Australia?", target="Canberra"
        ),
        Sample(id="cap-ca", input="What is the capital of Canada?", target="Ottawa"),
        Sample(id="cap-ch", input="What is the capital of Switzerland?", target="Bern"),
        Sample(
            id="math-1",
            input="What is 17 * 23?",
            target="391",
            metadata={"kind": "math"},
        ),
        Sample(
            id="math-2",
            input="What is 2 ** 10?",
            target="1024",
            metadata={"kind": "math"},
        ),
    ],
    name="tiny_qa",
)
print(len(qa_dataset), qa_dataset[0])

# input can be a message list (e.g. a few-shot prompt baked into the sample)
chat_sample = Sample(
    input=[
        ChatMessageSystem(content="Answer with a single word."),
        ChatMessageUser(content="Opposite of hot?"),
    ],
    target="cold",
)

# Multiple choice: put options in `choices`, target is the letter.
mc_dataset = MemoryDataset(
    [
        Sample(
            input="Which planet is closest to the sun?",
            choices=["Venus", "Mercury", "Mars", "Earth"],
            target="B",
        ),
        Sample(
            input="Which gas do plants absorb for photosynthesis?",
            choices=["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"],
            target="C",
        ),
    ]
)

# Loading real data. Three styles:

# (a) Built-in example data (jsonl with input/target already named correctly)
security = example_dataset("security_guide")
print(security[0])

# (b) FieldSpec: map column names -> Sample fields declaratively
gsm8k_fieldspec = hf_dataset(
    "openai/gsm8k",
    name="main",
    split="test",
    sample_fields=FieldSpec(input="question", target="answer"),
    limit=20,
)


# (c) record_to_sample: arbitrary Python for anything a FieldSpec can't express.
#     gsm8k targets look like "...reasoning... #### 72"; keep only the number.
def gsm8k_record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["question"],
        target=record["answer"].split("####")[-1].strip(),
        metadata={"full_solution": record["answer"]},
    )


gsm8k = hf_dataset(
    "openai/gsm8k",
    name="main",
    split="test",
    sample_fields=gsm8k_record_to_sample,
    limit=20,
)
print(gsm8k_fieldspec[0].target[-40:], "|", gsm8k[0].target)

# Dataset ops
math_only = qa_dataset.filter(lambda s: (s.metadata or {}).get("kind") == "math")
print(len(math_only), [s.id for s in math_only])
print([s.id for s in qa_dataset[:2]])  # slicing returns a Dataset
# EXERCISE: try qa_dataset.shuffle(seed=0) and check the order of ids.

# ==============================================================================
# 3. The smallest possible eval
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/tasks.html , https://inspect.aisi.org.uk/eval-logs.html
#
# Task = dataset + solver + scorer. `eval()` runs it and returns a **list** of EvalLog
# (one per task x model).

from inspect_ai import Task, eval, task
from inspect_ai.scorer import includes, match
from inspect_ai.solver import generate

logs = eval(
    Task(dataset=qa_dataset, solver=generate(), scorer=includes()),
    model=MODEL,
    log_dir=LOG_DIR,
)
log = logs[0]

print(log.status)  # "success" | "error" | "cancelled" | "started"
print(log.location)  # the .eval file on disk
print(log.results.scores[0].name, log.results.scores[0].metrics)
print(log.stats.model_usage)

s = log.samples[0]
print(s.id, "|", s.output.completion[:80], "|", s.scores)
for m in s.messages:
    print(type(m).__name__, ":", m.text[:80])


# The @task decorator registers a task by name and makes its params configurable
# (CLI: -T param=value, Python: task_args={...}).
@task
def tiny_qa(scorer_name: str = "includes"):
    return Task(
        dataset=qa_dataset,
        solver=generate(),
        scorer=includes() if scorer_name == "includes" else match(),
    )


logs = eval(tiny_qa(scorer_name="match"), model=MODEL, log_dir=LOG_DIR)
# EXERCISE: why does `match()` score worse than `includes()` here? Look at the
# completions. (Hint: match() checks the start/end of the answer by default.)

# You can also run tasks from files, exactly like the CLI does.
logs = eval(
    "scripts/security_guide_attack.py",
    task_args={"mode": "honest"},
    model=MODEL,
    limit=LIMIT,
    log_dir=LOG_DIR,
)

# ==============================================================================
# 4. Solvers and TaskState  (the most important section)
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/solvers.html
#
# A solver is `async def solve(state: TaskState, generate: Generate) -> TaskState`.
# `TaskState` holds the conversation (`messages`), the latest `output`, the
# sample's `metadata`, a per-sample `store`, and a `completed` flag.
# `generate(state)` = call the model on state.messages, append the reply, set state.output.

from inspect_ai.solver import (
    Generate,
    TaskState,
    chain,
    chain_of_thought,
    multiple_choice,
    prompt_template,
    self_critique,
    solver,
    system_message,
)

# Built-in solvers, one at a time. After each, open the sample in `inspect view`
# and look at exactly what each solver did to the messages.
solver_variants = {
    "plain": [generate()],
    "system_message": [
        system_message("Answer in as few words as possible."),
        generate(),
    ],
    "prompt_template": [
        prompt_template("Question: {prompt}\nGive only the answer."),
        generate(),
    ],
    "chain_of_thought": [chain_of_thought(), generate()],
    "self_critique": [generate(), self_critique()],
}
for name, solvers in solver_variants.items():
    eval(
        Task(
            dataset=qa_dataset, solver=solvers, scorer=includes(), name=f"solver_{name}"
        ),
        model=MODEL,
        log_dir=LOG_DIR,
    )

# Multiple choice: the multiple_choice() solver formats choices + calls generate
# itself, and pairs with the choice() scorer.
from inspect_ai.scorer import choice

eval(
    Task(dataset=mc_dataset, solver=multiple_choice(), scorer=choice()),
    model=MODEL,
    log_dir=LOG_DIR,
)

# chain() turns several solvers into one reusable solver.
concise_cot = chain(system_message("Be concise."), chain_of_thought(), generate())


# Custom solver #1: edit the prompt before generating.
@solver
def add_hint(hint: str = "Think carefully before answering."):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = f"{state.user_prompt.text}\n\n{hint}"
        return await generate(state)

    return solve


# Custom solver #2: call the model yourself (not via `generate`) and keep the
# shorter of two answers. Note that get_model() with no args = the eval's model.
@solver
def best_of_two():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        a = await model.generate(state.messages)
        b = await model.generate(state.messages)
        chosen = a if len(a.completion) <= len(b.completion) else b
        state.messages.append(chosen.message)
        state.output = chosen
        state.metadata["candidates"] = [a.completion, b.completion]
        return state

    return solve


# Custom solver #3: a second model critiques the first one's answer.
@solver
def critic(critic_model: str = MODEL):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)  # first draft from the eval's model
        review = await get_model(critic_model).generate(
            f"Question: {state.input_text}\nAnswer: {state.output.completion}\n"
            "Is this answer correct? Reply in one sentence."
        )
        state.messages.append(
            ChatMessageUser(
                content=f"A reviewer said: {review.completion}\nGive your final answer."
            )
        )
        return await generate(state)

    return solve


for s in [add_hint(), best_of_two(), critic()]:
    eval(
        Task(dataset=qa_dataset, solver=s, scorer=includes()),
        model=MODEL,
        log_dir=LOG_DIR,
    )

# EXERCISE: write a solver that sets `state.completed = True` for samples whose
# metadata kind == "math" without calling generate. Put generate() after it.
# What happens to those samples? Does the scorer still run?

# ==============================================================================
# 5. Scorers and metrics
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/scorers.html
#
# A scorer is `async def score(state: TaskState, target: Target) -> Score`.
# Metrics (accuracy, mean, stderr, ...) aggregate Scores across samples.

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    SampleScore,
    Score,
    Target,
    accuracy,
    exact,
    f1,
    mean,
    metric,
    model_graded_fact,
    pattern,
    scorer,
    stderr,
)

# Several scorers on the same task: one log, one result per scorer.
# Look for samples where they disagree.
logs = eval(
    Task(
        dataset=qa_dataset,
        solver=[system_message("Answer in a full sentence."), generate()],
        scorer=[includes(), match(), exact(), f1(), model_graded_fact()],
    ),
    model=MODEL,
    log_dir=LOG_DIR,
)
for r in logs[0].results.scores:
    print(r.name, {k: round(v.value, 3) for k, v in r.metrics.items()})

# pattern() extracts the answer with a regex first. Pairs with chain_of_thought's
# "ANSWER: ..." format.
eval(
    Task(
        dataset=qa_dataset,
        solver=[chain_of_thought(), generate()],
        scorer=pattern(r"ANSWER:\s*(.+)"),
    ),
    model=MODEL,
    log_dir=LOG_DIR,
)


# Custom metric: receives every SampleScore, returns one number.
@metric
def fraction_perfect():
    def compute(scores: list[SampleScore]) -> float:
        return sum(s.score.as_float() == 1.0 for s in scores) / len(scores)

    return compute


# Custom scorer #1: CORRECT/INCORRECT plus an explanation (shown in the viewer).
@scorer(metrics=[accuracy(), stderr(), fraction_perfect()])
def short_and_right(max_words: int = 5):
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        right = target.text.lower() in answer.lower()
        short = len(answer.split()) <= max_words
        return Score(
            value=CORRECT if (right and short) else INCORRECT,
            answer=answer,
            explanation=f"right={right}, words={len(answer.split())}",
        )

    return score


# Custom scorer #2: a dict value -> each key gets its own metrics.
@scorer(metrics={"correct": [accuracy()], "length": [mean()]})
def correct_and_length():
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        return Score(
            value={
                "correct": (
                    CORRECT if target.text.lower() in answer.lower() else INCORRECT
                ),
                "length": len(answer.split()),
            },
            answer=answer,
        )

    return score


logs = eval(
    Task(
        dataset=qa_dataset,
        solver=generate(),
        scorer=[short_and_right(), correct_and_length()],
        # Task(metrics=[...]) would override the metrics of *every* scorer here, which
        # would break on correct_and_length's dict value. So attach metrics per scorer.
    ),
    model=MODEL,
    log_dir=LOG_DIR,
)
for r in logs[0].results.scores:
    print(r.name, {k: round(v.value, 3) for k, v in r.metrics.items()})

# Epochs: run each sample N times, then reduce per sample ("mean", "max", "mode",
# "median", "at_least_2", "pass_at_2", ...) before the metrics.
from inspect_ai import Epochs

eval(
    Task(
        dataset=qa_dataset,
        solver=generate(),
        scorer=includes(),
        epochs=Epochs(3, ["mean", "max"]),
    ),
    model=MODEL,
    log_dir=LOG_DIR,
    temperature=1.0,  # GenerateConfig fields can be passed straight to eval()
)

# Re-score an existing log without calling the solver model again.
from inspect_ai import score

rescored = score(logs[0], scorers=match())
print([r.name for r in rescored.results.scores])
# CLI equivalent: uv run inspect score <log file> --scorer ...

# ==============================================================================
# 6. Tools
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/tools.html , https://inspect.aisi.org.uk/tools-custom.html
#
# A tool is an async function the model can call. The **type hints and docstring
# become the JSON schema the model sees**, so they matter.

from inspect_ai.solver import use_tools
from inspect_ai.tool import bash, python, tool


@tool
def multiply():
    async def execute(x: float, y: float) -> float:
        """Multiply two numbers.

        Args:
            x: First number.
            y: Second number.

        Returns:
            The product x * y.
        """
        return x * y

    return execute


arith = MemoryDataset(
    [
        Sample(input="Use the tool to compute 123.5 * 47.", target="5804.5"),
        Sample(input="Use the tool to compute 991 * 1013.", target="1003883"),
    ]
)

# use_tools() makes tools available; generate() runs the tool loop until the model
# stops calling tools. Read the log: assistant(tool_call) -> tool -> assistant.
eval(
    Task(dataset=arith, solver=[use_tools(multiply()), generate()], scorer=includes()),
    model=MODEL,
    log_dir=LOG_DIR,
)
# EXERCISE: try use_tools(multiply(), tool_choice="none"). What changes?

# Built-in tools that run code need a sandbox. "local" runs on your machine
# (fine for this harmless task, use "docker" for anything untrusted).
from inspect_ai.util import sandbox


@tool
def count_lines():
    async def execute(path: str) -> str:
        """Count lines in a file.

        Args:
            path: Path to the file.
        """
        result = await sandbox().exec(
            ["wc", "-l", path]
        )  # sandbox() = this sample's sandbox
        return result.stdout if result.success else result.stderr

    return execute


eval(
    Task(
        dataset=[
            Sample(input="How many lines are in pyproject.toml? Use a tool.", target="")
        ],
        solver=[use_tools(count_lines(), python()), generate()],
        sandbox="local",
    ),
    model=MODEL,
    log_dir=LOG_DIR,
)

# ==============================================================================
# 7. Agents
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/agents.html , https://inspect.aisi.org.uk/react-agent.html ,
# https://inspect.aisi.org.uk/agent-custom.html
#
# An agent is `async def execute(state: AgentState) -> AgentState`. Like a solver,
# but it only sees messages + output (no dataset target etc.), which makes agents
# composable: usable as a Task solver, as a tool, or as a handoff target.

from inspect_ai.agent import AgentState, agent, as_tool, handoff, react
from inspect_ai.model import execute_tools

# react(): the batteries-included loop, with a submit() tool and retry-on-wrong attempts.
eval(
    Task(
        dataset=arith,
        solver=react(
            prompt="Solve the problem, then submit the number.",
            tools=[multiply()],
            attempts=2,
        ),
        scorer=includes(),
        message_limit=20,  # limits stop runaway agents: also token_limit, time_limit, ...
    ),
    model=MODEL,
    log_dir=LOG_DIR,
)


# A minimal custom agent: this is roughly what react() / generate() do under the hood.
@agent
def tool_loop(tools, max_turns: int = 5):
    async def execute(state: AgentState) -> AgentState:
        model = get_model()
        for _ in range(max_turns):
            state.output = await model.generate(state.messages, tools=tools)
            state.messages.append(state.output.message)
            if not state.output.message.tool_calls:
                break  # model answered without calling a tool -> done
            result = await execute_tools(state.messages, tools)
            state.messages.extend(result.messages)
        return state

    return execute


eval(
    Task(dataset=arith, solver=tool_loop(tools=[multiply()]), scorer=includes()),
    model=MODEL,
    log_dir=LOG_DIR,
)

# Composition: an agent can be used as a tool (as_tool) or handed control (handoff).
math_agent = react(
    name="math_agent",
    description="Expert at arithmetic. Give it a calculation.",
    tools=[multiply()],
)
supervisor = react(
    prompt="Delegate arithmetic to the math agent.", tools=[handoff(math_agent)]
)
eval(
    Task(dataset=arith, solver=supervisor, scorer=includes()),
    model=MODEL,
    log_dir=LOG_DIR,
)
# EXERCISE: swap handoff(math_agent) for as_tool(math_agent). Compare the
# transcripts: who sees which messages?

# ==============================================================================
# 8. State across steps: metadata vs store
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/agents.html (Store section), https://inspect.aisi.org.uk/solvers.html
#
# `state.metadata` is the sample's metadata (fine for simple notes).
# The `store` is per-sample scratch space, recorded in the log with every change.
# `StoreModel` gives it a typed, pydantic interface.

from inspect_ai.util import StoreModel, store_as
from pydantic import Field


class Attempts(StoreModel):
    drafts: list[str] = Field(default_factory=list)


@solver
def draft_twice():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        attempts = store_as(Attempts)
        for _ in range(2):
            out = await get_model().generate(state.messages)
            attempts.drafts.append(out.completion)
        return await generate(state)

    return solve


@scorer(metrics=[mean()])
def drafts_agree():
    async def score(state: TaskState, target: Target) -> Score:
        drafts = state.store_as(Attempts).drafts  # read what the solver wrote
        return Score(value=float(len(set(drafts)) == 1), explanation=str(drafts))

    return score


eval(
    Task(dataset=qa_dataset, solver=draft_twice(), scorer=[includes(), drafts_agree()]),
    model=MODEL,
    log_dir=LOG_DIR,
    temperature=1.0,
)

# ==============================================================================
# 9. Running many evals
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/eval-sets.html , https://inspect.aisi.org.uk/models.html (Model Roles)

# Many tasks x many models in one call -> one log per (task, model).
logs = eval(
    [tiny_qa(), tiny_qa(scorer_name="match")],
    model=[MODEL, "anthropic/claude-sonnet-5-5"],
    log_dir=LOG_DIR,
)
print(len(logs))

# Model roles: name a role in code, bind it to a model at eval time.
# model_graded_fact() uses the "grader" role if one is bound (else the eval's model).
eval(
    Task(dataset=qa_dataset, solver=generate(), scorer=model_graded_fact()),
    model=MODEL,
    model_roles={"grader": "anthropic/claude-sonnet-5-5"},
    log_dir=LOG_DIR,
)
# In your own code: get_model(role="grader") or get_model(role="critic", default=MODEL)

# eval_set: idempotent + retrying. Re-run the same cell: completed work is skipped.
from inspect_ai import eval_retry, eval_set

success, logs = eval_set(
    [tiny_qa()],
    model=[MODEL],
    log_dir=f"{LOG_DIR}/set1",  # eval_set needs its own dir
)
print(success)

# eval_retry(log) re-runs only the failed/missing samples of an errored log.

# ==============================================================================
# 10. Log analysis in Python
# ==============================================================================
# Docs: https://inspect.aisi.org.uk/eval-logs.html , https://inspect.aisi.org.uk/dataframe.html

from inspect_ai.log import list_eval_logs, read_eval_log, read_eval_log_samples

log_files = list_eval_logs(LOG_DIR)
print(len(log_files))

# header_only=True skips samples: fast, even for huge logs (e.g. your SWE-bench runs)
header = read_eval_log(log_files[0], header_only=True)
print(
    header.eval.task,
    header.eval.model,
    header.results.scores[0].metrics if header.results else None,
)

# Stream samples one at a time instead of loading them all into memory
for sample in read_eval_log_samples(log_files[0]):
    print(sample.id, sample.scores, len(sample.events), "events")

# Dataframes: one row per eval / sample / message / event.
from inspect_ai.analysis import evals_df, messages_df, samples_df

evals = evals_df(LOG_DIR)
print(evals[["task_name", "model", "score_headline_value"]])

samples = samples_df(LOG_DIR)
print(samples.columns.tolist())

# Your earlier MMLU runs, compared across models:
mmlu = evals_df("logs")
mmlu = mmlu[mmlu.task_name.str.contains("mmlu")]
print(mmlu[["model", "score_headline_value", "total_samples"]])

# EXERCISE: with samples_df, find the samples where includes() and match() disagree
# in the multi-scorer log from section 5.
