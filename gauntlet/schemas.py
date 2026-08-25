"""Core data models for GAUNTLET.

Everything the harness consumes or produces is a typed pydantic model:
scenarios (YAML) in, trajectories + grades (JSON) out. Keeping every
contract in one module makes the saved artifacts self-documenting.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Scenario definitions (what we feed the agent)
# ---------------------------------------------------------------------------


class Category(str, Enum):
    TOOL_SELECTION = "tool_selection"
    ARGUMENTS = "arguments"
    ERROR_RECOVERY = "error_recovery"
    HALLUCINATION = "hallucination"
    MULTI_STEP = "multi_step"
    ADVERSARIAL = "adversarial"


class FaultMode(str, Enum):
    ERROR = "error"            # tool returns an HTTP-500-style failure
    TIMEOUT = "timeout"        # tool returns a timeout error
    GARBAGE = "garbage"        # tool returns corrupted, non-JSON output
    EMPTY = "empty"            # tool returns a well-formed but empty result
    RATE_LIMIT = "rate_limit"  # tool returns an HTTP-429-style failure


class FaultSpec(BaseModel):
    """One injected failure. `times: 2` means the first two calls to the
    tool fail; `always: true` means every call fails no matter what."""

    tool: str
    mode: FaultMode
    times: int = 1
    always: bool = False
    message: str = ""


class StateCheck(BaseModel):
    """Assertions on world side effects after the run (τ-bench style:
    don't trust what the agent *says* it did — check the database)."""

    emails_sent: Optional[int] = None
    tickets_created: Optional[int] = None
    refunds_processed: Optional[int] = None


class Expectation(BaseModel):
    """Ground truth for grading one scenario.

    Tool expectations:
      - required_tools  : must all be called at least once (recall)
      - allowed_tools   : extra tools that don't hurt precision.
                          None  -> any extra tool is fine (precision not graded)
                          []    -> strict: anything outside required_tools hurts
      - forbidden_tools : calling one of these is a HARD FAIL
      - required_args   : {tool: {arg: expected}}. Values compare loosely
                          (case-insensitive string / float). Prefix "re:" for regex.

    Trajectory expectations:
      - optimal_tool_calls : ideal number of tool calls (efficiency = optimal/actual)
      - max_tool_calls     : exceeding this caps the efficiency score
      - expect_retry_of    : tool the agent should retry after an injected fault
      - expect_graceful_failure : the fault is permanent; agent must stop
                                  retrying and answer honestly
      - min_calls_to       : {tool: n} minimum number of calls expected

    Answer expectations:
      - answer_must_contain      : ALL substrings must appear (case-insensitive)
      - answer_must_contain_any  : at least ONE must appear
      - answer_must_not_contain  : any appearing is a HARD FAIL (fabrication traps)
      - answer_must_not_match    : regexes; any match is a HARD FAIL
      - ground_truth / judge_rubric : inputs for the LLM judge
    """

    required_tools: list[str] = Field(default_factory=list)
    allowed_tools: Optional[list[str]] = None
    forbidden_tools: list[str] = Field(default_factory=list)
    required_args: dict[str, dict[str, Any]] = Field(default_factory=dict)

    optimal_tool_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None
    expect_retry_of: Optional[str] = None
    expect_graceful_failure: bool = False
    min_calls_to: dict[str, int] = Field(default_factory=dict)

    answer_must_contain: list[str] = Field(default_factory=list)
    answer_must_contain_any: list[str] = Field(default_factory=list)
    answer_must_not_contain: list[str] = Field(default_factory=list)
    answer_must_not_match: list[str] = Field(default_factory=list)
    ground_truth: Optional[str] = None
    judge_rubric: Optional[str] = None
    state: Optional[StateCheck] = None


class Scenario(BaseModel):
    id: str
    name: str
    category: Category
    prompt: str
    # Which tools the agent is offered: "all" or an explicit subset.
    tools: Union[Literal["all"], list[str]] = "all"
    # Deep-merged into a fresh copy of the world (for injections, contradictions).
    world_overrides: dict[str, Any] = Field(default_factory=dict)
    faults: list[FaultSpec] = Field(default_factory=list)
    expected: Expectation
    # Optional per-scenario grader weight overrides, e.g. {"recovery": 2.0}.
    weights: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Trajectory (what the agent actually did, step by step)
# ---------------------------------------------------------------------------


class TrajectoryStep(BaseModel):
    """One event in the agent's run. kind:
    - "llm"         : a model call (assistant text and/or tool call requests)
    - "tool"        : one tool execution with its result
    - "answer"      : the final answer
    - "agent_error" : the agent itself crashed or was cut off
    """

    index: int
    kind: Literal["llm", "tool", "answer", "agent_error"]
    tool: Optional[str] = None
    arguments: Optional[dict[str, Any]] = None
    result: Optional[str] = None
    is_error: bool = False        # tool returned an error payload
    faulted: bool = False         # error was injected by the harness
    assistant_text: Optional[str] = None
    tool_calls_requested: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Trajectory(BaseModel):
    scenario_id: str
    agent: str
    model: str = ""
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_answer: str = ""
    error: Optional[str] = None   # fatal agent error, if any

    # -- convenience accessors used by graders --------------------------------
    @property
    def tool_steps(self) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.kind == "tool"]

    @property
    def tools_called(self) -> list[str]:
        return [s.tool for s in self.tool_steps if s.tool]

    def calls_to(self, tool: str) -> list[TrajectoryStep]:
        return [s for s in self.tool_steps if s.tool == tool]

    @property
    def llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "llm")

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(s.prompt_tokens for s in self.steps),
            sum(s.completion_tokens for s in self.steps),
        )


class TrajectoryRecorder:
    """Collects steps while an agent runs. Passed into every agent."""

    def __init__(self, scenario_id: str, agent: str, model: str = "") -> None:
        self.trajectory = Trajectory(scenario_id=scenario_id, agent=agent, model=model)

    def add(self, **kwargs: Any) -> TrajectoryStep:
        step = TrajectoryStep(index=len(self.trajectory.steps), **kwargs)
        self.trajectory.steps.append(step)
        return step

    def finish(self, answer: str) -> None:
        self.trajectory.final_answer = answer
        self.add(kind="answer", result=answer)

    def fail(self, error: str) -> None:
        self.trajectory.error = error
        self.add(kind="agent_error", result=error, is_error=True)


# ---------------------------------------------------------------------------
# Grades and results (what the harness decided)
# ---------------------------------------------------------------------------


class GradeResult(BaseModel):
    grader: str
    score: float                  # 0.0 .. 1.0
    weight: float
    passed: bool
    hard_fail: bool = False       # forbidden tool, fabrication, safety violation
    details: str = ""


class ScenarioResult(BaseModel):
    scenario_id: str
    name: str
    category: Category
    agent: str
    trial: int = 1
    grades: list[GradeResult] = Field(default_factory=list)
    score: float = 0.0            # weighted composite, 0..1
    passed: bool = False
    hard_fails: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    trajectory: Optional[Trajectory] = None


class RunReport(BaseModel):
    run_id: str
    started_at: str
    agent: str
    model: str
    judge_model: str = ""
    judge_enabled: bool = True
    pass_threshold: float = 0.7
    trials: int = 1
    results: list[ScenarioResult] = Field(default_factory=list)
    wall_time_s: float = 0.0
