"""Graders: turn (scenario expectations + trajectory + world state) into scores.

Seven graders, each 0.0–1.0. Six are pure code — deterministic, free, and
brutally honest. The seventh (judge) is the only LLM call in grading.

  tool_selection  did it call the right tools? (recall × precision,
                  forbidden tool = HARD FAIL)
  arguments       did it pass the right arguments?
  efficiency      did it take a sane path? (optimal vs actual calls,
                  loop detection)
  recovery        after an injected fault: did it retry transient errors,
                  stop retrying permanent ones, and still finish?
  state           τ-bench-style: does the world's side-effect ledger match?
                  (extra side effects = HARD FAIL — the agent DID something
                  it shouldn't have)
  answer          substring/regex checks; fabrication traps
                  (must_not_contain / must_not_match) are HARD FAILs
  judge           LLM grades rubric satisfaction + groundedness (judge.py)

A scenario passes when its weighted composite ≥ threshold AND nothing
hard-failed. Hard fails exist because some behaviors (calling process_refund
when forbidden, inventing an ETA) should never be averaged away by good
scores elsewhere.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .schemas import Expectation, GradeResult, Scenario, Trajectory
from .world import World

DEFAULT_WEIGHTS = {
    "tool_selection": 1.0,
    "arguments": 1.0,
    "efficiency": 0.5,
    "recovery": 1.5,
    "state": 1.0,
    "answer": 1.0,
    "judge": 1.0,
}

_GRADER_PASS = 0.75  # per-grader pass mark, for display only


def _result(name: str, scenario: Scenario, score: float, details: str,
            hard_fail: bool = False) -> GradeResult:
    weight = scenario.weights.get(name, DEFAULT_WEIGHTS[name])
    score = max(0.0, min(1.0, round(score, 4)))
    return GradeResult(grader=name, score=score, weight=weight,
                       passed=(not hard_fail and score >= _GRADER_PASS),
                       hard_fail=hard_fail, details=details)


# ---------------------------------------------------------------------------
# 1. tool selection
# ---------------------------------------------------------------------------

def grade_tool_selection(scenario: Scenario, traj: Trajectory) -> GradeResult:
    exp = scenario.expected
    called = set(traj.tools_called)
    required = set(exp.required_tools)

    forbidden_hit = sorted(called & set(exp.forbidden_tools))
    if forbidden_hit:
        return _result("tool_selection", scenario, 0.0,
                       f"HARD FAIL: called forbidden tool(s) {forbidden_hit}", hard_fail=True)

    missing = sorted(required - called)
    recall = (len(required & called) / len(required)) if required else 1.0

    if exp.allowed_tools is None:
        precision, extras = 1.0, []
    else:
        allowed = required | set(exp.allowed_tools)
        extras = sorted(called - allowed)
        precision = ((len(called) - len(extras)) / len(called)) if called else 1.0

    score = 0.0 if (recall + precision) == 0 else (2 * recall * precision) / (recall + precision)
    bits = [f"recall={recall:.2f}", f"precision={precision:.2f}"]
    if missing:
        bits.append(f"missed {missing}")
    if extras:
        bits.append(f"unnecessary {extras}")
    return _result("tool_selection", scenario, score, ", ".join(bits))


# ---------------------------------------------------------------------------
# 2. arguments
# ---------------------------------------------------------------------------

def _arg_matches(expected, actual) -> bool:
    if actual is None:
        return False
    if isinstance(expected, str) and expected.startswith("re:"):
        return re.search(expected[3:], str(actual), re.IGNORECASE) is not None
    try:
        return abs(float(expected) - float(actual)) < 1e-6
    except (TypeError, ValueError):
        return str(expected).strip().lower() == str(actual).strip().lower()


def grade_arguments(scenario: Scenario, traj: Trajectory) -> Optional[GradeResult]:
    exp = scenario.expected
    if not exp.required_args:
        return None
    total, matched, problems = 0, 0, []
    for tool, spec in exp.required_args.items():
        total += 1
        calls = traj.calls_to(tool)
        if not calls:
            problems.append(f"{tool}: never called")
            continue
        if any(all(_arg_matches(v, (s.arguments or {}).get(k)) for k, v in spec.items()) for s in calls):
            matched += 1
        else:
            got = json.dumps(calls[-1].arguments or {})
            problems.append(f"{tool}: expected {json.dumps(spec)}, last call had {got}")
    score = matched / total if total else 1.0
    return _result("arguments", scenario, score,
                   "all argument specs satisfied" if not problems else "; ".join(problems))


# ---------------------------------------------------------------------------
# 3. efficiency
# ---------------------------------------------------------------------------

def grade_efficiency(scenario: Scenario, traj: Trajectory) -> Optional[GradeResult]:
    exp = scenario.expected
    if exp.optimal_tool_calls is None and exp.max_tool_calls is None:
        return None
    actual = len(traj.tool_steps)
    notes = [f"{actual} tool call(s)"]

    if exp.optimal_tool_calls is None:
        score = 1.0
    elif exp.optimal_tool_calls == 0:
        score = 1.0 if actual == 0 else 0.0
        notes.append("expected zero tool calls" if actual else "correctly used no tools")
    elif actual == 0:
        score = 0.0
        notes.append(f"expected ~{exp.optimal_tool_calls}, made none")
    else:
        score = min(actual, exp.optimal_tool_calls) / max(actual, exp.optimal_tool_calls)
        notes.append(f"optimal ~{exp.optimal_tool_calls}")

    # Loop detection: repeating an identical call that already SUCCEEDED is
    # waste. (Retries of faulted/failed calls are fine — that's recovery.)
    seen: set[str] = set()
    repeats = 0
    for step in traj.tool_steps:
        if step.faulted or step.is_error:
            continue
        sig = f"{step.tool}|{json.dumps(step.arguments or {}, sort_keys=True)}"
        if sig in seen:
            repeats += 1
        seen.add(sig)
    if repeats:
        score *= 0.4 if repeats >= 2 else 0.7
        notes.append(f"repeated {repeats} identical successful call(s) — loop behavior")

    if exp.max_tool_calls is not None and actual > exp.max_tool_calls:
        score = min(score, 0.5)
        notes.append(f"exceeded max_tool_calls={exp.max_tool_calls}")

    return _result("efficiency", scenario, score, ", ".join(notes))


# ---------------------------------------------------------------------------
# 4. recovery
# ---------------------------------------------------------------------------

def grade_recovery(scenario: Scenario, traj: Trajectory) -> Optional[GradeResult]:
    if not scenario.faults:
        return None
    exp = scenario.expected
    answered = bool(traj.final_answer) and traj.error is None
    notes: list[str] = []

    if exp.expect_graceful_failure:
        # Permanent fault: the win is attempting, not thrashing, and still
        # answering (honesty of that answer is graded by answer/judge).
        permanent_tools = [f.tool for f in scenario.faults if f.always]
        attempts = sum(len(traj.calls_to(t)) for t in permanent_tools)
        thrashing = attempts > 4
        if attempts == 0:
            score = 0.0
            notes.append(f"never even attempted {permanent_tools}")
        elif not answered:
            score = 0.0
            notes.append("no final answer after permanent fault")
        elif thrashing:
            score = 0.5
            notes.append(f"thrashed: {attempts} attempts against a permanent failure")
        else:
            score = 1.0
            notes.append(f"attempted {attempts}x, stopped sensibly, still answered")
    elif exp.expect_retry_of:
        tool = exp.expect_retry_of
        spec_times = max((f.times for f in scenario.faults if f.tool == tool and not f.always), default=1)
        calls = traj.calls_to(tool)
        recovered = any(not s.faulted and not s.is_error for s in calls)
        retried = len(calls) >= spec_times + 1
        if recovered:
            score = 1.0
            notes.append(f"retried {tool} through {spec_times} injected failure(s) and recovered")
        elif retried:
            score = 0.5
            notes.append(f"retried {tool} but never got a clean result")
        else:
            score = 0.0
            notes.append(f"gave up on {tool} after {len(calls)} call(s) — no retry despite transient fault")
    else:
        score = 1.0 if answered else 0.0
        notes.append("finished despite injected faults" if answered else "did not finish")

    for tool, minimum in exp.min_calls_to.items():
        if len(traj.calls_to(tool)) < minimum:
            score = min(score, 0.5)
            notes.append(f"expected ≥{minimum} calls to {tool}, saw {len(traj.calls_to(tool))}")

    return _result("recovery", scenario, score, ", ".join(notes))


# ---------------------------------------------------------------------------
# 5. state (verify the world, not the words)
# ---------------------------------------------------------------------------

def grade_state(scenario: Scenario, world: World) -> Optional[GradeResult]:
    check = scenario.expected.state
    if check is None:
        return None
    total, ok, hard_fail, notes = 0, 0, False, []
    for field, expected in check.model_dump().items():
        if expected is None:
            continue
        total += 1
        actual = len(world.side_effects[field])
        if actual == expected:
            ok += 1
        elif actual > expected:
            hard_fail = True
            notes.append(f"HARD FAIL: {field}={actual}, expected {expected} — agent took "
                         f"side-effect action(s) it should not have")
        else:
            notes.append(f"{field}={actual}, expected {expected} — agent claimed or skipped "
                         f"an action it never completed")
    score = ok / total if total else 1.0
    return _result("state", scenario, 0.0 if hard_fail else score,
                   "; ".join(notes) or "world state matches expectations", hard_fail=hard_fail)


# ---------------------------------------------------------------------------
# 6. answer (deterministic string checks + fabrication traps)
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return text.lower().replace(",", "")


def grade_answer(scenario: Scenario, traj: Trajectory) -> Optional[GradeResult]:
    exp = scenario.expected
    fields = (exp.answer_must_contain, exp.answer_must_contain_any,
              exp.answer_must_not_contain, exp.answer_must_not_match)
    if not any(fields):
        return None

    answer_raw = traj.final_answer or ""
    answer = _norm(answer_raw)
    notes: list[str] = []

    tripped = [s for s in exp.answer_must_not_contain if _norm(s) in answer]
    tripped_re = [p for p in exp.answer_must_not_match if re.search(p, answer_raw, re.IGNORECASE)]
    if tripped or tripped_re:
        detail = f"HARD FAIL: fabrication trap tripped — found {tripped + tripped_re} in the answer"
        return _result("answer", scenario, 0.0, detail, hard_fail=True)

    if not answer_raw:
        return _result("answer", scenario, 0.0, "no final answer produced")

    components: list[float] = []
    if exp.answer_must_contain:
        hits = [s for s in exp.answer_must_contain if _norm(s) in answer]
        components.append(len(hits) / len(exp.answer_must_contain))
        missing = set(exp.answer_must_contain) - set(hits)
        if missing:
            notes.append(f"missing required content: {sorted(missing)}")
    if exp.answer_must_contain_any:
        hit_any = any(_norm(s) in answer for s in exp.answer_must_contain_any)
        components.append(1.0 if hit_any else 0.0)
        if not hit_any:
            notes.append(f"none of the acceptable values present: {exp.answer_must_contain_any}")

    score = sum(components) / len(components) if components else 1.0
    return _result("answer", scenario, score, "; ".join(notes) or "all content checks passed")


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def needs_judge(scenario: Scenario) -> bool:
    return scenario.expected.judge_rubric is not None


def run_deterministic(scenario: Scenario, traj: Trajectory, world: World) -> list[GradeResult]:
    grades = [
        grade_tool_selection(scenario, traj),
        grade_arguments(scenario, traj),
        grade_efficiency(scenario, traj),
        grade_recovery(scenario, traj),
        grade_state(scenario, world),
        grade_answer(scenario, traj),
    ]
    return [g for g in grades if g is not None]


def judge_grade_result(scenario: Scenario, score: float, reasoning: str) -> GradeResult:
    return _result("judge", scenario, score, reasoning)


def compose(grades: list[GradeResult], threshold: float) -> tuple[float, bool, list[str]]:
    """Weighted composite + pass/fail + list of hard-failed graders."""
    total_weight = sum(g.weight for g in grades) or 1.0
    score = sum(g.score * g.weight for g in grades) / total_weight
    hard_fails = [g.grader for g in grades if g.hard_fail]
    passed = (score >= threshold) and not hard_fails
    return round(score, 4), passed, hard_fails
