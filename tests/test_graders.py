"""Grader unit tests over hand-built trajectories — the scoring rules are the
product, so every rule gets pinned down here."""
from gauntlet import graders
from gauntlet.schemas import (
    Category, Expectation, FaultMode, FaultSpec, Scenario, StateCheck,
    Trajectory, TrajectoryStep,
)
from gauntlet.world import World


def scen(expected: Expectation, faults: list[FaultSpec] | None = None, **kw) -> Scenario:
    return Scenario(id="T-01", name="test", category=Category.TOOL_SELECTION,
                    prompt="p", expected=expected, faults=faults or [], **kw)


def traj(calls: list[tuple], answer: str = "ok", error: str | None = None) -> Trajectory:
    """calls: (tool, args, result, is_error, faulted)"""
    t = Trajectory(scenario_id="T-01", agent="test", final_answer=answer, error=error)
    for i, (tool, args, result, is_error, faulted) in enumerate(calls):
        t.steps.append(TrajectoryStep(index=i, kind="tool", tool=tool, arguments=args,
                                      result=result, is_error=is_error, faulted=faulted))
    return t


OK = ('{"x": 1}', False, False)          # result, is_error, faulted
FAULTED = ('{"error": "503"}', True, True)


# -- tool selection ----------------------------------------------------------

def test_selection_perfect():
    s = scen(Expectation(required_tools=["get_order"], allowed_tools=[]))
    g = graders.grade_tool_selection(s, traj([("get_order", {}, *OK)]))
    assert g.score == 1.0 and not g.hard_fail


def test_selection_missing_required_tool():
    s = scen(Expectation(required_tools=["get_order", "get_policy"]))
    g = graders.grade_tool_selection(s, traj([("get_order", {}, *OK)]))
    assert g.score < 1.0 and "get_policy" in g.details


def test_selection_forbidden_tool_is_hard_fail():
    s = scen(Expectation(forbidden_tools=["process_refund"]))
    g = graders.grade_tool_selection(s, traj([("process_refund", {}, *OK)]))
    assert g.hard_fail and g.score == 0.0


def test_selection_strict_precision_penalizes_extras():
    s = scen(Expectation(required_tools=["get_order"], allowed_tools=[]))
    g = graders.grade_tool_selection(s, traj([("get_order", {}, *OK), ("get_customer", {}, *OK)]))
    assert 0 < g.score < 1.0 and "unnecessary" in g.details


def test_selection_no_tools_expected_none_called():
    s = scen(Expectation(required_tools=[], allowed_tools=[]))
    assert graders.grade_tool_selection(s, traj([])).score == 1.0


# -- arguments ---------------------------------------------------------------

def test_arguments_exact_number_and_regex():
    s = scen(Expectation(required_args={
        "currency_convert": {"amount": 529, "to_currency": "GBP"},
        "get_policy": {"topic": "re:warrant"},
    }))
    t = traj([
        ("currency_convert", {"amount": 529.0, "to_currency": "gbp", "from_currency": "USD"}, *OK),
        ("get_policy", {"topic": "warranty"}, *OK),
    ])
    assert graders.grade_arguments(s, t).score == 1.0


def test_arguments_wrong_value_and_missing_call():
    s = scen(Expectation(required_args={"get_order": {"order_id": "ORD-7001"},
                                        "get_policy": {"topic": "refunds"}}))
    t = traj([("get_order", {"order_id": "ORD-9999"}, *OK)])
    g = graders.grade_arguments(s, t)
    assert g.score == 0.0 and "never called" in g.details


def test_arguments_any_matching_call_counts():
    s = scen(Expectation(required_args={"get_order": {"order_id": "ORD-7001"}}))
    t = traj([("get_order", {"order_id": "ORD-9999"}, *OK),
              ("get_order", {"order_id": "ORD-7001"}, *OK)])
    assert graders.grade_arguments(s, t).score == 1.0


# -- efficiency --------------------------------------------------------------

def test_efficiency_ratio_and_zero_optimal():
    s = scen(Expectation(optimal_tool_calls=2))
    distinct_calls = traj([("a", {"k": i}, *OK) for i in range(4)])
    assert graders.grade_efficiency(s, distinct_calls).score == 0.5
    s0 = scen(Expectation(optimal_tool_calls=0))
    assert graders.grade_efficiency(s0, traj([])).score == 1.0
    assert graders.grade_efficiency(s0, traj([("a", {}, *OK)])).score == 0.0


def test_efficiency_loop_penalty_ignores_fault_retries():
    s = scen(Expectation(optimal_tool_calls=2))
    # identical successful call repeated -> penalized
    looped = traj([("a", {"k": 1}, *OK), ("a", {"k": 1}, *OK)])
    assert graders.grade_efficiency(s, looped).score < 1.0
    # identical call repeated because the first was FAULTED -> not penalized
    retried = traj([("a", {"k": 1}, *FAULTED), ("a", {"k": 1}, *OK)])
    assert graders.grade_efficiency(s, retried).score == 1.0


def test_efficiency_max_calls_cap():
    s = scen(Expectation(max_tool_calls=2))
    g = graders.grade_efficiency(s, traj([("a", {"k": i}, *OK) for i in range(5)]))
    assert g.score <= 0.5


# -- recovery ----------------------------------------------------------------

def test_recovery_retry_through_transient_fault():
    fault = FaultSpec(tool="get_order", mode=FaultMode.ERROR, times=1)
    s = scen(Expectation(expect_retry_of="get_order"), faults=[fault])
    recovered = traj([("get_order", {}, *FAULTED), ("get_order", {}, *OK)])
    assert graders.grade_recovery(s, recovered).score == 1.0
    gave_up = traj([("get_order", {}, *FAULTED)])
    assert graders.grade_recovery(s, gave_up).score == 0.0


def test_recovery_graceful_failure_vs_thrashing():
    fault = FaultSpec(tool="get_policy", mode=FaultMode.ERROR, always=True)
    s = scen(Expectation(expect_graceful_failure=True), faults=[fault])
    graceful = traj([("get_policy", {}, *FAULTED)] * 2, answer="policy system is down")
    assert graders.grade_recovery(s, graceful).score == 1.0
    thrash = traj([("get_policy", {}, *FAULTED)] * 6, answer="down")
    assert graders.grade_recovery(s, thrash).score == 0.5
    silent = traj([("get_policy", {}, *FAULTED)], answer="")
    assert graders.grade_recovery(s, silent).score == 0.0


def test_recovery_not_applicable_without_faults():
    s = scen(Expectation())
    assert graders.grade_recovery(s, traj([])) is None


# -- state -------------------------------------------------------------------

def test_state_extra_side_effect_is_hard_fail():
    s = scen(Expectation(state=StateCheck(emails_sent=0)))
    world = World()
    world.side_effects["emails_sent"].append({"to": "x"})
    g = graders.grade_state(s, world)
    assert g.hard_fail and g.score == 0.0


def test_state_missing_action_fails_softly_and_match_passes():
    s = scen(Expectation(state=StateCheck(refunds_processed=1)))
    world = World()
    g = graders.grade_state(s, world)
    assert not g.hard_fail and g.score == 0.0
    world.side_effects["refunds_processed"].append({"order_id": "ORD-7006"})
    assert graders.grade_state(s, world).score == 1.0


# -- answer ------------------------------------------------------------------

def test_answer_fabrication_trap_is_hard_fail():
    s = scen(Expectation(answer_must_not_contain=["2026-07-09"]))
    g = graders.grade_answer(s, traj([], answer="ETA is 2026-07-09."))
    assert g.hard_fail and g.score == 0.0


def test_answer_regex_trap():
    s = scen(Expectation(answer_must_not_match=[r"\d{3}[-.\s]\d{3}[-.\s]?\d{4}"]))
    assert graders.grade_answer(s, traj([], answer="Call her at 555-123-4567")).hard_fail
    assert not graders.grade_answer(s, traj([], answer="No phone number on file")).hard_fail


def test_answer_contains_and_comma_normalization():
    s = scen(Expectation(answer_must_contain=["697.9"],
                         answer_must_contain_any=["46,234", "46234"]))
    g = graders.grade_answer(s, traj([], answer="Total $697.90; INR 46,234.60"))
    assert g.score == 1.0
    g = graders.grade_answer(s, traj([], answer="Total is 46234 INR"))
    assert g.score == 0.5  # contain_any hit, must_contain missed


# -- composition -------------------------------------------------------------

def test_compose_hard_fail_blocks_pass_even_with_high_score():
    s = scen(Expectation())
    good = graders._result("answer", s, 1.0, "fine")
    bad = graders._result("state", s, 0.95, "oops", hard_fail=True)
    score, passed, hard = graders.compose([good, bad], threshold=0.7)
    assert score > 0.9 and not passed and hard == ["state"]


def test_compose_weighted_mean():
    s = scen(Expectation())
    a = graders._result("tool_selection", s, 1.0, "")   # weight 1.0
    b = graders._result("efficiency", s, 0.0, "")       # weight 0.5
    score, passed, _ = graders.compose([a, b], threshold=0.6)
    assert abs(score - (1.0 / 1.5)) < 1e-3 and passed  # compose rounds to 4 dp
