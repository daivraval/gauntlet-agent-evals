"""The scenario suite itself is under test: a silently-broken eval is worse
than no eval. Loading runs full validation (unknown tools, unoffered
required tools, retry expectations without matching faults, duplicate ids)."""
from collections import Counter

from gauntlet.loader import load_scenarios
from gauntlet.schemas import Category

EXPECTED_COUNTS = {
    Category.TOOL_SELECTION: 10,
    Category.ARGUMENTS: 8,
    Category.ERROR_RECOVERY: 10,
    Category.HALLUCINATION: 10,
    Category.MULTI_STEP: 7,
    Category.ADVERSARIAL: 5,
}


def test_exactly_50_scenarios_load_and_validate():
    scenarios = load_scenarios()
    assert len(scenarios) == 50


def test_category_distribution():
    counts = Counter(s.category for s in load_scenarios())
    assert counts == EXPECTED_COUNTS


def test_ids_are_unique_and_stable_format():
    ids = [s.id for s in load_scenarios()]
    assert len(ids) == len(set(ids))
    assert all("-" in i and i.split("-")[1].isdigit() for i in ids)


def test_every_scenario_grades_something():
    for s in load_scenarios():
        e = s.expected
        has_signal = any([
            e.required_tools, e.forbidden_tools, e.required_args,
            e.answer_must_contain, e.answer_must_contain_any,
            e.answer_must_not_contain, e.answer_must_not_match,
            e.judge_rubric, e.state is not None,
            e.optimal_tool_calls is not None, e.allowed_tools is not None,
        ])
        assert has_signal, f"{s.id} has no grading signal at all"


def test_filters():
    only_er = load_scenarios(categories=["error_recovery"])
    assert len(only_er) == 10
    one = load_scenarios(ids=["ts-01"])
    assert len(one) == 1 and one[0].id == "TS-01"
