"""Fault injection semantics: transient vs permanent, and the sneaky modes
(garbage/empty) that succeed at the transport level but carry bad payloads."""
import json

from gauntlet.faults import FaultingToolExecutor
from gauntlet.schemas import FaultMode, FaultSpec
from gauntlet.tools import ToolRegistry
from gauntlet.world import World


def make_executor(*faults: FaultSpec) -> FaultingToolExecutor:
    return FaultingToolExecutor(ToolRegistry(World()), list(faults))


def test_transient_fault_fails_n_times_then_recovers():
    ex = make_executor(FaultSpec(tool="get_order", mode=FaultMode.ERROR, times=2))
    for _ in range(2):
        result, faulted = ex.execute("get_order", {"order_id": "ORD-7001"})
        assert faulted and result.is_error and "503" in result.content
    result, faulted = ex.execute("get_order", {"order_id": "ORD-7001"})
    assert not faulted and not result.is_error
    assert json.loads(result.content)["total"] == 418.90


def test_permanent_fault_never_recovers():
    ex = make_executor(FaultSpec(tool="get_policy", mode=FaultMode.TIMEOUT, always=True))
    for _ in range(5):
        result, faulted = ex.execute("get_policy", {"topic": "refunds"})
        assert faulted and result.is_error


def test_garbage_is_not_flagged_as_error():
    ex = make_executor(FaultSpec(tool="get_customer", mode=FaultMode.GARBAGE, times=1))
    result, faulted = ex.execute("get_customer", {"email": "maya.chen@example.com"})
    assert faulted and not result.is_error
    try:
        json.loads(result.content)
        raise AssertionError("garbage payload should not be valid JSON")
    except json.JSONDecodeError:
        pass


def test_empty_fault_is_wellformed_but_empty():
    ex = make_executor(FaultSpec(tool="search_products", mode=FaultMode.EMPTY, times=1))
    result, faulted = ex.execute("search_products", {"query": "tent"})
    assert faulted and not result.is_error
    assert json.loads(result.content)["results"] == []
    # second call sees the real catalog again
    result, faulted = ex.execute("search_products", {"query": "tent"})
    assert not faulted and json.loads(result.content)["results"]


def test_faults_only_hit_their_target_tool():
    ex = make_executor(FaultSpec(tool="get_order", mode=FaultMode.ERROR, always=True))
    result, faulted = ex.execute("get_customer", {"email": "maya.chen@example.com"})
    assert not faulted and not result.is_error
