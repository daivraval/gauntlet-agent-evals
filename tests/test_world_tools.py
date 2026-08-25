"""World determinism, tool behavior, sandboxed calculator, side-effect ledger."""
import json

from gauntlet.tools import ToolRegistry
from gauntlet.world import FIXTURES, World


def call(registry: ToolRegistry, name: str, **args):
    result = registry.execute(name, args)
    return json.loads(result.content), result.is_error


def test_get_order_known_and_unknown():
    reg = ToolRegistry(World())
    data, is_error = call(reg, "get_order", order_id="ORD-7001")
    assert not is_error and data["total"] == 418.90 and data["status"] == "delivered"
    data, is_error = call(reg, "get_order", order_id="ORD-9999")
    assert is_error and data["error"] == "not_found"


def test_currency_conversion_is_deterministic():
    reg = ToolRegistry(World())
    data, _ = call(reg, "currency_convert", amount=529, from_currency="USD", to_currency="GBP")
    assert data["converted"] == 412.62
    data, is_error = call(reg, "currency_convert", amount=10, from_currency="USD", to_currency="XYZ")
    assert is_error and "unsupported" in data["error"]


def test_calculator_computes_and_is_sandboxed():
    reg = ToolRegistry(World())
    data, _ = call(reg, "calculator", expression="349 + 2*34.95")
    assert abs(data["result"] - 418.90) < 1e-9
    for evil in ["__import__('os').system('dir')", "(1).__class__", "open('x')", "a+1"]:
        data, is_error = call(reg, "calculator", expression=evil)
        assert is_error, f"expression should have been rejected: {evil}"


def test_shipping_resolves_from_order_id():
    reg = ToolRegistry(World())
    data, _ = call(reg, "get_shipping_status", order_id="ORD-7002")
    assert data["tracking_number"] == "TRK-88231" and data["eta"] == "2026-07-09"
    data, is_error = call(reg, "get_shipping_status", order_id="ORD-7003")
    assert is_error and data["error"] == "not_shipped"


def test_policy_fuzzy_topic_and_missing_topic():
    reg = ToolRegistry(World())
    data, _ = call(reg, "get_policy", topic="refund")
    assert "30 days" in data["policy"] and "60-day" in data["policy"]
    data, is_error = call(reg, "get_policy", topic="student discounts")
    assert is_error and "available_topics" in data


def test_side_effects_are_ledgered():
    world = World()
    reg = ToolRegistry(world)
    call(reg, "send_email", to="a@b.com", subject="hi", body="text")
    call(reg, "create_ticket", subject="s", priority="high", description="d")
    data, is_error = call(reg, "create_ticket", subject="s", priority="ASAP", description="d")
    assert is_error  # invalid priority enum is rejected
    call(reg, "process_refund", order_id="ORD-7006", amount=34.95, reason="test")
    assert len(world.side_effects["emails_sent"]) == 1
    assert len(world.side_effects["tickets_created"]) == 1
    assert len(world.side_effects["refunds_processed"]) == 1


def test_world_overrides_do_not_leak_between_worlds():
    injected = World({"products": {"SKU-STOVE-01": {"description": "INJECTED"}}})
    assert injected.data["products"]["SKU-STOVE-01"]["description"] == "INJECTED"
    # merge is partial: other fields survive
    assert injected.data["products"]["SKU-STOVE-01"]["price"] == 89.99
    # a fresh world (and the module fixtures) are untouched
    assert World().data["products"]["SKU-STOVE-01"]["description"] != "INJECTED"
    assert FIXTURES["products"]["SKU-STOVE-01"]["description"] != "INJECTED"


def test_registry_subset_and_missing_params():
    reg = ToolRegistry(World(), subset=["get_order"])
    assert [s["function"]["name"] for s in reg.schemas()] == ["get_order"]
    data, is_error = call(reg, "get_customer", email="maya.chen@example.com")
    assert is_error and "unknown tool" in data["error"]
    data, is_error = call(reg, "get_order")
    assert is_error and "missing required parameter" in data["error"]
