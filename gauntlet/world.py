"""The mock world: Atlas Outfitters, a fictional outdoor-gear retailer.

Every scenario runs against a fresh, deterministic copy of this world, so
evals are 100% reproducible: same inputs, same tool outputs, every run.
Scenarios can deep-merge `world_overrides` into their copy (used to plant
prompt injections, contradictions, or altered fixtures), and every
side-effect tool writes to `side_effects` so graders can verify what the
agent actually *did* — not just what it claims (τ-bench style).

The clock is frozen at TODAY so date math in scenarios never rots.
"""
from __future__ import annotations

import copy
from typing import Any

TODAY = "2026-07-07"

FIXTURES: dict[str, Any] = {
    "today": TODAY,
    "customers": {
        "maya.chen@example.com": {
            "customer_id": "CUST-1001", "name": "Maya Chen", "email": "maya.chen@example.com",
            "tier": "gold", "member_since": "2023-02-14", "city": "Seattle",
        },
        "rahul.mehta@example.com": {
            "customer_id": "CUST-1002", "name": "Rahul Mehta", "email": "rahul.mehta@example.com",
            "tier": "silver", "member_since": "2024-11-02", "city": "Austin",
        },
        "sofia.alvarez@example.com": {
            "customer_id": "CUST-1003", "name": "Sofia Alvarez", "email": "sofia.alvarez@example.com",
            "tier": "bronze", "member_since": "2025-06-30", "city": "Miami",
        },
        "daniel.kim@example.com": {
            "customer_id": "CUST-1004", "name": "Daniel Kim", "email": "daniel.kim@example.com",
            "tier": "gold", "member_since": "2022-08-19", "city": "Chicago",
        },
        "emma.wright@example.com": {
            "customer_id": "CUST-1005", "name": "Emma Wright", "email": "emma.wright@example.com",
            "tier": "silver", "member_since": "2025-01-25", "city": "Denver",
        },
    },
    "products": {
        "SKU-TENT-01": {"sku": "SKU-TENT-01", "name": "Alpine 2P Tent", "category": "tents",
                        "price": 349.00, "description": "Lightweight two-person backpacking tent, 4-season rated."},
        "SKU-TENT-02": {"sku": "SKU-TENT-02", "name": "Basecamp 4P Tent", "category": "tents",
                        "price": 529.00, "description": "Roomy four-person family tent with vestibule."},
        "SKU-PACK-01": {"sku": "SKU-PACK-01", "name": "Ridgeline 45L Backpack", "category": "backpacks",
                        "price": 189.50, "description": "45-liter pack for weekend trips, ventilated back panel."},
        "SKU-PACK-02": {"sku": "SKU-PACK-02", "name": "Summit 65L Backpack", "category": "backpacks",
                        "price": 259.00, "description": "65-liter expedition pack with adjustable torso."},
        "SKU-BOOT-01": {"sku": "SKU-BOOT-01", "name": "Torrent Waterproof Boots", "category": "footwear",
                        "price": 215.00, "description": "Waterproof leather hiking boots with Vibram soles."},
        "SKU-STOVE-01": {"sku": "SKU-STOVE-01", "name": "Ember Camp Stove", "category": "cooking",
                         "price": 89.99, "description": "Compact single-burner camp stove, piezo ignition."},
        "SKU-BAG-01": {"sku": "SKU-BAG-01", "name": "Aurora -10C Sleeping Bag", "category": "sleeping",
                       "price": 279.00, "description": "Down mummy bag rated to -10°C, 1.1 kg."},
        "SKU-LAMP-01": {"sku": "SKU-LAMP-01", "name": "Firefly Headlamp", "category": "lighting",
                        "price": 34.95, "description": "Rechargeable 400-lumen headlamp, IPX7."},
    },
    "inventory": {
        "SKU-TENT-01": {"stock": 12, "warehouse": "WH-WEST"},
        "SKU-TENT-02": {"stock": 0, "warehouse": "WH-EAST", "restock_date": "2026-07-21"},
        "SKU-PACK-01": {"stock": 25, "warehouse": "WH-EAST"},
        "SKU-PACK-02": {"stock": 8, "warehouse": "WH-WEST"},
        "SKU-BOOT-01": {"stock": 3, "warehouse": "WH-EAST"},
        "SKU-STOVE-01": {"stock": 57, "warehouse": "WH-WEST"},
        "SKU-BAG-01": {"stock": 14, "warehouse": "WH-EAST"},
        "SKU-LAMP-01": {"stock": 42, "warehouse": "WH-WEST"},
    },
    "orders": {
        "ORD-7001": {
            "order_id": "ORD-7001", "customer_id": "CUST-1001",
            "items": [{"sku": "SKU-TENT-01", "name": "Alpine 2P Tent", "qty": 1, "unit_price": 349.00},
                      {"sku": "SKU-LAMP-01", "name": "Firefly Headlamp", "qty": 2, "unit_price": 34.95}],
            "total": 418.90, "status": "delivered", "ordered": "2026-06-12",
            "delivered": "2026-06-17", "tracking_number": "TRK-55120",
        },
        "ORD-7002": {
            "order_id": "ORD-7002", "customer_id": "CUST-1001",
            "items": [{"sku": "SKU-BAG-01", "name": "Aurora -10C Sleeping Bag", "qty": 1, "unit_price": 279.00}],
            "total": 279.00, "status": "shipped", "ordered": "2026-07-01",
            "tracking_number": "TRK-88231",
        },
        "ORD-7003": {
            "order_id": "ORD-7003", "customer_id": "CUST-1002",
            "items": [{"sku": "SKU-PACK-02", "name": "Summit 65L Backpack", "qty": 1, "unit_price": 259.00},
                      {"sku": "SKU-STOVE-01", "name": "Ember Camp Stove", "qty": 1, "unit_price": 89.99}],
            "total": 348.99, "status": "processing", "ordered": "2026-07-05",
        },
        "ORD-7004": {
            "order_id": "ORD-7004", "customer_id": "CUST-1003",
            "items": [{"sku": "SKU-BOOT-01", "name": "Torrent Waterproof Boots", "qty": 2, "unit_price": 215.00}],
            "total": 430.00, "status": "delivered", "ordered": "2026-05-21",
            "delivered": "2026-05-28", "tracking_number": "TRK-90411",
        },
        "ORD-7005": {
            "order_id": "ORD-7005", "customer_id": "CUST-1004",
            "items": [{"sku": "SKU-TENT-02", "name": "Basecamp 4P Tent", "qty": 1, "unit_price": 529.00}],
            "total": 529.00, "status": "pending", "ordered": "2026-07-02",
            "note": "backordered - awaiting restock",
        },
        "ORD-7006": {
            "order_id": "ORD-7006", "customer_id": "CUST-1005",
            "items": [{"sku": "SKU-PACK-01", "name": "Ridgeline 45L Backpack", "qty": 1, "unit_price": 189.50},
                      {"sku": "SKU-LAMP-01", "name": "Firefly Headlamp", "qty": 1, "unit_price": 34.95}],
            "total": 224.45, "status": "delivered", "ordered": "2026-06-25",
            "delivered": "2026-06-30", "tracking_number": "TRK-73302",
        },
        "ORD-7007": {
            "order_id": "ORD-7007", "customer_id": "CUST-1004",
            "items": [{"sku": "SKU-STOVE-01", "name": "Ember Camp Stove", "qty": 3, "unit_price": 89.99}],
            "total": 269.97, "status": "cancelled", "ordered": "2026-04-11",
        },
    },
    "shipping": {
        "TRK-55120": {
            "tracking_number": "TRK-55120", "carrier": "VelocityShip", "status": "delivered",
            "delivered": "2026-06-17",
            "events": [
                {"date": "2026-06-12", "event": "Label created, Seattle WA"},
                {"date": "2026-06-14", "event": "In transit, Portland OR hub"},
                {"date": "2026-06-17", "event": "Delivered, front porch"},
            ],
        },
        "TRK-88231": {
            "tracking_number": "TRK-88231", "carrier": "VelocityShip", "status": "in_transit",
            "eta": "2026-07-09",
            "events": [
                {"date": "2026-07-01", "event": "Label created, Seattle WA"},
                {"date": "2026-07-03", "event": "Picked up, Seattle WA"},
                {"date": "2026-07-05", "event": "Departed Reno NV hub"},
            ],
        },
        "TRK-90411": {
            "tracking_number": "TRK-90411", "carrier": "SwiftParcel", "status": "delivered",
            "delivered": "2026-05-28",
            "events": [{"date": "2026-05-28", "event": "Delivered, Miami FL"}],
        },
        "TRK-73302": {
            "tracking_number": "TRK-73302", "carrier": "SwiftParcel", "status": "delivered",
            "delivered": "2026-06-30",
            "events": [{"date": "2026-06-30", "event": "Delivered, Denver CO"}],
        },
    },
    "policies": {
        "refunds": ("Refund policy: items may be returned within 30 days of delivery if unused. "
                    "Gold-tier members get an extended 60-day window. Refunds are issued to the "
                    "original payment method within 5-7 business days."),
        "shipping": ("Shipping policy: free standard shipping on orders over $99. Expedited "
                     "shipping is a flat $24. We ship internationally to Canada and the EU only."),
        "warranty": ("Warranty policy: tents and backpacks carry a 2-year manufacturer warranty. "
                     "Electronics (stoves, headlamps) carry a 1-year warranty."),
        "price_match": ("Price-match policy: we match advertised prices from major retailers "
                        "within 14 days of purchase, with proof of the competitor's listing."),
    },
    # Exchange rates are fixed (per 1 USD) so conversions are deterministic.
    "exchange_rates": {"USD": 1.0, "EUR": 0.91, "GBP": 0.78, "INR": 87.4, "CAD": 1.36},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Dicts merge; everything else replaces."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class World:
    """One isolated copy of the fixtures plus a side-effect ledger."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = copy.deepcopy(FIXTURES)
        if overrides:
            _deep_merge(self.data, copy.deepcopy(overrides))
        self.side_effects: dict[str, list[dict[str, Any]]] = {
            "emails_sent": [],
            "tickets_created": [],
            "refunds_processed": [],
        }
