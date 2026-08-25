"""GAUNTLET — an evaluation harness that stress-tests tool-calling agents.

Feed it scenarios, it runs your agent through them, records every step of
the trajectory, injects tool failures, and grades the result on seven axes:
tool selection, arguments, efficiency, recovery, world-state integrity,
answer correctness, and rubric satisfaction (LLM-as-judge).
"""

__version__ = "1.0.0"
