"""
Unit tests for the Quasi-Polynomial Local Search algorithm.

Tests use the restricted max-min allocation model:
each item j has a fixed size p_j, and each agent either values it at p_j or 0.

Programmer: Rotem Melamed
Date: 08/11/2025
"""

import logging

import pytest
import random
from itertools import product
from fairpyx import Instance
from fairpyx.algorithms.qp_local_search import (
    qp_max_min_allocation,
)


# ==============================================================================
#  Helper: brute-force optimal max-min value
# ==============================================================================

def brute_force_max_min(instance: Instance) -> float:
    """
    Compute the optimal max-min value by enumerating all possible
    assignments of items to agents.

    Each item is assigned to exactly one agent (or left unassigned).
    Returns the maximum over all partitions of the minimum agent value.

    Only feasible for small instances (complexity: (n+1)^m).
    """
    agents = list(instance.agents)
    items = list(instance.items)
    n = len(agents)
    m = len(items)

    best_min_value = 0

    # Each item can go to agent 0..n-1 or be unassigned (index n)
    for assignment in product(range(n + 1), repeat=m):
        agent_values = [0.0] * n
        for item_idx, agent_idx in enumerate(assignment):
            if agent_idx < n:
                agent_values[agent_idx] += instance.agent_item_value(
                    agents[agent_idx], items[item_idx]
                )

        min_value = min(agent_values)
        best_min_value = max(best_min_value, min_value)

    return best_min_value


def assert_approximation(instance: Instance, result: dict, epsilon: float, opt: float):
    """
    Assert that the allocation satisfies the approximation guarantee:
    every agent's value >= OPT / (4 + epsilon).

    Also checks basic validity (no duplicate items, all items/agents valid).
    """
    agents = list(instance.agents)
    items = list(instance.items)
    alpha = 4 + epsilon
    required = opt / alpha

    assert isinstance(result, dict)
    assert set(result.keys()) == set(agents), \
        f"Result agents {set(result.keys())} != expected {set(agents)}"

    all_allocated = set()
    for agent, bundle in result.items():
        for item in bundle:
            assert item in items, f"Unknown item {item} in allocation"
            assert item not in all_allocated, \
                f"Item {item} assigned to multiple agents"
            all_allocated.add(item)

    for agent, bundle in result.items():
        value = sum(instance.agent_item_value(agent, item) for item in bundle)
        assert value >= required - 1e-6, \
            f"Agent {agent} has value {value:.4f}, expected >= OPT/(4+eps) = {required:.4f} " \
            f"(OPT={opt:.4f}, epsilon={epsilon})"


def make_restricted_valuations(agents, items, item_sizes, eligible):
    """
    Build valuations for the restricted model.

    Args:
        agents: list of agent names
        items: list of item names
        item_sizes: dict mapping item -> its fixed size p_j
        eligible: dict mapping agent -> set of items the agent can use

    Returns:
        valuations dict where agent values item at p_j if eligible, else 0.
    """
    valuations = {}
    for agent in agents:
        valuations[agent] = {}
        for item in items:
            if item in eligible.get(agent, set()):
                valuations[agent][item] = item_sizes[item]
            else:
                valuations[agent][item] = 0
    return valuations


# ==============================================================================
#  Tests with approximation guarantee checks
# ==============================================================================

def test_example_1_easy():
    """
    Single agent, single item (size 10). OPT = 10.
    p0 is eligible for r1.
    """
    agents = ["p0"]
    items = ["r1"]
    item_sizes = {"r1": 10}
    eligible = {"p0": {"r1"}}
    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)

    epsilon = 0.1
    opt = brute_force_max_min(instance)  # 10
    result = qp_max_min_allocation(instance, epsilon=epsilon)
    assert result == {'p0': {'r1'}}
    assert_approximation(instance, result, epsilon, opt)


def test_example_2_simple():
    """
    r1 has size 10, r2 has size 5, r3 has size 10.
    p0 eligible for {r1, r2}, p1 eligible for {r2, r3}.
    OPT = 10 (p0={r1}=10, p1={r2,r3}=15 or p0={r1,r2}=15, p1={r3}=10).
    Algorithm must give each agent >= 10/4.1 ≈ 2.44.
    """
    agents = ["p0", "p1"]
    items = ["r1", "r2", "r3"]
    item_sizes = {"r1": 10, "r2": 5, "r3": 10}
    eligible = {
        "p0": {"r1", "r2"},
        "p1": {"r2", "r3"},
    }
    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)

    epsilon = 0.1
    opt = brute_force_max_min(instance)
    result = qp_max_min_allocation(instance, epsilon=epsilon)
    assert_approximation(instance, result, epsilon, opt)


def test_example_3_collapse():
    """
    r1 size=3, r2 size=3, r3 size=10.
    p0 eligible for {r1, r2, r3}, p1 eligible for {r1, r2}.
    OPT = 6 (p0={r3}=10, p1={r1,r2}=6).
    Algorithm must give each agent >= 6/4.1 ≈ 1.46.
    """
    agents = ["p0", "p1"]
    items = ["r1", "r2", "r3"]
    item_sizes = {"r1": 3, "r2": 3, "r3": 10}
    eligible = {
        "p0": {"r1", "r2", "r3"},
        "p1": {"r1", "r2"},
    }
    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)

    epsilon = 0.1
    opt = brute_force_max_min(instance)
    result = qp_max_min_allocation(instance, epsilon=epsilon)
    assert_approximation(instance, result, epsilon, opt)


# ==============================================================================
#  Edge Cases
# ==============================================================================

def test_edge_case_minimal_input():
    """
    Minimal valid instance (1 agent, 1 item with size 0).
    OPT = 0, so any allocation is valid.
    """
    agents = ["p0"]
    items = ["r1"]
    item_sizes = {"r1": 0}
    eligible = {"p0": set()}
    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)

    result = qp_max_min_allocation(instance, epsilon=0.1)
    assert result == {"p0": set()}


def test_large_input():
    """
    Restricted large instance (10 agents, 30 items).
    Each item has a random fixed size. Each agent is eligible for their
    3 zone items plus some random others.
    """
    from fairpyx.algorithms.qp_local_search import logger
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    agents = [f"p{i}" for i in range(10)]
    items = [f"r{j}" for j in range(30)]

    epsilon = 0.1
    rng = random.Random(42)

    # Each item has a single fixed size
    item_sizes = {item: rng.randint(5, 20) for item in items}

    # Each agent is eligible for their 3 zone items + random others
    eligible = {}
    for idx, agent in enumerate(agents):
        agent_items = set(items[idx * 3: idx * 3 + 3])  # zone items
        for jdx, item in enumerate(items):
            if jdx < idx * 3 or jdx >= idx * 3 + 3:
                if rng.random() < 0.3:
                    agent_items.add(item)
        eligible[agent] = agent_items

    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)
    result = qp_max_min_allocation(instance, epsilon=epsilon)

    # Lower bound on OPT: assign each agent their zone items
    opt_lower_bound = min(
        sum(item_sizes[items[j]] for j in range(i * 3, i * 3 + 3))
        for i in range(10)
    )
    alpha = 4 + epsilon
    required = opt_lower_bound / alpha

    # Validity checks
    assert isinstance(result, dict)
    all_allocated_items = set()
    for agent, bundle in result.items():
        assert agent in agents, f"Unknown agent {agent}"
        for item in bundle:
            assert item not in all_allocated_items, \
                f"Item {item} assigned to multiple agents"
            all_allocated_items.add(item)
        assert all(item in items for item in bundle)

    # Approximation check against lower bound
    for agent, bundle in result.items():
        value = sum(instance.agent_item_value(agent, item) for item in bundle)
        assert value >= required - 1e-6, \
            f"Agent {agent} has value {value:.4f}, expected >= {required:.4f} " \
            f"(OPT_lb={opt_lower_bound}, epsilon={epsilon})"


# ==============================================================================
#  Allocation tests with brute-force OPT verification
# ==============================================================================

def test_max_min_allocation():
    """
    3 agents, 4 items.
    r1 size=10, r2 size=5, r3 size=8, r4 size=3.
    p0 eligible for {r1, r2}, p1 eligible for {r2, r3, r4}, p2 eligible for {r3, r4}.
    """
    agents = ["p0", "p1", "p2"]
    items = ["r1", "r2", "r3", "r4"]
    item_sizes = {"r1": 10, "r2": 5, "r3": 8, "r4": 3}
    eligible = {
        "p0": {"r1", "r2"},
        "p1": {"r2", "r3", "r4"},
        "p2": {"r3", "r4"},
    }
    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)

    epsilon = 0.1
    opt = brute_force_max_min(instance)
    result = qp_max_min_allocation(instance, epsilon=epsilon)
    assert_approximation(instance, result, epsilon, opt)


def test_max_min_allocation_larger():
    """
    2 agents, 4 items with symmetric eligibility.
    r1 size=8, r2 size=6, r3 size=6, r4 size=8.
    p0 eligible for {r1, r2}, p1 eligible for {r3, r4}.
    OPT = 14 (p0={r1,r2}=14, p1={r3,r4}=14).
    """
    agents = ["p0", "p1"]
    items = ["r1", "r2", "r3", "r4"]
    item_sizes = {"r1": 8, "r2": 6, "r3": 6, "r4": 8}
    eligible = {
        "p0": {"r1", "r2"},
        "p1": {"r3", "r4"},
    }
    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)

    epsilon = 0.1
    opt = brute_force_max_min(instance)
    result = qp_max_min_allocation(instance, epsilon=epsilon)
    assert_approximation(instance, result, epsilon, opt)


def test_single_dominant_item():
    """
    One large item (r1, size=100) and small items (r2-r5, size=2 each).
    All agents eligible for all items.
    OPT = 2: with 4 agents and only 4 small items (size 2), the minimum is 2
    regardless of whether the large item is assigned. The large item cannot
    raise the bottleneck.
    """
    agents = ["p0", "p1", "p2", "p3"]
    items = ["r1", "r2", "r3", "r4", "r5"]
    item_sizes = {"r1": 100, "r2": 2, "r3": 2, "r4": 2, "r5": 2}
    eligible = {
        "p0": {"r1", "r2", "r3", "r4", "r5"},
        "p1": {"r1", "r2", "r3", "r4", "r5"},
        "p2": {"r1", "r2", "r3", "r4", "r5"},
        "p3": {"r1", "r2", "r3", "r4", "r5"},
    }
    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)

    epsilon = 0.1
    opt = brute_force_max_min(instance)
    result = qp_max_min_allocation(instance, epsilon=epsilon)
    assert_approximation(instance, result, epsilon, opt)


def test_all_agents_want_everything():
    """
    all agents eligible for all items.
    r1 size=12, r2 size=8, r3 size=6, r4 size=4. Total=30.
    OPT = 14 (e.g. p0={r1,r4}=16, p1={r2,r3}=14 → min=14).
    """
    agents = ["p0", "p1"]
    items = ["r1", "r2", "r3", "r4"]
    item_sizes = {"r1": 12, "r2": 8, "r3": 6, "r4": 4}
    eligible = {
        "p0": {"r1", "r2", "r3", "r4"},
        "p1": {"r1", "r2", "r3", "r4"},
    }
    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)

    epsilon = 0.1
    opt = brute_force_max_min(instance)
    result = qp_max_min_allocation(instance, epsilon=epsilon)
    assert_approximation(instance, result, epsilon, opt)


def test_disjoint_eligibility():
    """
    Agents have completely disjoint eligible sets.
    Each agent can only use their own items, so OPT is determined by
    the agent with the smallest total eligible value.
    """
    agents = ["p0", "p1", "p2"]
    items = ["r1", "r2", "r3", "r4", "r5", "r6"]
    item_sizes = {"r1": 7, "r2": 3, "r3": 5, "r4": 5, "r5": 4, "r6": 6}
    eligible = {
        "p0": {"r1", "r2"},       # total = 10
        "p1": {"r3", "r4"},       # total = 10
        "p2": {"r5", "r6"},       # total = 10
    }
    valuations = make_restricted_valuations(agents, items, item_sizes, eligible)
    instance = Instance(agents=agents, items=items, valuations=valuations)

    epsilon = 0.1
    opt = brute_force_max_min(instance)  # 10
    result = qp_max_min_allocation(instance, epsilon=epsilon)
    assert_approximation(instance, result, epsilon, opt)
