"""
Test the Fair Division Under Cardinality Constraints algorithm.

Programmer: Sapir Dahan
Since:  2026-04
"""

import math
import pytest
import numpy as np
import networkz as nz
from fairpyx import Instance, AllocationBuilder, divide
from fairpyx.algorithms.fair_division_under_cardinality_constraints import (
    fair_division_under_cardinality_constraints,
    greedy_round_robin,
    eliminate_envy_cycles,
    validate_fair_division_inputs,
)

NUM_OF_RANDOM_INSTANCES = 10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_ef1(result, valuations):
    """
    Return True iff the allocation satisfies EF1 (Envy-Free up to one good).

    EF1 definition: for every pair of agents (i, j), there EXISTS some good g in j's
    bundle such that when g is removed, i no longer envies j:
        v_i(result[i]) >= v_i(result[j] \\ {g})

    To check whether such a g exists, we look at the good in j's bundle that i values
    MOST. Removing the highest-valued item causes the largest drop in v_i(result[j]),
    giving us the best possible chance of eliminating the envy. If even removing that
    item is not enough, then no single removal can fix the envy and EF1 fails.

    Formally: EF1 holds for pair (i, j) iff
        v_i(result[i]) >= v_i(result[j]) - max_{g in result[j]} v_i(g)
    """
    agents = list(result.keys())
    for i in agents:
        # How much agent i values their own bundle
        val_i_own = sum(valuations[i].get(g, 0) for g in result[i])
        for j in agents:
            if i == j:
                continue
            # How much agent i values j's bundle
            val_i_other = sum(valuations[i].get(g, 0) for g in result[j])
            if val_i_other <= val_i_own:
                continue  # i does not envy j at all — EF1 trivially holds for this pair

            # i envies j, check EF1: can removing one good from j's bundle fix it?
            if not result[j]:
                # j has an empty bundle but i envies them — logically impossible, skip
                continue

            # The best single removal: take away the item i values most in j's bundle.
            # If val_i_own >= val_i_other - max_val, envy disappears → EF1 holds for (i, j).
            # If not, no single removal helps → EF1 is violated.
            max_val = max(valuations[i].get(g, 0) for g in result[j])
            if val_i_own < val_i_other - max_val:
                return False  # EF1 violated: even removing j's best item is not enough

    return True


def check_cardinality_constraints(result, item_categories, category_capacities):
    """
    Return True iff every agent's bundle respects the per-category threshold k_h.

    For each agent and each category h, count how many goods from that category the
    agent received. This count must not exceed k_h (the shared capacity for category h).
    """
    for agent, bundle in result.items():
        for category, items in item_categories.items():
            # Count how many goods from this category the agent received
            count = sum(1 for g in bundle if g in items)
            # k_h is the maximum allowed goods per agent from this category
            if count > category_capacities[category]:
                return False  # agent received too many goods from this category
    return True


def random_valid_instance(num_agents, num_items, num_categories, seed):
    """
    Generate a valid random instance for fair_division_under_cardinality_constraints.
    Returns (valuations, item_categories, category_capacities, agent_order).

    All generated instances satisfy the algorithm's preconditions:
      - goods are partitioned into exactly num_categories non-empty categories
      - each threshold k_h >= ceil(|C_h| / n)  (feasibility condition from the paper)
      - all valuations are non-negative integers
    """
    rng = np.random.default_rng(seed)

    # Build agent and good name lists.
    # Example: num_agents=3, num_items=6 → agents=['a1','a2','a3'], goods=['g1',...,'g6']
    agents = [f"a{i}" for i in range(1, num_agents + 1)]
    goods = [f"g{i}" for i in range(1, num_items + 1)]

    # --- Partition goods into num_categories non-empty categories ---
    # We want each category to contain a random subset of goods, not just a prefix.
    # Strategy: shuffle the goods list first, then cut it at (num_categories-1) random
    # positions to produce num_categories non-empty slices.
    #
    # Example with 6 goods and 3 categories:
    #   shuffled  = ['g4', 'g1', 'g6', 'g2', 'g5', 'g3']
    #   cut points chosen from {1,2,3,4,5}: say [2, 4]
    #   → full cut_points = [0, 2, 4, 6]
    #   → c1 = shuffled[0:2] = ['g4','g1']
    #     c2 = shuffled[2:4] = ['g6','g2']
    #     c3 = shuffled[4:6] = ['g5','g3']
    shuffled = list(goods)
    rng.shuffle(shuffled)

    # replace=False guarantees all cut points are distinct integers from {1,...,num_items-1},
    # so after sorting: each consecutive pair differs by >= 1 → every slice is non-empty.
    cut_points = sorted(rng.choice(range(1, num_items), num_categories - 1, replace=False).tolist())
    cut_points = [0] + cut_points + [num_items]
    item_categories = {
        f"c{h + 1}": shuffled[cut_points[h]:cut_points[h + 1]]
        for h in range(num_categories)
    }

    # --- Set thresholds: k_h >= ceil(|C_h| / n) ---
    # The paper requires n * k_h >= |C_h| so that all goods in category h can be
    # distributed (n agents, each taking at most k_h goods from h).
    # Rearranged: k_h >= ceil(|C_h| / n).
    #
    # Example: |C_h| = 5 goods, n = 3 agents → k_h >= ceil(5/3) = 2.
    #   With k_h=2 every agent takes at most 2 goods from c_h; 3*2=6 >= 5, so all goods fit.
    #   With k_h=1 only 3*1=3 goods could be allocated, leaving 2 goods unallocated — invalid.
    #
    # We add a small random slack of 0 or 1 so we also test non-tight (loose) thresholds.
    category_capacities = {
        cat: int(math.ceil(len(items) / num_agents)) + int(rng.integers(0, 2))
        for cat, items in item_categories.items()
    }

    # --- Generate non-negative integer valuations ---
    # Each agent independently assigns a random value in [0, 19] to every good.
    # Valuations are additive: an agent's value for a bundle is the sum over its goods.
    #
    # Example with 2 agents and 3 goods:
    #   valuations = {
    #       'a1': {'g1': 12, 'g2': 0, 'g3': 7},
    #       'a2': {'g1': 3,  'g2': 15, 'g3': 9},
    #   }
    valuations = {
        a: {g: int(rng.integers(0, 20)) for g in goods}
        for a in agents
    }

    # --- Shuffle agent order to avoid bias toward any fixed picking sequence ---
    # The initial_agent_order determines who picks first in the first category.
    # Shuffling here ensures tests cover many different starting orders, not always a1 first.
    agent_order = list(agents)
    rng.shuffle(agent_order)

    return valuations, item_categories, category_capacities, agent_order


# ---------------------------------------------------------------------------
# Section 1: fair_division_under_cardinality_constraints (Algorithm 1)
# ---------------------------------------------------------------------------

def test_algorithm1_doctest_instances():
    """Replicate the doctest examples 1-10 with exact expected outputs."""

    # Example 1: basic — 2 agents, 2 categories, k_h=1
    valuations = {
        'Agent1': {'m1': 9, 'm2': 3, 'm3': 8, 'm4': 2},
        'Agent2': {'m1': 3, 'm2': 9, 'm3': 2, 'm4': 8},
    }
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations=valuations),
        item_categories={'c1': ['m1', 'm2'], 'c2': ['m3', 'm4']},
        category_capacities={'c1': 1, 'c2': 1},
        initial_agent_order=['Agent1', 'Agent2'],
    )
    assert result == {'Agent1': ['m1', 'm3'], 'Agent2': ['m2', 'm4']}

    # Example 2: 3 agents, 2 categories of 3 goods each, k_h=1
    valuations = {
        'Agent1': {'m1': 9, 'm2': 6, 'm3': 3, 'm4': 8, 'm5': 5, 'm6': 2},
        'Agent2': {'m1': 3, 'm2': 9, 'm3': 6, 'm4': 2, 'm5': 8, 'm6': 5},
        'Agent3': {'m1': 6, 'm2': 3, 'm3': 9, 'm4': 5, 'm5': 2, 'm6': 8},
    }
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations=valuations),
        item_categories={'c1': ['m1', 'm2', 'm3'], 'c2': ['m4', 'm5', 'm6']},
        category_capacities={'c1': 1, 'c2': 1},
        initial_agent_order=['Agent1', 'Agent2', 'Agent3'],
    )
    assert result == {'Agent1': ['m1', 'm4'], 'Agent2': ['m2', 'm5'], 'Agent3': ['m3', 'm6']}

    # Example 3: single agent — trivially receives all goods
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations={'Alice': {'m1': 10, 'm2': 7, 'm3': 4}}),
        item_categories={'c1': ['m1', 'm2', 'm3']},
        category_capacities={'c1': 3},
        initial_agent_order=['Alice'],
    )
    assert result == {'Alice': ['m1', 'm2', 'm3']}

    # Example 4: single category — greedy_round_robin only, k_h=2
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations={'A': {'x': 10, 'y': 5, 'z': 1}, 'B': {'x': 1, 'y': 5, 'z': 10}}),
        item_categories={'c1': ['x', 'y', 'z']},
        category_capacities={'c1': 2},
        initial_agent_order=['A', 'B'],
    )
    assert result == {'A': ['x', 'y'], 'B': ['z']}

    # Example 5: single good per category — envy after c1 reverses order for c2
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations={'Agent1': {'m1': 8, 'm2': 5}, 'Agent2': {'m1': 5, 'm2': 8}}),
        item_categories={'c1': ['m1'], 'c2': ['m2']},
        category_capacities={'c1': 1, 'c2': 1},
        initial_agent_order=['Agent1', 'Agent2'],
    )
    assert result == {'Agent1': ['m1'], 'Agent2': ['m2']}

    # Example 6: asymmetric category sizes
    valuations = {
        'Agent1': {'m1': 10, 'm2': 8, 'm3': 6, 'm4': 4, 'm5': 9},
        'Agent2': {'m1': 4,  'm2': 6, 'm3': 8, 'm4': 10, 'm5': 7},
    }
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations=valuations),
        item_categories={'c1': ['m1', 'm2', 'm3', 'm4'], 'c2': ['m5']},
        category_capacities={'c1': 2, 'c2': 1},
        initial_agent_order=['Agent1', 'Agent2'],
    )
    assert result == {'Agent1': ['m1', 'm2', 'm5'], 'Agent2': ['m3', 'm4']}

    # Example 7: minimal base case — 1 agent, 1 good
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations={'a1': {'g1': 2}}),
        item_categories={'C1': ['g1']},
        category_capacities={'C1': 1},
        initial_agent_order=['a1'],
    )
    assert result == {'a1': ['g1']}

    # Example 8: 2 agents, 2 goods, 1 category, k_h=1
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations={'a1': {'g1': 2, 'g2': 3}, 'a2': {'g1': 1, 'g2': 4}}),
        item_categories={'C1': ['g1', 'g2']},
        category_capacities={'C1': 1},
        initial_agent_order=['a1', 'a2'],
    )
    assert result == {'a1': ['g2'], 'a2': ['g1']}

    # Example 9: 2 agents, 6 goods, 2 categories, k_h=2 — order reverses for C2
    valuations = {
        'a1': {'g1': 9, 'g2': 6, 'g3': 3, 'g4': 8, 'g5': 5, 'g6': 2},
        'a2': {'g1': 4, 'g2': 7, 'g3': 8, 'g4': 3, 'g5': 9, 'g6': 6},
    }
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations=valuations),
        item_categories={'C1': ['g1', 'g2', 'g3'], 'C2': ['g4', 'g5', 'g6']},
        category_capacities={'C1': 2, 'C2': 2},
        initial_agent_order=['a1', 'a2'],
    )
    assert result == {'a1': ['g1', 'g2', 'g4'], 'a2': ['g3', 'g5', 'g6']}

    # Example 10: 3 agents, 6 goods, 2 categories — multiple overlapping cycles
    valuations = {
        'a1': {'g1': 9, 'g2': 2, 'g3': 1, 'g4': 1, 'g5': 10, 'g6': 0},
        'a2': {'g1': 10, 'g2': 8, 'g3': 1, 'g4': 10, 'g5': 1, 'g6': 0},
        'a3': {'g1': 10, 'g2': 9, 'g3': 5, 'g4': 8, 'g5': 1, 'g6': 7},
    }
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations=valuations),
        item_categories={'C1': ['g1', 'g2', 'g3'], 'C2': ['g4', 'g5', 'g6']},
        category_capacities={'C1': 1, 'C2': 1},
        initial_agent_order=['a1', 'a2', 'a3'],
    )
    assert result == {'a1': ['g2', 'g5'], 'a2': ['g3', 'g4'], 'a3': ['g1', 'g6']}

    # Example 11: 4 agents, 26 goods, 4 categories, k_h=2 (large instance from the paper)
    # The topological sort uses lex secondary sort (by agent name), making the output fully
    # deterministic. Once the implementation runs, replace the property checks below with
    # an exact assertion (assert result11 == {...}).
    # For now we verify the three core guarantees:
    valuations_11 = {
        'a1': {'g1':9,'g2':4,'g3':8,'g4':1,'g5':6,'g6':2,'g7':10,'g8':3,'g9':5,'g10':1,'g11':7,
               'g12':6,'g13':2,'g14':9,'g15':4,'g16':8,'g17':1,'g18':3,
               'g19':5,'g20':7,'g21':2,'g22':10,'g23':4,'g24':6,'g25':1,'g26':8},
        'a2': {'g1':3,'g2':10,'g3':2,'g4':7,'g5':5,'g6':9,'g7':1,'g8':8,'g9':4,'g10':6,'g11':2,
               'g12':1,'g13':7,'g14':3,'g15':10,'g16':5,'g17':8,'g18':4,
               'g19':6,'g20':2,'g21':9,'g22':1,'g23':7,'g24':3,'g25':10,'g26':5},
        'a3': {'g1':8,'g2':1,'g3':5,'g4':9,'g5':2,'g6':4,'g7':6,'g8':10,'g9':1,'g10':7,'g11':3,
               'g12':9,'g13':5,'g14':2,'g15':6,'g16':1,'g17':10,'g18':4,
               'g19':8,'g20':3,'g21':6,'g22':2,'g23':9,'g24':1,'g25':5,'g26':7},
        'a4': {'g1':2,'g2':6,'g3':10,'g4':3,'g5':7,'g6':5,'g7':2,'g8':1,'g9':10,'g10':8,'g11':4,
               'g12':3,'g13':9,'g14':6,'g15':2,'g16':7,'g17':5,'g18':10,
               'g19':1,'g20':8,'g21':4,'g22':6,'g23':2,'g24':10,'g25':3,'g26':9},
    }
    item_categories_11 = {
        'C1': ['g1','g2','g3','g4','g5'],
        'C2': ['g6','g7','g8','g9','g10','g11'],
        'C3': ['g12','g13','g14','g15','g16','g17','g18'],
        'C4': ['g19','g20','g21','g22','g23','g24','g25','g26'],
    }
    category_capacities_11 = {'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2}
    all_goods_11 = [f'g{i}' for i in range(1, 27)]
    result11 = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=Instance(valuations=valuations_11),
        item_categories=item_categories_11,
        category_capacities=category_capacities_11,
        initial_agent_order=['a1', 'a2', 'a3', 'a4'],
    )
    assert sorted(g for bundle in result11.values() for g in bundle) == sorted(all_goods_11), \
        "Example 11: not all 26 goods were allocated"
    assert check_cardinality_constraints(result11, item_categories_11, category_capacities_11), \
        f"Example 11: cardinality constraint violated. result={result11}"
    assert check_ef1(result11, valuations_11), \
        f"Example 11: EF1 violated. result={result11}"
    # Exact output — hand-traced (no value ties anywhere, fully deterministic):
    # C1 σ=['a1','a2','a3','a4']: a1←g1,g5  a2←g2  a3←g4  a4←g3
    # Envy after C1: a3→a1 only. Topo sort → ['a2','a3','a4','a1']
    # C2 σ=['a2','a3','a4','a1']: a2←g6,g10  a3←g8,g11  a4←g9  a1←g7
    # No envy after C2. Topo sort → ['a1','a2','a3','a4']
    # C3 σ=['a1','a2','a3','a4']: a1←g14,g16  a2←g15,g13  a3←g17,g12  a4←g18
    # No envy after C3. Topo sort → ['a1','a2','a3','a4']
    # C4 σ=['a1','a2','a3','a4']: a1←g22,g26  a2←g25,g21  a3←g23,g19  a4←g24,g20
    assert result11 == {
        'a1': ['g1', 'g14', 'g16', 'g22', 'g26', 'g5', 'g7'],
        'a2': ['g10', 'g13', 'g15', 'g2', 'g21', 'g25', 'g6'],
        'a3': ['g11', 'g12', 'g17', 'g19', 'g23', 'g4', 'g8'],
        'a4': ['g18', 'g20', 'g24', 'g3', 'g9'],
    }


def test_algorithm1_all_goods_allocated():
    """Every good appears in exactly one bundle (no omissions, no duplicates)."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        # Generate a fresh random instance for each iteration using a deterministic seed,
        # so failures are reproducible.
        valuations, item_categories, category_capacities, agent_order = random_valid_instance(
            num_agents=4, num_items=20, num_categories=3, seed=i * 7
        )
        # Collect the ground-truth list of all goods across all categories.
        all_goods = [g for items in item_categories.values() for g in items]
        instance = Instance(valuations=valuations)
        result = divide(
            algorithm=fair_division_under_cardinality_constraints,
            instance=instance,
            item_categories=item_categories,
            category_capacities=category_capacities,
            initial_agent_order=agent_order,
        )
        # Flatten all bundles into one list and sort both sides before comparing,
        # so the check is order-independent.
        allocated = sorted([g for bundle in result.values() for g in bundle])
        assert allocated == sorted(all_goods), (
            f"Seed {i * 7}: allocated goods {allocated} != all goods {sorted(all_goods)}"
        )


def test_algorithm1_cardinality_constraints():
    """No agent receives more than k_h goods from any category."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        valuations, item_categories, category_capacities, agent_order = random_valid_instance(
            num_agents=3, num_items=15, num_categories=2, seed=i * 13
        )
        instance = Instance(valuations=valuations)
        result = divide(
            algorithm=fair_division_under_cardinality_constraints,
            instance=instance,
            item_categories=item_categories,
            category_capacities=category_capacities,
            initial_agent_order=agent_order,
        )
        # check_cardinality_constraints verifies that for every agent and every category h,
        # the number of goods from h in the agent's bundle does not exceed k_h.
        assert check_cardinality_constraints(result, item_categories, category_capacities), (
            f"Seed {i * 13}: cardinality constraint violated. result={result}"
        )


def test_algorithm1_ef1():
    """The algorithm guarantees EF1 (Biswas & Barman 2018, Theorem 1)."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        valuations, item_categories, category_capacities, agent_order = random_valid_instance(
            num_agents=4, num_items=20, num_categories=3, seed=i * 17
        )
        instance = Instance(valuations=valuations)
        result = divide(
            algorithm=fair_division_under_cardinality_constraints,
            instance=instance,
            item_categories=item_categories,
            category_capacities=category_capacities,
            initial_agent_order=agent_order,
        )
        # check_ef1 verifies that for every pair (i, j), removing the good i values most
        # from j's bundle eliminates i's envy — the EF1 guarantee from the paper.
        # We include the valuations dict so check_ef1 can look up values by agent and good name.
        assert check_ef1(result, valuations), (
            f"Seed {i * 17}: EF1 violated. valuations={valuations}, result={result}"
        )


def test_algorithm1_default_order():
    """Passing initial_agent_order=None defaults to sorted(agents), giving the same result."""
    valuations = {
        'Agent1': {'m1': 9, 'm2': 3, 'm3': 8, 'm4': 2},
        'Agent2': {'m1': 3, 'm2': 9, 'm3': 2, 'm4': 8},
    }
    instance = Instance(valuations=valuations)
    item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3', 'm4']}
    category_capacities = {'c1': 1, 'c2': 1}

    result_none = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=instance,
        item_categories=item_categories,
        category_capacities=category_capacities,
        initial_agent_order=None,
    )
    result_sorted = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=instance,
        item_categories=item_categories,
        category_capacities=category_capacities,
        initial_agent_order=sorted(valuations.keys()),
    )
    assert result_none == result_sorted


def test_algorithm1_topo_order_influences_next_category():
    """
    Example 9: after C1, a2 envies a1 (envy graph: a2→a1).
    Topo sort gives σ = ['a2', 'a1'] for C2, so a2 picks first in C2.
    a2 values C2 goods as: g4=3, g5=9, g6=6 → picks g5 (value 9) first.
    Verify a2's C2 allocation contains g5, proving topo sort influenced picking order.
    """
    valuations = {
        'a1': {'g1': 9, 'g2': 6, 'g3': 3, 'g4': 8, 'g5': 5, 'g6': 2},
        'a2': {'g1': 4, 'g2': 7, 'g3': 8, 'g4': 3, 'g5': 9, 'g6': 6},
    }
    instance = Instance(valuations=valuations)
    result = divide(
        algorithm=fair_division_under_cardinality_constraints,
        instance=instance,
        item_categories={'C1': ['g1', 'g2', 'g3'], 'C2': ['g4', 'g5', 'g6']},
        category_capacities={'C1': 2, 'C2': 2},
        initial_agent_order=['a1', 'a2'],
    )
    # a2 picked first in C2 (due to topo sort) → a2 must have g5 (a2's top C2 good, value 9)
    assert 'g5' in result['a2'], (
        f"Expected a2 to get g5 (top C2 good) since a2 envies a1 after C1 → picks first in C2. "
        f"Got: a2={result['a2']}"
    )
    # a1 should have gotten its best remaining C2 good: g4 (value 8) after a2 took g5
    assert 'g4' in result['a1'], (
        f"Expected a1 to get g4 after a2 took g5. Got: a1={result['a1']}"
    )


# ---------------------------------------------------------------------------
# Section 2: greedy_round_robin (Algorithm 2)
# ---------------------------------------------------------------------------

def test_greedy_picks_top_good():
    """Each agent greedily picks their personal highest-valued good."""
    valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 8}}
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    greedy_round_robin(alloc, ['m1', 'm2'], ['Alice', 'Bob'], {'c1': 1}, 'c1')
    assert alloc.sorted() == {'Alice': ['m1'], 'Bob': ['m2']}


def test_greedy_multiple_rounds():
    """k_h=2: each agent makes 2 picks across 2 rounds."""
    valuations = {
        'Alice': {'m1': 10, 'm2': 8, 'm3': 5, 'm4': 3},
        'Bob':   {'m1': 3,  'm2': 5, 'm3': 8, 'm4': 10},
    }
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    greedy_round_robin(alloc, ['m1', 'm2', 'm3', 'm4'], ['Alice', 'Bob'], {'c1': 2}, 'c1')
    assert alloc.sorted() == {'Alice': ['m1', 'm2'], 'Bob': ['m3', 'm4']}


def test_greedy_partial_rounds():
    """More agents than goods: only the first agents in order receive goods."""
    valuations = {
        'A': {'g1': 5, 'g2': 3},
        'B': {'g1': 3, 'g2': 5},
        'C': {'g1': 4, 'g2': 4},
    }
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    # Only 2 goods, 3 agents, k_h=1 → only first 2 agents in order get a good
    greedy_round_robin(alloc, ['g1', 'g2'], ['A', 'B', 'C'], {'c1': 1}, 'c1')
    sorted_result = alloc.sorted()
    # Both g1 and g2 must be allocated
    allocated = [g for bundle in sorted_result.values() for g in bundle]
    assert sorted(allocated) == ['g1', 'g2']
    # C (third in order) should get nothing
    assert sorted_result['C'] == []


def test_greedy_all_goods_allocated():
    """Every good in the category ends up in exactly one agent's bundle."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = np.random.default_rng(i * 5)
        n, m = 3, 9
        agents = [f"a{j}" for j in range(1, n + 1)]
        goods = [f"g{j}" for j in range(1, m + 1)]
        valuations = {a: {g: int(rng.integers(1, 15)) for g in goods} for a in agents}
        k_h = math.ceil(m / n)
        instance = Instance(valuations=valuations)
        alloc = AllocationBuilder(instance)
        greedy_round_robin(alloc, goods, agents, {'c1': k_h}, 'c1')
        allocated = sorted([g for bundle in alloc.bundles.values() for g in bundle])
        assert allocated == sorted(goods), f"Seed {i * 5}: not all goods allocated"


def test_greedy_respects_order():
    """
    The agent first in agent_order gets the globally most-valued good (if they value it highest).
    Reversing the order changes who gets the top good.
    """
    valuations = {'A': {'x': 10, 'y': 1}, 'B': {'x': 10, 'y': 1}}
    # Both value x=10 equally; whoever is first gets x.
    instance = Instance(valuations=valuations)

    alloc1 = AllocationBuilder(instance)
    greedy_round_robin(alloc1, ['x', 'y'], ['A', 'B'], {'c1': 1}, 'c1')
    assert 'x' in alloc1.bundles['A']  # A is first, gets x

    alloc2 = AllocationBuilder(instance)
    greedy_round_robin(alloc2, ['x', 'y'], ['B', 'A'], {'c1': 1}, 'c1')
    assert 'x' in alloc2.bundles['B']  # B is first, gets x


def test_greedy_tied_valuations():
    """Tied valuations must not crash; both goods must be allocated."""
    valuations = {'A': {'g1': 5, 'g2': 5}, 'B': {'g1': 5, 'g2': 5}}
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    greedy_round_robin(alloc, ['g1', 'g2'], ['A', 'B'], {'c1': 1}, 'c1')
    allocated = sorted([g for bundle in alloc.bundles.values() for g in bundle])
    assert allocated == ['g1', 'g2']


# ---------------------------------------------------------------------------
# Section 3: eliminate_envy_cycles
# ---------------------------------------------------------------------------

def test_eliminate_no_cycle():
    """No envy: graph is already a DAG, bundles are unchanged."""
    valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 9}}
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    alloc.give('Alice', 'm1')
    alloc.give('Bob', 'm2')
    G = eliminate_envy_cycles(alloc)
    assert list(nz.simple_cycles(G)) == []
    assert sorted(alloc.bundles['Alice']) == ['m1']
    assert sorted(alloc.bundles['Bob']) == ['m2']


def test_eliminate_2agent_cycle():
    """2-agent mutual envy cycle: bundles are swapped."""
    valuations = {'Alice': {'m1': 3, 'm2': 7}, 'Bob': {'m1': 7, 'm2': 3}}
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    alloc.give('Alice', 'm1')
    alloc.give('Bob', 'm2')
    G = eliminate_envy_cycles(alloc)
    assert list(nz.simple_cycles(G)) == []
    assert sorted(alloc.bundles['Alice']) == ['m2']
    assert sorted(alloc.bundles['Bob']) == ['m1']


def test_eliminate_3agent_cycle():
    """3-agent cycle: bundle rotation resolves all envy."""
    valuations = {
        'A': {'m1': 3, 'm2': 5, 'm3': 1},
        'B': {'m1': 1, 'm2': 3, 'm3': 5},
        'C': {'m1': 5, 'm2': 1, 'm3': 3},
    }
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    alloc.give('A', 'm1')
    alloc.give('B', 'm2')
    alloc.give('C', 'm3')
    G = eliminate_envy_cycles(alloc)
    assert list(nz.simple_cycles(G)) == []
    assert sorted(alloc.bundles['A']) == ['m2']
    assert sorted(alloc.bundles['B']) == ['m3']
    assert sorted(alloc.bundles['C']) == ['m1']


def test_eliminate_result_is_dag():
    """After eliminate_envy_cycles, the returned graph always has no directed cycles."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = np.random.default_rng(i * 3)
        n, m = 4, 8
        agents = [f"a{j}" for j in range(1, n + 1)]
        goods = [f"g{j}" for j in range(1, m + 1)]
        valuations = {a: {g: int(rng.integers(0, 15)) for g in goods} for a in agents}
        instance = Instance(valuations=valuations)
        alloc = AllocationBuilder(instance)
        # Assign goods randomly to agents
        shuffled = list(goods)
        rng.shuffle(shuffled)
        for idx, good in enumerate(shuffled):
            alloc.give(agents[idx % n], good)
        G = eliminate_envy_cycles(alloc)
        assert list(nz.simple_cycles(G)) == [], f"Seed {i * 3}: cycle found in result graph"


def test_eliminate_no_value_decrease():
    """Lemma 1: after cycle elimination, no agent's total valuation decreases."""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        rng = np.random.default_rng(i * 11)
        n, m = 4, 8
        agents = [f"a{j}" for j in range(1, n + 1)]
        goods = [f"g{j}" for j in range(1, m + 1)]
        valuations = {a: {g: int(rng.integers(0, 15)) for g in goods} for a in agents}
        instance = Instance(valuations=valuations)
        alloc = AllocationBuilder(instance)
        shuffled = list(goods)
        rng.shuffle(shuffled)
        for idx, good in enumerate(shuffled):
            alloc.give(agents[idx % n], good)
        # Record values before
        value_before = {
            a: sum(instance.agent_item_value(a, g) for g in alloc.bundles[a])
            for a in agents
        }
        eliminate_envy_cycles(alloc)
        value_after = {
            a: sum(instance.agent_item_value(a, g) for g in alloc.bundles[a])
            for a in agents
        }
        for a in agents:
            assert value_after[a] >= value_before[a], (
                f"Seed {i * 11}: agent {a} value decreased from {value_before[a]} to {value_after[a]}"
            )


def test_eliminate_complex_overlapping():
    """
    Example 10 (after C2, before final cycle elimination):
    Initial bundles: a1=[g2,g5], a2=[g1,g6], a3=[g3,g4].
    Envy edges: a1→a2, a2→a1, a2→a3, a3→a1 (two overlapping cycles).
    After elimination, result must be a DAG with the final EF1 bundles.
    """
    valuations = {
        'a1': {'g1': 9, 'g2': 2, 'g3': 1, 'g4': 1, 'g5': 10, 'g6': 0},
        'a2': {'g1': 10, 'g2': 8, 'g3': 1, 'g4': 10, 'g5': 1, 'g6': 0},
        'a3': {'g1': 10, 'g2': 9, 'g3': 5, 'g4': 8, 'g5': 1, 'g6': 7},
    }
    instance = Instance(valuations=valuations)
    alloc = AllocationBuilder(instance)
    # Set up bundles directly to mirror the state after C2 round-robin
    for g in ['g2', 'g5']:
        alloc.give('a1', g)
    for g in ['g1', 'g6']:
        alloc.give('a2', g)
    for g in ['g3', 'g4']:
        alloc.give('a3', g)
    G = eliminate_envy_cycles(alloc)
    assert list(nz.simple_cycles(G)) == []
    # Both valid elimination paths lead to the same allocation
    assert sorted(alloc.bundles['a1']) == ['g2', 'g5']
    assert sorted(alloc.bundles['a2']) == ['g3', 'g4']
    assert sorted(alloc.bundles['a3']) == ['g1', 'g6']


# ---------------------------------------------------------------------------
# Section 4: validate_fair_division_inputs
# ---------------------------------------------------------------------------

def _make_alloc(valuations):
    """Helper: construct an AllocationBuilder from a valuations dict."""
    return AllocationBuilder(Instance(valuations=valuations))


def test_validate_valid_inputs():
    """Valid inputs do not raise any exception."""
    alloc = _make_alloc({'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}})
    validate_fair_division_inputs(
        alloc,
        item_categories={'c1': ['m1', 'm2']},
        category_capacities={'c1': 1},
        initial_agent_order=['Alice', 'Bob'],
    )  # must not raise


def test_validate_boundary_k_h():
    """k_h == ceil(|C_h| / n) is the minimum valid threshold — must not raise."""
    # 3 goods, 2 agents → k_h >= ceil(3/2) = 2
    alloc = _make_alloc({
        'Alice': {'m1': 5, 'm2': 3, 'm3': 1},
        'Bob':   {'m1': 1, 'm2': 3, 'm3': 5},
    })
    validate_fair_division_inputs(
        alloc,
        item_categories={'c1': ['m1', 'm2', 'm3']},
        category_capacities={'c1': 2},  # exactly ceil(3/2)
        initial_agent_order=['Alice', 'Bob'],
    )  # must not raise


def test_validate_empty_order():
    """Empty initial_agent_order while instance has agents → ValueError."""
    alloc = _make_alloc({'Alice': {'m1': 5}, 'Bob': {'m1': 3}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ['m1']}, {'c1': 1}, [])


def test_validate_duplicate_in_order():
    """Duplicate agent in initial_agent_order → ValueError."""
    alloc = _make_alloc({'Alice': {'m1': 5}, 'Bob': {'m1': 3}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ['m1']}, {'c1': 1}, ['Alice', 'Alice'])


def test_validate_item_in_two_categories():
    """Same item listed in two categories → ValueError."""
    alloc = _make_alloc({'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['m1', 'm2'], 'c2': ['m1']},
            {'c1': 1, 'c2': 1},
            ['Alice', 'Bob'],
        )


def test_validate_item_missing_from_categories():
    """Item in valuations but absent from all categories → ValueError."""
    # alloc has m1 and m2 in instance, but categories only cover m1
    alloc = _make_alloc({'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ['m1']}, {'c1': 1}, ['Alice', 'Bob'])


def test_validate_missing_threshold():
    """Category in item_categories has no entry in category_capacities → ValueError."""
    alloc = _make_alloc({'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['m1'], 'c2': ['m2']},
            {'c1': 1},           # missing 'c2'
            ['Alice', 'Bob'],
        )


def test_validate_extra_threshold():
    """Key in category_capacities with no matching category in item_categories → ValueError."""
    alloc = _make_alloc({'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['m1', 'm2']},
            {'c1': 1, 'c99': 2},  # 'c99' not in item_categories
            ['Alice', 'Bob'],
        )


def test_validate_k_h_zero():
    """k_h = 0 is not a positive integer → ValueError."""
    alloc = _make_alloc({'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(alloc, {'c1': ['m1', 'm2']}, {'c1': 0}, ['Alice', 'Bob'])


def test_validate_k_h_too_small():
    """k_h < ceil(|C_h| / n) violates the feasibility condition → ValueError."""
    # 3 goods, 2 agents → k_h must be >= ceil(3/2) = 2; k_h=1 is too small
    alloc = _make_alloc({
        'Alice': {'m1': 5, 'm2': 3, 'm3': 1},
        'Bob':   {'m1': 1, 'm2': 3, 'm3': 5},
    })
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['m1', 'm2', 'm3']},
            {'c1': 1},
            ['Alice', 'Bob'],
        )


def test_validate_negative_valuation():
    """Negative valuation value → ValueError."""
    alloc = _make_alloc({'Alice': {'m1': -1, 'm2': 3}, 'Bob': {'m1': 4, 'm2': 6}})
    with pytest.raises(ValueError):
        validate_fair_division_inputs(
            alloc,
            {'c1': ['m1', 'm2']},
            {'c1': 1},
            ['Alice', 'Bob'],
        )


# ---------------------------------------------------------------------------
# Large random integration tests (Algorithm 1, varying sizes)
# ---------------------------------------------------------------------------

def test_random_integration_small():
    """3-5 agents, 10-20 goods, 2-4 categories: EF1 and cardinality must hold."""
    configs = [
        (3, 12, 2, 101),
        (4, 16, 3, 202),
        (5, 20, 4, 303),
        (3, 10, 2, 404),
        (4, 15, 3, 505),
    ]
    for num_agents, num_items, num_categories, seed in configs:
        valuations, item_categories, category_capacities, agent_order = random_valid_instance(
            num_agents, num_items, num_categories, seed
        )
        instance = Instance(valuations=valuations)
        result = divide(
            algorithm=fair_division_under_cardinality_constraints,
            instance=instance,
            item_categories=item_categories,
            category_capacities=category_capacities,
            initial_agent_order=agent_order,
        )
        assert check_cardinality_constraints(result, item_categories, category_capacities), (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): cardinality violated"
        )
        assert check_ef1(result, valuations), (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): EF1 violated"
        )


def test_random_integration_large():
    """6-8 agents, 30-50 goods, 4-6 categories: EF1 and cardinality must hold."""
    configs = [
        (6, 30, 4, 1001),
        (7, 35, 5, 2002),
        (8, 50, 6, 3003),
    ]
    for num_agents, num_items, num_categories, seed in configs:
        valuations, item_categories, category_capacities, agent_order = random_valid_instance(
            num_agents, num_items, num_categories, seed
        )
        instance = Instance(valuations=valuations)
        result = divide(
            algorithm=fair_division_under_cardinality_constraints,
            instance=instance,
            item_categories=item_categories,
            category_capacities=category_capacities,
            initial_agent_order=agent_order,
        )
        assert check_cardinality_constraints(result, item_categories, category_capacities), (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): cardinality violated"
        )
        assert check_ef1(result, valuations), (
            f"Config ({num_agents},{num_items},{num_categories},seed={seed}): EF1 violated"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])